"""The STML judge for logic worlds: relational GNN over the premise graph, read out at the claim's two kinds.

  logits = MLP([h_subject, h_object, mean-pool(h), max-pool(h)])  -> follows / contradicted / open
Calibration: temperature scaling fitted on validation items (a single scalar; it cannot change the argmax).
Abstention: commit only if the calibrated probability of the best answer exceeds (lam - k) / (1 + lam).
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metacog import Item as MItem
from ..metacog import commit_threshold, metrics
from .logic_data import EDGE_TYPES, NODE_DIM, TRUTHS, Item
from .network import GraphBatch, RelationalGNN


class LogicJudge(nn.Module):
    def __init__(self, d: int = 64, layers: int = 4, gnn: nn.Module | None = None, n_types: int | None = None,
                 node_dim: int | None = None):
        super().__init__()
        self.gnn = gnn if gnn is not None else RelationalGNN(node_dim or NODE_DIM, n_types or len(EDGE_TYPES), d, layers)
        self.head = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Linear(d, len(TRUTHS)))
        # "why" head: for every premise, is it part of the proof? (reads the premise's two kinds, its type, the claim's kinds)
        self.etype = nn.Embedding(len(EDGE_TYPES), d)
        self.core_head = nn.Sequential(nn.Linear(5 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, items: list[Item], device, return_core: bool = False, return_h: bool = False):
        b = GraphBatch.build([it.graph for it in items], device)
        h = self.gnn.nodes(b)
        off = np.cumsum([0] + [it.n for it in items])[:-1]
        ia = torch.as_tensor(off + [it.claim.a for it in items], device=device)
        ib = torch.as_tensor(off + [it.claim.b for it in items], device=device)
        n = h.shape[0]
        cnt = torch.zeros(b.n_graphs, device=device).index_add_(0, b.graph, torch.ones(n, device=device))
        mean = torch.zeros(b.n_graphs, h.shape[1], device=device).index_add_(0, b.graph, h) / cnt.unsqueeze(1)
        dense = torch.full((b.n_graphs, b.max_nodes, h.shape[1]), -1e4, device=device)
        dense[b.graph, b.pos] = h
        out = self.head(torch.cat([h[ia], h[ib], mean, dense.max(1).values], 1))
        if return_h:
            return out, h, b
        if not return_core:
            return out
        eoff = np.cumsum([0] + [len(it.graph.edges) for it in items])[:-1]
        idx = np.concatenate([eoff[g] + 2 * np.arange(it.n_prem) for g, it in enumerate(items)]).astype(np.int64)
        owner = np.concatenate([np.full(it.n_prem, g) for g, it in enumerate(items)])
        idx_t, own_t = torch.as_tensor(idx, device=device), torch.as_tensor(owner, device=device)
        z = torch.cat([h[b.src[idx_t]], h[b.dst[idx_t]], self.etype(b.etype[idx_t]), h[ia][own_t], h[ib][own_t]], 1)
        return out, self.core_head(z).squeeze(1)


def fit(model: LogicJudge, train: list[Item], val: list[Item], device, epochs: int = 20, lr: float = 1e-3,
        batch: int = 128, seed: int = 0, log=print, why_weight: float = 0.0, weight_decay: float = 1e-4) -> LogicJudge:
    """Cross-entropy; keeps the epoch with the best validation loss."""
    rnd = random.Random(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, state = 1e9, None
    for ep in range(epochs):
        model.train()
        order = train[:]
        rnd.shuffle(order)
        for i in range(0, len(order), batch):
            chunk = order[i:i + batch]
            y = torch.as_tensor([it.label for it in chunk], device=device)
            if why_weight > 0:
                logits, core_logits = model(chunk, device, return_core=True)
                target = torch.zeros(sum(it.n_prem for it in chunk), device=device)
                off = 0
                for it in chunk:
                    for j in it.core:
                        target[off + j] = 1.0
                    off += it.n_prem
                loss = F.cross_entropy(logits, y) + why_weight * F.binary_cross_entropy_with_logits(core_logits, target)
            else:
                loss = F.cross_entropy(model(chunk, device), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = float(F.cross_entropy(logits_of(model, val, device), torch.as_tensor([it.label for it in val], device=device)))
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"epoch {ep + 1}/{epochs}: val loss {vl:.3f}")
    model.load_state_dict(state)
    return model


@torch.no_grad()
def logits_of(model: LogicJudge, items: list[Item], device, batch: int = 512) -> torch.Tensor:
    model.eval()
    return torch.cat([model(items[i:i + batch], device) for i in range(0, len(items), batch)])


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    t = torch.zeros(1, requires_grad=True, device=logits.device)       # log-temperature
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / t.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.exp())


def ece(conf: np.ndarray, right: np.ndarray, bins: int = 10) -> float:
    b = np.clip((conf * bins).astype(int), 0, bins - 1)
    return float(sum(abs(conf[b == k].mean() - right[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


def answer_with_abstention(probs: np.ndarray, items: list[Item], lam: float, k: float = 0.2) -> dict:
    """probs: [N, 3] calibrated. Commit iff max prob > threshold, else 'unknown'."""
    thr = commit_threshold(lam, k)
    ans = ["true", "false", "open"]
    out = []
    for p, it in zip(probs, items):
        j = int(p.argmax())
        out.append(MItem(ans[j] if p[j] > thr else "unknown", TRUTHS[it.label], float(p[j]), float(it.support)))
    return metrics(out, lam, k)


@torch.no_grad()
def judge_with_why(model: LogicJudge, items: list[Item], device, batch: int = 256):
    """Logits [N,3] and, per item, the probability that each premise is part of the proof."""
    model.eval()
    lg, cp = [], []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        o, c = model(chunk, device, return_core=True)
        lg.append(o)
        c = torch.sigmoid(c).cpu().numpy()
        off = 0
        for it in chunk:
            cp.append(c[off:off + it.n_prem])
            off += it.n_prem
    return torch.cat(lg), cp


@torch.no_grad()
def certificate_scores(model: LogicJudge, items: list[Item], device, temp: float = 1.0, max_nec: int = 8) -> dict:
    """The learner's own check of its answer (no referee): re-judge the claim on the premises it named.

    For a verdict follows / contradicted:
      sufficiency  P(same verdict) when ONLY the premises the judge named as its proof are kept
      necessity    1 - max over named premises of P(same verdict) when that premise is removed
                   (a real proof should break without any of its steps)
    For a verdict open (a counterexample exists, nothing to name): both scores are 1 - the largest
    probability the judge puts on any premise being 'part of a proof'.
    Also returns the predicted proof length. Higher scores = more reason to commit.
    """
    from .logic_data import sub_item
    lg, cp = judge_with_why(model, items, device)
    probs = torch.softmax(lg / temp, 1).cpu().numpy()
    verdict = probs.argmax(1)
    subs, owner, kind = [], [], []
    named = []
    for g, (it, p) in enumerate(zip(items, cp)):
        order = np.argsort(-p)
        C = [int(j) for j in order if p[j] > 0.5]
        named.append(len(C))
        if verdict[g] == 2:
            continue
        subs.append(sub_item(it, sorted(C)))
        owner.append(g)
        kind.append(-1)
        for j in C[:max_nec]:
            subs.append(sub_item(it, [k for k in range(it.n_prem) if k != j]))
            owner.append(g)
            kind.append(j)
    sp = np.zeros((0, 3))
    if subs:
        sp = torch.softmax(logits_of(model, subs, device) / temp, 1).cpu().numpy()
    suff = np.ones(len(items))
    nec = np.ones(len(items))
    worst = np.zeros(len(items))
    for k, (g, kd) in enumerate(zip(owner, kind)):
        pv = sp[k, verdict[g]]
        if kd == -1:
            suff[g] = pv
        else:
            worst[g] = max(worst[g], pv)
    for g in range(len(items)):
        if verdict[g] == 2:
            quiet = 1.0 - (cp[g].max() if len(cp[g]) else 0.0)
            suff[g] = nec[g] = quiet
        else:
            nec[g] = 1.0 - worst[g]
    return {"probs": probs, "suff": suff, "nec": nec, "cert": suff * nec, "named": np.array(named)}

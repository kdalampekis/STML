"""The STML learner: relational graph encoder + causal history transformer + heads.

  graph encoder      R-GCN-style message passing over typed edges; one embedding per graph.
  task encoder       for ARC, every (input, output) example pair -> one vector;
                     the mean over pairs is the task token.
  candidate tokens   each possible answer, read in the light of the task and the test input.
  history transformer  causal attention over [task token, attempt 1, attempt 2, ...], each
                     attempt = the answer tried + the feedback it got + its position.
                     This is where learning from one's own mistakes happens in context.
  heads              choice head: a score per candidate answer (what to try next);
                     confidence head: P(this answer is right) (what it knows, for H3).

Plain PyTorch, no graph library, so it runs on Apple Silicon (Metal / 'mps'),
CUDA or CPU. Use pick_device() to choose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphs import Graph


def pick_device(prefer: str | None = None) -> torch.device:
    """Apple Metal first, then CUDA, then CPU. Override with prefer='cpu' etc. or the environment variable STML_DEVICE (used when the GPU is throttled on battery)."""
    prefer = prefer or os.environ.get("STML_DEVICE")
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- batching graphs

@dataclass
class GraphBatch:
    x: torch.Tensor          # [N, node_dim]
    src: torch.Tensor        # [E]
    dst: torch.Tensor        # [E]
    etype: torch.Tensor      # [E]
    graph: torch.Tensor      # [N] graph index of every node
    n_graphs: int
    max_nodes: int
    pos: torch.Tensor        # [N] position of every node inside its graph

    @staticmethod
    def build(graphs: list[Graph], device: torch.device) -> "GraphBatch":
        xs, src, dst, et, gi, pos = [], [], [], [], [], []
        off = 0
        for k, g in enumerate(graphs):
            xs.append(g.x)
            if len(g.edges):
                src.append(g.edges[:, 0] + off)
                dst.append(g.edges[:, 1] + off)
                et.append(g.edges[:, 2])
            gi.append(np.full(g.n, k))
            pos.append(np.arange(g.n))
            off += g.n
        cat = lambda a, dt: torch.as_tensor(np.concatenate(a) if a else np.zeros(0), dtype=dt, device=device)
        return GraphBatch(cat(xs, torch.float32).reshape(off, -1), cat(src, torch.long), cat(dst, torch.long),
                          cat(et, torch.long), cat(gi, torch.long), len(graphs),
                          max((g.n for g in graphs), default=1), cat(pos, torch.long))


class RelationalGNN(nn.Module):
    """h_v <- h_v + ReLU(LN(W0 h_v + sum_r mean_{u in N_r(v)} W_r h_u)), then mean+max pooling."""

    def __init__(self, node_dim: int, n_types: int, d: int = 128, layers: int = 3):
        super().__init__()
        self.inp = nn.Linear(node_dim, d)
        self.self_w = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.rel_w = nn.ModuleList(nn.Linear(d, d * n_types, bias=False) for _ in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.out = nn.Linear(2 * d, d)
        self.n_types, self.d = n_types, d

    def nodes(self, b: GraphBatch) -> torch.Tensor:
        """Node states after message passing."""
        h = F.relu(self.inp(b.x))
        n = h.shape[0]
        for ws, wr, ln in zip(self.self_w, self.rel_w, self.norm):
            msg = torch.zeros(n, self.d, device=h.device)
            if b.src.numel():
                allrel = wr(h).view(n, self.n_types, self.d)            # every node, every relation
                m = allrel[b.src, b.etype]                               # [E, d]
                key = b.dst * self.n_types + b.etype
                cnt = torch.zeros(n * self.n_types, device=h.device).index_add_(
                    0, key, torch.ones_like(key, dtype=torch.float32))
                summed = torch.zeros(n * self.n_types, self.d, device=h.device).index_add_(0, key, m)
                mean = summed / cnt.clamp(min=1).unsqueeze(1)
                msg = mean.view(n, self.n_types, self.d).sum(1)
            h = h + F.relu(ln(ws(h) + msg))
        return h

    def forward(self, b: GraphBatch) -> torch.Tensor:
        h = self.nodes(b)
        n = h.shape[0]
        # pooling: mean and max per graph
        counts = torch.zeros(b.n_graphs, device=h.device).index_add_(0, b.graph, torch.ones(n, device=h.device))
        mean = torch.zeros(b.n_graphs, self.d, device=h.device).index_add_(0, b.graph, h) / counts.clamp(min=1).unsqueeze(1)
        dense = torch.full((b.n_graphs, b.max_nodes, self.d), -1e4, device=h.device)
        dense[b.graph, b.pos] = h
        mx = dense.max(1).values
        return self.out(torch.cat([mean, mx], 1))


# ----------------------------------------------------------------------------- the learner

class STMLLearner(nn.Module):
    def __init__(self, node_dim: int, n_types: int, d: int = 128, gnn_layers: int = 3,
                 tf_layers: int = 2, heads: int = 4, max_attempts: int = 8, dropout: float = 0.1,
                 out_match: bool = False, hand_dim: int = 0, wide: bool = False, gnn: nn.Module | None = None):
        super().__init__()
        self.gnn = gnn if gnn is not None else RelationalGNN(node_dim, n_types, d, gnn_layers)
        self.pair = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.task_in = nn.Linear(2 * d, d)
        self.out_match = out_match
        self.cand = nn.Sequential(nn.Linear((4 if out_match else 3) * d, d), nn.ReLU(), nn.Linear(d, d))
        self.hand = nn.Sequential(nn.Linear(hand_dim, d), nn.ReLU(), nn.Linear(d, d)) if hand_dim else None
        self.feedback = nn.Embedding(3, d)          # 0 wrong, 1 right, 2 = task token
        self.position = nn.Embedding(max_attempts + 1, d)
        self.episode = nn.Embedding(8, d)          # how many puzzles ago (1 = the previous one); used only with context
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=dropout, batch_first=True)
        self.history = nn.TransformerEncoder(layer, tf_layers)
        self.choice = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.confidence = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))
        self.progress = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))   # auxiliary: share of cells wrong
        self.wide_s = nn.Linear(hand_dim, 1) if (wide and hand_dim) else None    # linear shortcut: hand features -> score
        self.wide_c = nn.Linear(hand_dim, 1) if (wide and hand_dim) else None    # ... and -> confidence logit
        self.d = d

    # -- encoding a puzzle --------------------------------------------------------
    def encode_puzzles(self, puzzles: list, device: torch.device) -> list[dict]:
        """One GNN call for every grid of every puzzle. Each puzzle must carry
        .graph_train (list of (in, out) Graph), .graph_test_in, .graph_cands."""
        graphs, index = [], []
        for p in puzzles:
            idx = {"pairs": [], "cands": []}
            for gi, go in p.graph_train:
                idx["pairs"].append((len(graphs), len(graphs) + 1))
                graphs += [gi, go]
            idx["test"] = len(graphs)
            graphs.append(p.graph_test_in)
            for g in p.graph_cands:
                idx["cands"].append(len(graphs))
                graphs.append(g)
            idx["aux"] = []
            for k in p.aux_active:
                idx["aux"].append(len(graphs))
                graphs.append(p.aux_graphs[k])
            index.append(idx)
        p_hand = [getattr(p, 'hand', None) for p in puzzles]
        emb = self.gnn(GraphBatch.build(graphs, device))
        out = []
        for idx in index:
            pin = emb[[a for a, _ in idx["pairs"]]]
            pout = emb[[b for _, b in idx["pairs"]]]
            pairs = self.pair(torch.cat([pin, pout, pout - pin], 1)).mean(0)
            test = emb[idx["test"]]
            task = self.task_in(torch.cat([pairs, test], 0))
            c = emb[idx["cands"]]
            ctx = task.expand(len(idx["cands"]), -1)
            outs = pout.mean(0).expand(len(idx["cands"]), -1)
            cands = self.cand(torch.cat([c, c - test, ctx] + ([c - outs] if self.out_match else []), 1))
            ht = None
            if self.hand is not None and p_hand[len(out)] is not None:
                ht = torch.as_tensor(p_hand[len(out)], dtype=torch.float32, device=device)
                cands = cands + self.hand(ht)
            enc = {"task": task, "cands": cands, "hand": ht}
            if idx["aux"]:
                a = emb[idx["aux"]]
                ctx_a = task.expand(len(idx["aux"]), -1)
                enc["aux"] = self.cand(torch.cat([a, a - test, ctx_a] + ([a - pout.mean(0).expand(len(idx["aux"]), -1)] if self.out_match else []), 1))
            out.append(enc)
        return out

    # -- one decision ---------------------------------------------------------------
    def state(self, enc: dict, tried: list[tuple[int, bool]], context: list | None = None) -> torch.Tensor:
        """Causal pass over [earlier puzzles..., task, attempt_1, ..., attempt_k]; returns the last state.

        context = earlier puzzles of the same learner, oldest first: [(enc, [(candidate, was_right), ...]), ...].
        Without context this is exactly the single-puzzle model.
        """
        dev = enc["task"].device
        toks = []
        if context:
            n = len(context)
            for j, (cenc, ctried) in enumerate(context):
                age = self.episode.weight[min(n - j, self.episode.num_embeddings - 1)]
                toks.append(cenc["task"] + self.feedback.weight[2] + self.position.weight[0] + age)
                for k, (i, ok) in enumerate(ctried, start=1):
                    toks.append(cenc["cands"][i] + self.feedback.weight[int(ok)]
                                + self.position.weight[min(k, self.position.num_embeddings - 1)] + age)
        now = self.episode.weight[0] if context else 0.0
        toks.append(enc["task"] + self.feedback.weight[2] + self.position.weight[0] + now)
        for k, (i, ok) in enumerate(tried, start=1):
            toks.append(enc["cands"][i] + self.feedback.weight[int(ok)]
                        + self.position.weight[min(k, self.position.num_embeddings - 1)] + now)
        seq = torch.stack(toks).unsqueeze(0)
        L = seq.shape[1]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=dev), diagonal=1)
        return self.history(seq, mask=mask)[0, -1]

    def logits(self, enc: dict, tried: list[tuple[int, bool]], context: list | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.state(enc, tried, context)
        c = enc["cands"]
        sx = s.expand(c.shape[0], -1)
        scores = self.choice(torch.cat([sx, c, sx * c], 1)).squeeze(1)
        conf = self.confidence(torch.cat([sx, c], 1)).squeeze(1)          # logit of P(right)
        if self.wide_s is not None and enc.get("hand") is not None:
            scores = scores + self.wide_s(enc["hand"]).squeeze(1)
            conf = conf + self.wide_c(enc["hand"]).squeeze(1)
        allowed = torch.ones_like(scores, dtype=torch.bool)
        for i, _ in tried:
            allowed[i] = False
        scores = scores.masked_fill(~allowed, -1e9)
        return scores, conf


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


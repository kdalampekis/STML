"""Auxiliary 'world model' task for the hidden-premise learner: from what it sees (visible premises + revealed verdicts) predict which premises of the FULL world exist.

For every ordered pair of kinds (a, b) and every form f, the target is 1 if the full world contains premise f(a, b) (symmetric forms count both orders). Pairs that are
already visible premises are masked out (copying them is trivial). This is a supervised link-prediction loss that asks the graph encoder to infer hidden structure,
the explicit version of 'predict the world, learn from the prediction error' (Pillar 1 / C1), used here only as a representation-learning signal.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..logic import FORM_ORDER, SYMMETRIC_FORMS
from .logic_judge import LogicJudge, logits_of

NF = len(FORM_ORDER)


def aux_targets(ep) -> tuple[np.ndarray, np.ndarray]:
    """(target, mask) arrays of shape [n, n, NF] for an Episode with `.full` (all premises) and `.visible`."""
    n = ep.n
    tgt = np.zeros((n, n, NF), dtype=np.float32)
    seen = np.zeros((n, n, NF), dtype=bool)
    for p in ep.full:
        f = FORM_ORDER.index(p.form)
        tgt[p.a, p.b, f] = 1
        if p.form in SYMMETRIC_FORMS:
            tgt[p.b, p.a, f] = 1
    for p in ep.visible:
        f = FORM_ORDER.index(p.form)
        seen[p.a, p.b, f] = True
        if p.form in SYMMETRIC_FORMS:
            seen[p.b, p.a, f] = True
    mask = ~seen
    idx = np.arange(n)
    mask[idx, idx, :] = False
    return tgt, mask


def attach_aux(items, eps, shuffle_seed: int | None = None) -> None:
    """items must be episode_items(eps, ...) in order; stores `.aux` (target, mask) on each. shuffle_seed: pair each item with ANOTHER world's targets (control, same size only)."""
    k = len(eps[0].queries)
    per = [aux_targets(ep) for ep in eps]
    if shuffle_seed is not None:
        rnd = random.Random(shuffle_seed)
        by_n: dict[int, list[int]] = {}
        for i, ep in enumerate(eps):
            by_n.setdefault(ep.n, []).append(i)
        perm = list(range(len(eps)))
        for ids in by_n.values():
            sh = ids[:]
            rnd.shuffle(sh)
            for a_, b_ in zip(ids, sh):
                perm[a_] = b_
        per = [(per[perm[i]][0], per[i][1]) for i in range(len(eps))]
    for i in range(len(eps)):
        for t in range(k):
            items[i * k + t].aux = per[i]


def ensure_aux_head(model: LogicJudge) -> None:
    d = model.head[0].in_features // 4
    if not hasattr(model, "aux_head"):
        model.aux_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, NF))


def aux_loss(model: LogicJudge, h: torch.Tensor, b, chunk, device) -> torch.Tensor:
    off = np.cumsum([0] + [it.n for it in chunk])[:-1]
    ia, ib, tg, mk = [], [], [], []
    for g, it in enumerate(chunk):
        t, m = it.aux
        a_, b_ = np.nonzero(m.any(2))
        ia.append(off[g] + a_)
        ib.append(off[g] + b_)
        tg.append(t[a_, b_])
        mk.append(m[a_, b_])
    ia, ib = torch.as_tensor(np.concatenate(ia), device=device), torch.as_tensor(np.concatenate(ib), device=device)
    tg, mk = torch.as_tensor(np.concatenate(tg), device=device), torch.as_tensor(np.concatenate(mk), device=device)
    logit = model.aux_head(torch.cat([h[ia], h[ib]], 1))
    return (F.binary_cross_entropy_with_logits(logit, tg, reduction="none") * mk).sum() / mk.sum().clamp(min=1)


def fit_aux(model: LogicJudge, train, val, device, epochs: int = 10, lr: float = 1e-3, batch: int = 128, seed: int = 0, aux_weight: float = 0.5, weight_decay: float = 1e-4):
    """Same as logic_judge.fit but adds aux_weight * (hidden-premise prediction loss). Keeps the epoch with the best validation CE of the main task."""
    ensure_aux_head(model)
    model.to(device)
    rnd = random.Random(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, state = 1e9, None
    yv = torch.as_tensor([it.label for it in val], device=device)
    for ep in range(epochs):
        model.train()
        order = train[:]
        rnd.shuffle(order)
        for i in range(0, len(order), batch):
            chunk = order[i:i + batch]
            y = torch.as_tensor([it.label for it in chunk], device=device)
            out, h, b = model(chunk, device, return_h=True)
            loss = F.cross_entropy(out, y)
            if aux_weight > 0:
                loss = loss + aux_weight * aux_loss(model, h, b, chunk, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = float(F.cross_entropy(logits_of(model, val, device), yv))
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model

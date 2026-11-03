"""Surprise-weighted training for the logic judge (component C1/C6 of docs/ARCHITECTURE_V2.md), tested where labels are exact.

Surprise of a training example = how wrong the learner's own expectation was: delta = 1 - p(true class) (the unsigned reward
prediction error of a classifier). Four ways to use it, all normalised so the effective step size matches plain training:
  plain         cross-entropy
  surprise      weight 1 + beta * delta                       (learn more from what you did not expect)
  hypercorrect  weight 1 + beta * confidence * [wrong]        (learn more from CONFIDENT mistakes; Metcalfe 2017)
  focal         weight (1 - p_true)^gamma                      (the standard ML loss that also emphasises hard examples; a control)
  replay        sample training examples with probability proportional to (delta + 0.05)^alpha, refreshed every epoch
                (prioritised replay by surprise; Mattar & Daw 2018, Kumaran et al. 2016)
Weights are computed from the current forward pass and carry no gradient.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from .logic_data import Item
from .logic_judge import LogicJudge, logits_of

MODES = ("plain", "surprise", "hypercorrect", "focal", "replay")


def fit_weighted(model: LogicJudge, train: list[Item], val: list[Item], device, mode: str = "plain", beta: float = 1.0,
                 gamma: float = 2.0, alpha: float = 1.0, epochs: int = 10, lr: float = 1e-3, batch: int = 128, seed: int = 0) -> LogicJudge:
    """Multi-epoch training; keeps the epoch with the lowest (unweighted) validation loss."""
    assert mode in MODES
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    yv = torch.as_tensor([it.label for it in val], device=device)
    y_all = np.array([it.label for it in train])
    best, state = 1e9, None
    n = len(train)
    for ep in range(epochs):
        if mode == "replay" and ep > 0:
            p = torch.softmax(logits_of(model, train, device), 1).cpu().numpy()
            delta = 1.0 - p[np.arange(n), y_all]
            prob = (delta + 0.05) ** alpha
            order = rng.choice(n, size=n, p=prob / prob.sum())
        else:
            order = np.array(rnd.sample(range(n), n))
        model.train()
        for i in range(0, n, batch):
            chunk = [train[j] for j in order[i:i + batch]]
            y = torch.as_tensor([it.label for it in chunk], device=device)
            logits = model(chunk, device)
            ce = F.cross_entropy(logits, y, reduction="none")
            with torch.no_grad():
                p = torch.softmax(logits, 1)
                p_true = p.gather(1, y[:, None]).squeeze(1)
                conf, pred = p.max(1)
                if mode == "surprise":
                    w = 1 + beta * (1 - p_true)
                elif mode == "hypercorrect":
                    w = 1 + beta * conf * (pred != y).float()
                elif mode == "focal":
                    w = (1 - p_true).pow(gamma) + 1e-3
                else:
                    w = torch.ones_like(ce)
            loss = (w * ce).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = float(F.cross_entropy(logits_of(model, val, device), yv))
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model

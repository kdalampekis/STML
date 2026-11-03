"""Cross-fitted predictions of the history-free ('static') judge, used to build feedback that depends on the learner's own guess
(binary right/wrong feedback, surprise flags). Training predictions come from 2-fold cross-fitting so that they are not memorised;
validation and test predictions average the two half-data judges so that their quality matches the training ones.
"""

from __future__ import annotations

import numpy as np
import torch

from .logic_episodes import N_TYPES, episode_items
from .logic_judge import LogicJudge, fit, logits_of


def _softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def cross_fit_static_probs(tr, va, te, device, epochs: int = 10, seed: int = 0):
    """Returns class probabilities [n_items, 3] for train / val / test, items in `episode_items(eps, "static")` order."""
    st_tr, st_va, st_te = (episode_items(x, "static") for x in (tr, va, te))
    k = len(tr[0].queries)
    probs_tr = np.zeros((len(st_tr), 3))
    probs_va, probs_te = np.zeros((len(st_va), 3)), np.zeros((len(st_te), 3))
    half = [i % 2 for i in range(len(tr))]
    for h in (0, 1):
        torch.manual_seed(seed)
        m = LogicJudge(64, 4, n_types=N_TYPES["static"]).to(device)
        fit(m, episode_items([e for e, g in zip(tr, half) if g != h], "static"), st_va, device, epochs=epochs, seed=seed, log=lambda s: None)
        held = [i * k + t for i, g in enumerate(half) if g == h for t in range(k)]
        probs_tr[held] = _softmax(logits_of(m, [st_tr[j] for j in held], device).cpu().numpy())
        probs_va += _softmax(logits_of(m, st_va, device).cpu().numpy()) / 2
        probs_te += _softmax(logits_of(m, st_te, device).cpu().numpy()) / 2
    return probs_tr, probs_va, probs_te


def correctness_flags(eps, probs) -> list[list[int]]:
    """flags[i][t] = 1 if the static judge's most likely verdict for claim t of episode i was the true one."""
    k = len(eps[0].queries)
    return [[int(probs[i * k + t].argmax() == ep.labels[t]) for t in range(k)] for i, ep in enumerate(eps)]

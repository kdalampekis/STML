"""Does weighting training by surprise help the logic judge? (C1 / C6 of docs/ARCHITECTURE_V2.md, tested with exact labels)

PREDICTIONS, written before running:
  P1  surprise / hypercorrect weighting beats plain cross-entropy by >= 2 accuracy points when data are scarce (N <= 1000), on unseen worlds.
  P2  it improves calibration under shift (ECE on the 12-kind chain worlds), not just accuracy.
  P3  if focal loss (the standard hard-example reweighting) does the same, then 'surprise' adds nothing beyond known loss reweighting.
  Falsifier: no difference larger than the seed-to-seed sd (3 seeds), or no difference between surprise weighting and focal loss.
Conditions: plain; surprise beta = 1, 4; hypercorrect beta = 1, 4; focal gamma = 2; replay alpha = 1. Same model, data, steps, early stopping.

    python experiments/logic_surprise.py --quick
    python experiments/logic_surprise.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import stml.model.logic_data as ld  # noqa: E402
from logic_h3_certificate import aurc  # noqa: E402
from stml.model.logic_data import make_big_items, make_items  # noqa: E402
from stml.model.logic_judge import LogicJudge, ece, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.model.surprise import fit_weighted  # noqa: E402
from stml.runlog import Run  # noqa: E402

CONDS = {"plain": ("plain", {}), "surprise b=1": ("surprise", {"beta": 1.0}), "surprise b=4": ("surprise", {"beta": 4.0}),
         "hypercorrect b=1": ("hypercorrect", {"beta": 1.0}), "hypercorrect b=4": ("hypercorrect", {"beta": 4.0}),
         "focal g=2": ("focal", {"gamma": 2.0}), "replay a=1": ("replay", {"alpha": 1.0})}


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    ns, nw, epochs = (250, 500, 1000, 2000, 5000), (300, 50, 100), 10
    if a.quick:
        a.seeds, ns, nw, epochs = 1, (200,), (30, 10, 20), 2
    device = pick_device(None)
    with Run("logic_surprise" + ("_quick" if a.quick else ""), {**vars(a), "conds": list(CONDS), "ns": ns}) as run:
        pool = make_items([1, 2, 3, 4], range(nw[0]), rng_seed=1)
        val = make_items([1, 2, 3, 4], range(nw[0], nw[0] + nw[1]), rng_seed=2)
        tests = {"levels 1-4": make_items([1, 2, 3, 4], range(1000, 1000 + nw[2]), rng_seed=3),
                 "level 5": make_items([5], range(1000, 1000 + nw[2]), rng_seed=4),
                 "12 kinds, chains": make_big_items(12, range(1000, 1000 + (30 if not a.quick else 4)), rng_seed=12, chain=0.9)}
        R = {c: {n: {t: {"acc": [], "ece": [], "aurc": []} for t in tests} for n in ns} for c in CONDS}
        for seed in range(a.seeds):
            order = pool[:]
            random.Random(seed).shuffle(order)
            for name, (mode, kw) in CONDS.items():
                for n in ns:
                    torch.manual_seed(seed)
                    m = LogicJudge(64, 4).to(device)
                    fit_weighted(m, order[:n], val, device, mode=mode, epochs=epochs, seed=seed, **kw)
                    for t, items in tests.items():
                        y = np.array([it.label for it in items])
                        p = softmax(logits_of(m, items, device).cpu().numpy())
                        right = (p.argmax(1) == y).astype(float)
                        R[name][n][t]["acc"].append(float(right.mean()))
                        R[name][n][t]["ece"].append(ece(p.max(1), right))
                        R[name][n][t]["aurc"].append(aurc(p.max(1), right))
            run.log(f"seed {seed} done")
        mean = lambda c, n, t, k: float(np.mean(R[c][n][t][k]))
        sd = lambda c, n, t, k: float(np.std(R[c][n][t][k]))
        L = ["# Surprise-weighted training on the logic judge", "", f"{a.seeds} seeds; mean (sd). Predictions P1-P3 are in the script docstring.", ""]
        for t in tests:
            for k, label, f in (("acc", "accuracy", "{:.1%}"), ("ece", "ECE (raw softmax; lower is better)", "{:.3f}")):
                L += [f"## {t}: {label}", "", "| condition | " + " | ".join(f"N={n}" for n in ns) + " |", "| --- | " + " | ".join("---:" for _ in ns) + " |"]
                for c in CONDS:
                    L.append(f"| {c} | " + " | ".join(f"{f.format(mean(c, n, t, k))} ({f.format(sd(c, n, t, k))})" for n in ns) + " |")
                L.append("")
        L += ["## Paired accuracy difference to plain (points, mean over seeds, levels 1-4 / 12-kind chains)", "",
              "| condition | " + " | ".join(f"N={n}" for n in ns) + " |", "| --- | " + " | ".join("---:" for _ in ns) + " |"]
        for c in CONDS:
            if c == "plain":
                continue
            cells = []
            for n in ns:
                d1 = np.array(R[c][n]["levels 1-4"]["acc"]) - np.array(R["plain"][n]["levels 1-4"]["acc"])
                d2 = np.array(R[c][n]["12 kinds, chains"]["acc"]) - np.array(R["plain"][n]["12 kinds, chains"]["acc"])
                cells.append(f"{100 * d1.mean():+.1f} / {100 * d2.mean():+.1f}")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        run.finish({c: {str(n): v for n, v in d.items()} for c, d in R.items()}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

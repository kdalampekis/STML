"""Q14: does asking by uncertainty still beat random asking when the revealed verdicts are noisy (each wrong with probability p)?

Setting of logic_active_query (hidden premises, pool of 12, B = 6 asks, 8 held-out claims judged afterwards; held-out targets are the true verdicts). Noise only affects the verdicts
SHOWN for asked claims. Feedback judges: `matched` (trained on episodes with the same noise) and `clean` (trained on exact feedback, tested noisy).
PREDICTIONS, written before running:
  Y1  entropy minus random at p = 0.2 stays >= +3 points for the matched judge (it was +5.3 with exact feedback): choosing where the model is uncertain does not depend on the teacher being exact.
  Y2  the absolute accuracy of both policies falls with noise by >= 3 points (a wrong edge misleads).
  Falsifier of Y1: entropy minus random <= +1 at p = 0.2.

    python experiments/logic_noisy_asking.py --quick
    python experiments/logic_noisy_asking.py
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
from logic_active_query import run_policy  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, make_episodes, noisy_feedback  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

POOL, HELD, BUDGET, K = 12, 8, 6, 20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--p", type=float, default=0.2)
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, epochs = (1000, 50, 100), 10
    if a.quick:
        a.seeds, nw, epochs = 1, (24, 8, 10), 2
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    conds = [("clean", 0.0, 0.0), ("matched", a.p, a.p), ("clean_on_noisy", 0.0, a.p)]       # name, training noise, test noise
    with Run("logic_noisy_asking" + ("_quick" if a.quick else ""), vars(a)) as run:
        tr = make_episodes(levels, range(nw[0]), K, 0.5, rng_seed=1)
        va = make_episodes(levels, range(nw[0], nw[0] + nw[1]), K, 0.5, rng_seed=2)
        te = [e for e in make_episodes(levels, range(1000, 1000 + nw[2]), K, 0.5, rng_seed=3) if len(e.queries) >= POOL + HELD]
        heldout = [list(range(POOL, POOL + HELD)) for _ in te]
        res = {c[0]: {"random": [], "entropy": []} for c in conds}
        models = {}
        for seed in range(a.seeds):
            for name, ptr, pte in conds:
                if ptr not in models or models[ptr][0] != seed:
                    torch.manual_seed(seed)
                    m = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
                    trn, van = (noisy_feedback(tr, ptr, 10 * seed + 1), noisy_feedback(va, ptr, 10 * seed + 2)) if ptr else (tr, va)
                    fit(m, episode_items(trn, "fb"), episode_items(van, "fb"), device, epochs=epochs, seed=seed, log=lambda s: None)
                    models[ptr] = (seed, m)
                m = models[ptr][1]
                tst = noisy_feedback(te, pte, 10 * seed + 3) if pte else te
                for pol in ("random", "entropy"):
                    acc, _ = run_policy(m, tst, POOL, heldout, BUDGET, pol, device, random.Random(seed))
                    res[name][pol].append(acc[-1])
            run.log(f"seed {seed}: " + ", ".join(f"{n} random {res[n]['random'][-1]:.1%} entropy {res[n]['entropy'][-1]:.1%}" for n, _, _ in conds))
        L = ["# Asking by uncertainty with noisy feedback", "", f"{a.seeds} seeds; {len(te)} test episodes; p = {a.p}; accuracy on {HELD} held-out claims after {BUDGET} asks. Predictions Y1-Y2 in the script.", "",
             "| condition | random | entropy | entropy minus random |", "| --- | ---: | ---: | ---: |"]
        for n, _, _ in conds:
            r, e = np.array(res[n]["random"]), np.array(res[n]["entropy"])
            d = 100 * (e - r)
            L.append(f"| {n} | {100 * r.mean():.1f}% ({100 * r.std():.1f}) | {100 * e.mean():.1f}% ({100 * e.std():.1f}) | {d.mean():+.1f} ({d.std():.1f}) |")
        run.finish(res, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

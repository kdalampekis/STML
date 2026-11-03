"""Q21 (combination): RE-MASKING x NOISY TEACHER. Q11/Q11b: with 20% wrong verdicts the feedback gain falls to 38% (600 worlds) / 55% (2000 worlds) of the clean gain, and the learner only
learns to discount wrong edges with more data. Re-masking (Q17/Q18) is free data. If every re-masked draw also redraws WHICH verdicts are wrong, the learner sees the same world with many noise patterns.

300 worlds per level (levels 2-5), hidden-premise episodes (50% visible, 10 claims), p = 0.2 of the shown verdicts wrong. Conditions (all on the same worlds, 10 epochs):
  static_R10  no feedback, 10 draws (reference for the gain)
  clean_R10   clean feedback, 10 draws; tested on clean episodes (the reference gain) and on noisy episodes (clean_on_noisy)
  noisy_R1    noisy feedback, one draw (1200 episodes)
  noisy_R10   noisy feedback, 10 draws with the noise redrawn in each (12000 episodes)
Metric: accuracy at claims 7-10 on test episodes with noisy feedback (static: no feedback); share of the clean gain = (cond - static) / (clean - static).
PREDICTIONS, written before running:
  J1  noisy_R10 keeps >= 70% of the clean gain (noisy_R1 keeps about 40%).
  J2  the clean-trained learner tested on noisy feedback keeps at least 3 points less accuracy than noisy_R10.
  Falsifier of J1: noisy_R10 share below 55%.

    python experiments/logic_remask_noise.py --quick
    python experiments/logic_remask_noise.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stml.model.logic_data as ld  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, make_episodes, noisy_feedback  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402


def late(m, items, device) -> float:
    y = np.array([x.label for x in items])
    t = np.array([x.t for x in items])
    right = (logits_of(m, items, device).argmax(1).cpu().numpy() == y).astype(float)
    return float(right[t >= 7].mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--p", type=float, default=0.2)
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, draws, nv, nte, epochs, k = 300, 10, 50, 100, 10, 10
    if a.quick:
        a.seeds, nw, draws, nv, nte, epochs, k = 1, 8, 3, 8, 12, 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_remask_noise" + ("_quick" if a.quick else ""), vars(a)) as run:
        draws_ep = [make_episodes(levels, range(nw), k, 0.5, rng_seed=100 + r) for r in range(draws)]
        clean10 = sum(draws_ep, [])
        noisy10 = sum((noisy_feedback(e, a.p, 1000 + r) for r, e in enumerate(draws_ep)), [])
        noisy1 = noisy_feedback(draws_ep[0], a.p, 1000)
        va = make_episodes(levels, range(10 ** 4, 10 ** 4 + nv), k, 0.5, rng_seed=2)
        te = make_episodes(levels, range(1000, 1000 + nte), k, 0.5, rng_seed=3)
        te_noisy = noisy_feedback(te, a.p, 77)
        va_noisy = noisy_feedback(va, a.p, 78)
        train = {"static_R10": (episode_items(clean10, "static"), episode_items(va, "static"), "static"),
                 "clean_R10": (episode_items(clean10, "fb"), episode_items(va, "fb"), "fb"),
                 "noisy_R1": (episode_items(noisy1, "fb"), episode_items(va_noisy, "fb"), "fb"),
                 "noisy_R10": (episode_items(noisy10, "fb"), episode_items(va_noisy, "fb"), "fb")}
        t_clean, t_noisy, t_static = episode_items(te, "fb"), episode_items(te_noisy, "fb"), episode_items(te, "static")
        res = {"static_R10": [], "clean_R10 (clean test)": [], "clean_R10 (noisy test)": [], "noisy_R1 (noisy test)": [], "noisy_R10 (noisy test)": []}
        for seed in range(a.seeds):
            for c, (a_tr, a_va, mode) in train.items():
                torch.manual_seed(seed)
                m = LogicJudge(64, 4, n_types=N_TYPES[mode]).to(device)
                fit(m, a_tr, a_va, device, epochs=epochs, seed=seed, log=lambda s: None)
                if c == "static_R10":
                    res[c].append(late(m, t_static, device))
                elif c == "clean_R10":
                    res["clean_R10 (clean test)"].append(late(m, t_clean, device))
                    res["clean_R10 (noisy test)"].append(late(m, t_noisy, device))
                else:
                    res[f"{c} (noisy test)"].append(late(m, t_noisy, device))
            run.log(f"seed {seed}: " + ", ".join(f"{k_} {v[-1]:.1%}" for k_, v in res.items()))
        st, cl = np.array(res["static_R10"]), np.array(res["clean_R10 (clean test)"])
        L = ["# Re-masking x noisy teacher", "", f"{a.seeds} seeds; p = {a.p}; accuracy at claims 7-10 (mean, sd). Predictions J1-J2 in the script.", "",
             "| condition | accuracy | gain over static | share of the clean gain |", "| --- | ---: | ---: | ---: |"]
        for c, v in res.items():
            v = np.array(v)
            g = 100 * (v - st)
            share = 100 * (v - st).mean() / (cl - st).mean() if c != "static_R10" else float("nan")
            L.append(f"| {c} | {100 * v.mean():.1f}% ({100 * v.std():.1f}) | {g.mean():+.1f} ({g.std():.1f}) | {share:.0f}% |")
        d = 100 * (np.array(res["noisy_R10 (noisy test)"]) - np.array(res["clean_R10 (noisy test)"]))
        L += ["", f"noisy_R10 minus clean_R10 (both on noisy test): {d.mean():+.1f} ({d.std():.1f})"]
        run.finish(res, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

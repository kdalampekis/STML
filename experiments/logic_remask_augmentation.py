"""Q17 (out of the box): FREE DATA. The feedback learner is data-limited (more worlds help most). A world can be turned into many different training episodes by re-drawing which
premises are hidden and which claims are asked, at no labelling cost. Does 'same worlds, re-masked' act like new worlds?

Hidden-premise episodes as in logic_feedback_type (50% visible, 10 claims, verdicts revealed as edges), 300 training worlds per level (levels 2-5).
Conditions (same model, same epochs over the data): base (1 episode per world, 1200 episodes); remask3 (3 independent re-draws of the visible subset and the asked claims per world, 3600 episodes
from the SAME 1200 worlds); worlds3 (900 distinct worlds per level, 3600 episodes: the real-data reference); remask10 (10 re-draws, 12000 episodes).
PREDICTIONS, written before running (accuracy at claims 7-10):
  F1  remask3 beats base by >= 1.5 points (some of the gain of more data is available for free).
  F2  remask3 recovers at most 60% of the gain of worlds3 over base (new worlds carry information that re-masking cannot: the world structure stays the same).
  F3  remask10 is not better than remask3 by more than the seed sd (diminishing returns of one world's structure).
  Falsifier of F1: remask3 - base within the seed sd.
Q17b (--conds base,base_x3,base_x10,remask3,remask10): seed 0 of Q17 gave base 73.9, remask3 77.2, worlds3 76.6, remask10 79.7 (overall), which breaks F2; but remask3/10 also take 3x/10x more gradient steps.
  base_x3 / base_x10 train the 1200-episode baseline for 3x / 10x the epochs (the same number of updates). PREDICTIONS, written before running:
  F4  base_x3 and base_x10 recover at most half of the gain of remask3 and remask10 over base (the benefit is the diversity of the episodes, not the number of steps).
  F5  remask10 is better than remask3 by >= 1 point (diversity keeps helping up to 10 draws).
  Falsifier of F4: base_x10 within the seed sd of remask10.

    python experiments/logic_remask_augmentation.py --quick
    python experiments/logic_remask_augmentation.py
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
from stml.model.logic_episodes import N_TYPES, episode_items, make_episodes  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

BANDS = {"t=0": (0, 0), "t=1-3": (1, 3), "t=4-6": (4, 6), "t=7-9": (7, 9)}
COND = ["base", "remask3", "worlds3", "remask10"]
EPOCH_MULT = {"base_x3": 3, "base_x10": 10}          # Q17b: the baseline given as many gradient steps as remask3 / remask10 (same 1200 episodes, more epochs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--conds", default=",".join(COND), help="comma-separated subset / extension of the conditions (Q17b adds base_x3,base_x10)")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, nv, nte, epochs, k = 300, 50, 100, 10, 10
    if a.quick:
        a.seeds, nw, nv, nte, epochs, k = 1, 12, 8, 12, 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    conds = a.conds.split(',')
    with Run("logic_remask_augmentation" + ("_quick" if a.quick else ""), {**vars(a), "conditions": conds}) as run:
        va = make_episodes(levels, range(10 ** 4, 10 ** 4 + nv), k, 0.5, rng_seed=2)
        te = make_episodes(levels, range(1000, 1000 + nte), k, 0.5, rng_seed=3)
        remask = lambda reps: sum((make_episodes(levels, range(nw), k, 0.5, rng_seed=100 + r) for r in range(reps)), [])
        train = {"base_x3": remask(1), "base_x10": remask(1), "base": remask(1), "remask3": remask(3), "worlds3": make_episodes(levels, range(3 * nw), k, 0.5, rng_seed=1), "remask10": remask(10)}
        run.log("episodes: " + ", ".join(f"{c} {len(train[c])}" for c in conds))
        a_va, a_te = episode_items(va, "fb"), episode_items(te, "fb")
        y = np.array([x.label for x in a_te])
        tt = np.array([x.t for x in a_te])
        acc = {c: {b: [] for b in list(BANDS) + ["all"]} for c in conds}
        for seed in range(a.seeds):
            for c in conds:
                torch.manual_seed(seed)
                m = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
                fit(m, episode_items(train[c], "fb"), a_va, device, epochs=epochs * EPOCH_MULT.get(c, 1), seed=seed, log=lambda s: None)
                right = (logits_of(m, a_te, device).argmax(1).cpu().numpy() == y).astype(float)
                for b, (lo, hi) in BANDS.items():
                    acc[c][b].append(float(right[(tt >= lo) & (tt <= hi)].mean()))
                acc[c]["all"].append(float(right.mean()))
            run.log(f"seed {seed}: " + ", ".join(f"{c} {acc[c]['all'][-1]:.1%}" for c in conds))
        L = ["# Re-masking as free data", "", f"{a.seeds} seeds; {len(te)} test episodes x {k} claims; {nw} worlds per level; accuracy by position (mean, sd). Predictions F1-F3 in the script.", "",
             "| condition | " + " | ".join(list(BANDS) + ["all"]) + " |", "| --- | " + " | ".join("---:" for _ in range(len(BANDS) + 1)) + " |"]
        for c in conds:
            L.append(f"| {c} | " + " | ".join(f"{100 * np.mean(acc[c][b]):.1f}% ({100 * np.std(acc[c][b]):.1f})" for b in list(BANDS) + ["all"]) + " |")
        L += ["", "| comparison | " + " | ".join(list(BANDS) + ["all"]) + " |", "| --- | " + " | ".join("---:" for _ in range(len(BANDS) + 1)) + " |"]
        for x, y_ in [p_ for p_ in (("remask3", "base"), ("worlds3", "base"), ("remask10", "remask3"), ("worlds3", "remask3"), ("base_x3", "base"), ("base_x10", "base"), ("remask3", "base_x3"), ("remask10", "base_x10")) if p_[0] in conds and p_[1] in conds]:
            cells = []
            for b in list(BANDS) + ["all"]:
                d = 100 * (np.array(acc[x][b]) - np.array(acc[y_][b]))
                cells.append(f"{d.mean():+.1f} ({d.std():.1f})")
            L.append(f"| {x} minus {y_} | " + " | ".join(cells) + " |")
        if not {"worlds3", "base", "remask3"} <= set(conds):
            run.finish(acc, "\n".join(L) + "\n")
            return
        gain = 100 * (np.array(acc["worlds3"]["t=7-9"]) - np.array(acc["base"]["t=7-9"])).mean()
        got = 100 * (np.array(acc["remask3"]["t=7-9"]) - np.array(acc["base"]["t=7-9"])).mean()
        L += ["", f"At claims 7-10, remask3 recovers {100 * got / gain:.0f}% of the gain of 3x distinct worlds." if abs(gain) > 0.5 else ""]
        run.finish(acc, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

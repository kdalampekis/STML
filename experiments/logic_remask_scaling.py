"""Q18 (rule 7b, a promising result): how far does RE-MASKING scale as free data? (Q17/Q17b: +3.3 at 3 draws, +6 at 10 draws over an equal-steps baseline.)

Hidden-premise episodes as in logic_feedback_type (50% visible, 10 claims, verdicts as edges), levels 2-5. A 're-mask' = a new random draw of the visible subset and the asked claims for the same world.
Conditions: W worlds per level in {100, 300} x draws R in {1, 3, 10, 30} (R x W x 4 episodes; 10 epochs over the set), plus the reference 'distinct1000' = 1000 distinct worlds per level, one episode each.
PREDICTIONS, written before running (accuracy at claims 7-10):
  G1  more draws keep helping but saturate: at 300 worlds remask30 - remask10 >= +0.5 and smaller than remask10 - remask3 (diminishing returns).
  G2  300 worlds with 30 draws match or beat 1000 distinct worlds with one episode each (free data beats 3.3x real worlds).
  G3  the gain of 10 draws over 1 draw is at least as large at 100 worlds as at 300 (the poorer the data, the more re-masking helps).
  Falsifier of G2: distinct1000 beats remask30 on 300 worlds by more than the seed sd.

    python experiments/logic_remask_scaling.py --quick
    python experiments/logic_remask_scaling.py
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    worlds, draws, nref, nv, nte, epochs, k = (100, 300), (1, 3, 10, 30), 1000, 50, 100, 10, 10
    if a.quick:
        a.seeds, worlds, draws, nref, nv, nte, epochs, k = 1, (6, 12), (1, 3), 18, 8, 12, 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_remask_scaling" + ("_quick" if a.quick else ""), vars(a)) as run:
        va = make_episodes(levels, range(10 ** 4, 10 ** 4 + nv), k, 0.5, rng_seed=2)
        te = make_episodes(levels, range(1000, 1000 + nte), k, 0.5, rng_seed=3)
        a_va, a_te = episode_items(va, "fb"), episode_items(te, "fb")
        y = np.array([x.label for x in a_te])
        tt = np.array([x.t for x in a_te])
        sets = {f"W{w}_R{r}": (w, r) for w in worlds for r in draws}
        sets["distinct1000"] = (nref, 1)
        acc = {n: {"late": [], "all": []} for n in sets}
        for name, (w, r) in sets.items():
            ep = sum((make_episodes(levels, range(w), k, 0.5, rng_seed=(1 if name == "distinct1000" else 100 + i)) for i in range(r)), [])
            items = episode_items(ep, "fb")
            run.log(f"{name}: {len(ep)} episodes")
            for seed in range(a.seeds):
                torch.manual_seed(seed)
                m = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
                fit(m, items, a_va, device, epochs=epochs, seed=seed, log=lambda s: None)
                right = (logits_of(m, a_te, device).argmax(1).cpu().numpy() == y).astype(float)
                acc[name]["late"].append(float(right[tt >= 7].mean()))
                acc[name]["all"].append(float(right.mean()))
            run.log(f"{name}: late {100 * np.mean(acc[name]['late']):.1f}%, all {100 * np.mean(acc[name]['all']):.1f}%")
        L = ["# Scaling of re-masking", "", f"{a.seeds} seeds; {len(te)} test episodes x {k} claims; accuracy (mean, sd over seeds). Predictions G1-G3 in the script.", "",
             "| set | episodes | late (claims 7-10) | all |", "| --- | ---: | ---: | ---: |"]
        for name, (w, r) in sets.items():
            L.append(f"| {name} | {w * r * 4} | {100 * np.mean(acc[name]['late']):.1f}% ({100 * np.std(acc[name]['late']):.1f}) | {100 * np.mean(acc[name]['all']):.1f}% ({100 * np.std(acc[name]['all']):.1f}) |")
        L += ["", "## Paired differences (points, late; sd)", "", "| comparison | difference |", "| --- | ---: |"]
        pairs = []
        for w in worlds:
            pairs += [(f"W{w}_R{draws[i]}", f"W{w}_R{draws[i - 1]}") for i in range(1, len(draws))]
        pairs += [(f"W{worlds[-1]}_R{draws[-1]}", "distinct1000")]
        for x, y_ in pairs:
            d = 100 * (np.array(acc[x]["late"]) - np.array(acc[y_]["late"]))
            L.append(f"| {x} minus {y_} | {d.mean():+.1f} ({d.std():.1f}) |")
        run.finish(acc, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

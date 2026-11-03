"""Q20 (rule 7b): CAPACITY once data is plentiful. The earlier hyperparameter search (logic_hparam_search, 300-1000 worlds, one episode each) found width/depth/epochs inside the seed sd:
the learner was data-limited. Re-masking (Q17/Q18) removes much of the data limit. Is the learner now capacity-limited?

300 worlds per level x 10 draws (12000 episodes, the Q17/Q18 setting), hidden-premise episodes with verdicts as edges, 10 epochs. Configs: width in {64, 128} x GNN layers in {4, 6}.
PREDICTIONS, written before running (accuracy at claims 7-10, reference: width 64 / 4 layers = 84.8% in Q18):
  K1  width 128 beats width 64 by >= 1 point (with plentiful data a larger model helps; at 1 draw the best tuned config gained <= 0.7).
  K2  6 layers are not better than 4 by more than the seed sd (the problem is not depth-limited, shown at 1 draw).
  Falsifier of K1: width 128 within the seed sd of width 64.

    python experiments/logic_remask_capacity.py --quick
    python experiments/logic_remask_capacity.py
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

CONFIGS = {"w64_l4": (64, 4), "w128_l4": (128, 4), "w64_l6": (64, 6), "w128_l6": (128, 6)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, draws, nv, nte, epochs, k = 300, 10, 50, 100, 10, 10
    if a.quick:
        a.seeds, nw, draws, nv, nte, epochs, k = 1, 8, 3, 8, 12, 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_remask_capacity" + ("_quick" if a.quick else ""), {**vars(a), "configs": CONFIGS}) as run:
        tr = sum((make_episodes(levels, range(nw), k, 0.5, rng_seed=100 + r) for r in range(draws)), [])
        a_tr = episode_items(tr, "fb")
        a_va = episode_items(make_episodes(levels, range(10 ** 4, 10 ** 4 + nv), k, 0.5, rng_seed=2), "fb")
        a_te = episode_items(make_episodes(levels, range(1000, 1000 + nte), k, 0.5, rng_seed=3), "fb")
        y = np.array([x.label for x in a_te])
        tt = np.array([x.t for x in a_te])
        acc = {c: {"late": [], "all": []} for c in CONFIGS}
        for seed in range(a.seeds):
            for c, (d, l) in CONFIGS.items():
                torch.manual_seed(seed)
                m = LogicJudge(d, l, n_types=N_TYPES["fb"]).to(device)
                fit(m, a_tr, a_va, device, epochs=epochs, seed=seed, log=lambda s: None)
                right = (logits_of(m, a_te, device).argmax(1).cpu().numpy() == y).astype(float)
                acc[c]["late"].append(float(right[tt >= 7].mean()))
                acc[c]["all"].append(float(right.mean()))
            run.log(f"seed {seed}: " + ", ".join(f"{c} {acc[c]['late'][-1]:.1%}" for c in CONFIGS))
        L = ["# Capacity on re-masked data", "", f"{a.seeds} seeds; 300 worlds x {draws} draws; accuracy (mean, sd). Predictions K1-K2 in the script.", "",
             "| config | late (claims 7-10) | all |", "| --- | ---: | ---: |"]
        for c in CONFIGS:
            L.append(f"| {c} | {100 * np.mean(acc[c]['late']):.1f}% ({100 * np.std(acc[c]['late']):.1f}) | {100 * np.mean(acc[c]['all']):.1f}% ({100 * np.std(acc[c]['all']):.1f}) |")
        L += ["", "| comparison | late | all |", "| --- | ---: | ---: |"]
        for x, y_ in (("w128_l4", "w64_l4"), ("w64_l6", "w64_l4"), ("w128_l6", "w64_l4"), ("w128_l6", "w128_l4")):
            dl, da = (100 * (np.array(acc[x][m_]) - np.array(acc[y_][m_])) for m_ in ("late", "all"))
            L.append(f"| {x} minus {y_} | {dl.mean():+.1f} ({dl.std():.1f}) | {da.mean():+.1f} ({da.std():.1f}) |")
        run.finish(acc, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

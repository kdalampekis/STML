"""Q19 (out of the box): DOMAIN RANDOMISATION of the visible fraction inside the re-masked draws. Q17/Q18: re-masking is free data. Does varying HOW MUCH of the world is visible
across the draws (0.3 ... 0.7) make the learner robust to a different visible fraction at test time, at no cost in distribution?

300 worlds per level (levels 2-5), 10 draws per world (12000 episodes), hidden-premise episodes with verdicts as edges. Training: 'fixed' = every draw has 50% of the premises visible;
'mixed' = the 10 draws cycle through visible fractions 0.3, 0.4, 0.5, 0.6, 0.7 (two each). Test: the same 100 worlds per level with 30%, 50% and 70% visible.
PREDICTIONS, written before running (accuracy at claims 7-10):
  H1  mixed beats fixed by >= 3 points on the shifted tests (30% and 70% visible).
  H2  mixed costs at most 1 point at 50% visible (in distribution).
  Falsifier of H1: mixed - fixed within the seed sd on both shifted tests.

    python experiments/logic_remask_vis.py --quick
    python experiments/logic_remask_vis.py
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

VIS = (0.3, 0.5, 0.7)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, draws, nv, nte, epochs, k = 300, 10, 50, 100, 10, 10
    if a.quick:
        a.seeds, nw, draws, nv, nte, epochs, k = 1, 8, 5, 8, 12, 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    cycle = (0.3, 0.4, 0.5, 0.6, 0.7)
    with Run("logic_remask_vis" + ("_quick" if a.quick else ""), vars(a)) as run:
        va = make_episodes(levels, range(10 ** 4, 10 ** 4 + nv), k, 0.5, rng_seed=2) + make_episodes(levels, range(2 * 10 ** 4, 2 * 10 ** 4 + nv // 2), k, 0.3, rng_seed=5) \
            + make_episodes(levels, range(3 * 10 ** 4, 3 * 10 ** 4 + nv // 2), k, 0.7, rng_seed=6)
        a_va = episode_items(va, "fb")
        tests = {v: episode_items(make_episodes(levels, range(1000, 1000 + nte), k, v, rng_seed=3), "fb") for v in VIS}
        train = {"fixed": sum((make_episodes(levels, range(nw), k, 0.5, rng_seed=100 + r) for r in range(draws)), []),
                 "mixed": sum((make_episodes(levels, range(nw), k, cycle[r % len(cycle)], rng_seed=100 + r) for r in range(draws)), [])}
        acc = {c: {v: [] for v in VIS} for c in train}
        for seed in range(a.seeds):
            for c, ep in train.items():
                torch.manual_seed(seed)
                m = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
                fit(m, episode_items(ep, "fb"), a_va, device, epochs=epochs, seed=seed, log=lambda s: None)
                for v, items in tests.items():
                    y = np.array([x.label for x in items])
                    tt = np.array([x.t for x in items])
                    right = (logits_of(m, items, device).argmax(1).cpu().numpy() == y).astype(float)
                    acc[c][v].append(float(right[tt >= 7].mean()))
            run.log(f"seed {seed}: " + ", ".join(f"{c} @{v}: {acc[c][v][-1]:.1%}" for c in train for v in VIS))
        L = ["# Randomised visible fraction", "", f"{a.seeds} seeds; {len(next(iter(tests.values())))} test claims per visible fraction; accuracy at claims 7-10 (mean, sd). Predictions H1-H2 in the script.", "",
             "| training | " + " | ".join(f"test {int(v * 100)}% visible" for v in VIS) + " |", "| --- | " + " | ".join("---:" for _ in VIS) + " |"]
        for c in train:
            L.append(f"| {c} | " + " | ".join(f"{100 * np.mean(acc[c][v]):.1f}% ({100 * np.std(acc[c][v]):.1f})" for v in VIS) + " |")
        d = {v: 100 * (np.array(acc["mixed"][v]) - np.array(acc["fixed"][v])) for v in VIS}
        L += ["", "| comparison | " + " | ".join(f"{int(v * 100)}%" for v in VIS) + " |", "| --- | " + " | ".join("---:" for _ in VIS) + " |",
              "| mixed minus fixed | " + " | ".join(f"{d[v].mean():+.1f} ({d[v].std():.1f})" for v in VIS) + " |"]
        run.finish(acc, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

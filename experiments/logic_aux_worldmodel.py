"""Q16 (out of the box): an AUXILIARY WORLD-MODEL LOSS for the data-limited feedback learner. Ask the encoder to predict the HIDDEN premises of the world (link prediction), not just the verdict.

Bottleneck targeted: the feedback learner is data-limited (gap closed 22% -> 35% -> 45% at 100 / 300 / 1000 training worlds; Q5: +3.8 points from 3.3x more data). Hidden-premise episodes as in
logic_feedback_type (50% of the premises visible, verdicts revealed as edges). Extra supervision: for every pair of kinds and every form, 'is this a premise of the FULL world?' (visible pairs masked),
weight 0.5. This is the explicit 'predict the world, learn from the prediction error' idea of Pillar 1 (C1), used as representation learning; cf. auxiliary tasks in RL (Jaderberg 2017), world models (Ha & Schmidhuber 2018).
Conditions (accuracy at claims 7-10 and overall; N = 300 and 1000 training worlds per level): static, static_aux, fb, fb_aux, fb_aux_shuf (aux targets taken from ANOTHER world of the same size: a regulariser control).
PREDICTIONS, written before running:
  W1  fb_aux beats fb by >= 2 points at claims 7-10 at N = 300.
  W2  the gain at N = 1000 is at most half of the gain at N = 300 (an auxiliary signal substitutes for data).
  W3  fb_aux_shuf is not better than fb by more than 1 point (the gain, if any, comes from true information about the world).
  W4  static_aux beats static by >= 1 point (inferring hidden structure helps even without feedback).
  Falsifier of W1: fb_aux - fb within the seed sd.

    python experiments/logic_aux_worldmodel.py --quick
    python experiments/logic_aux_worldmodel.py
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
from stml.model.logic_aux import attach_aux, fit_aux  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, make_episodes  # noqa: E402
from stml.model.logic_judge import LogicJudge, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

BANDS = {"t=0": (0, 0), "t=1-3": (1, 3), "t=4-6": (4, 6), "t=7-9": (7, 9)}
COND = ["static", "static_aux", "fb", "fb_aux", "fb_aux_shuf"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sizes", default="300,1000")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    a = ap.parse_args()
    ld.SIZE_FREE = True
    sizes, nv, nte, epochs, k = [int(x) for x in a.sizes.split(",")], 50, 100, 10, 10
    if a.quick:
        a.seeds, sizes, nv, nte, epochs, k = 1, [12], 8, 12, 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_aux_worldmodel" + ("_quick" if a.quick else ""), {**vars(a), "conditions": COND}) as run:
        te = make_episodes(levels, range(1000, 1000 + nte), k, 0.5, rng_seed=3)
        res = {N: {c: {b: [] for b in list(BANDS) + ["all"]} for c in COND} for N in sizes}
        for N in sizes:
            tr = make_episodes(levels, range(N), k, 0.5, rng_seed=1)
            va = make_episodes(levels, range(N, N + nv), k, 0.5, rng_seed=2)
            data = {}
            for mode in ("static", "fb"):
                data[mode] = tuple(episode_items(x, mode) for x in (tr, va, te))
            for mode in ("static", "fb"):
                attach_aux(data[mode][0], tr)
                attach_aux(data[mode][1], va)
            shuf = (episode_items(tr, "fb"), episode_items(va, "fb"))
            attach_aux(shuf[0], tr, shuffle_seed=5)
            attach_aux(shuf[1], va, shuffle_seed=6)
            for seed in range(a.seeds):
                for c in COND:
                    mode = "static" if c.startswith("static") else "fb"
                    a_tr, a_va, a_te = data[mode]
                    if c == "fb_aux_shuf":
                        a_tr, a_va = shuf
                    torch.manual_seed(seed)
                    m = LogicJudge(64, 4, n_types=N_TYPES[mode]).to(device)
                    fit_aux(m, a_tr, a_va, device, epochs=epochs, seed=seed, aux_weight=a.aux_weight if "aux" in c else 0.0)
                    y = np.array([x.label for x in a_te])
                    tt = np.array([x.t for x in a_te])
                    right = (logits_of(m, a_te, device).argmax(1).cpu().numpy() == y).astype(float)
                    for b, (lo, hi) in BANDS.items():
                        res[N][c][b].append(float(right[(tt >= lo) & (tt <= hi)].mean()))
                    res[N][c]["all"].append(float(right.mean()))
                run.log(f"N={N} seed {seed}: " + ", ".join(f"{c} {res[N][c]['all'][-1]:.1%}" for c in COND))
        L = ["# Auxiliary world-model loss", "", f"{a.seeds} seeds; {len(te)} test episodes x {k} claims; aux weight {a.aux_weight}; accuracy by position (mean, sd). Predictions W1-W4 in the script.", ""]
        for N in sizes:
            L += [f"## {N} training worlds per level", "", "| condition | " + " | ".join(list(BANDS) + ["all"]) + " |", "| --- | " + " | ".join("---:" for _ in range(len(BANDS) + 1)) + " |"]
            for c in COND:
                L.append(f"| {c} | " + " | ".join(f"{100 * np.mean(res[N][c][b]):.1f}% ({100 * np.std(res[N][c][b]):.1f})" for b in list(BANDS) + ["all"]) + " |")
            L += ["", "| comparison | " + " | ".join(list(BANDS) + ["all"]) + " |", "| --- | " + " | ".join("---:" for _ in range(len(BANDS) + 1)) + " |"]
            for x, y_ in (("fb_aux", "fb"), ("fb_aux_shuf", "fb"), ("fb_aux", "fb_aux_shuf"), ("static_aux", "static"), ("fb", "static")):
                cells = []
                for b in list(BANDS) + ["all"]:
                    d = 100 * (np.array(res[N][x][b]) - np.array(res[N][y_][b]))
                    cells.append(f"{d.mean():+.1f} ({d.std():.1f})")
                L.append(f"| {x} minus {y_} | " + " | ".join(cells) + " |")
            L.append("")
        run.finish({str(N): res[N] for N in sizes}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

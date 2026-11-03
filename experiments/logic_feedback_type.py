"""Feedback TYPE in the sequential regime (Q8): the sequential version of H2, and the basis of a diagnostic-vs-scalar arm in the game.

Hidden-premise episodes (50% of the premises visible, 10 claims judged in a row). After each judgement the learner receives:
  static   nothing
  asked    only that the claim was asked (no verdict)
  bin      BINARY feedback: whether the history-free judge's guess for that claim was right (cross-fitted, no leakage)   <- the 'scalar' arm
  bin_shuf the same flags shuffled within the episode (control: right amount of 0/1, wrong claims)
  fb       the TRUE VERDICT (follows / contradicted / open)                                                             <- the 'diagnostic' arm (without the why)
Feedback is added to the world graph as edges (stml/model/logic_episodes.py).

PREDICTIONS, written before running:
  F1  the full verdict beats binary feedback by >= 3 accuracy points at late positions (claims 7-10).
  F2  binary feedback beats static by >= 1 point at late positions, and the shuffled control does not (the single bit carries real information).
  F3  binary feedback gains less than half of what the full verdict gains over static.
  Falsifier of F2: bin - static within the seed sd, or bin_shuf as good as bin.

    python experiments/logic_feedback_type.py --quick
    python experiments/logic_feedback_type.py
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

import stml.model.logic_data as ld  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, make_episodes  # noqa: E402
from stml.model.logic_flags import correctness_flags, cross_fit_static_probs  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

BANDS = {"t=0": (0, 0), "t=1-3": (1, 3), "t=4-6": (4, 6), "t=7-9": (7, 9)}
COND = ["static", "asked", "bin", "bin_shuf", "fb"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, epochs, k = (600, 50, 100), 10, 10
    if a.quick:
        a.seeds, nw, epochs, k = 1, (24, 8, 12), 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_feedback_type" + ("_quick" if a.quick else ""), {**vars(a), "conditions": COND}) as run:
        tr = make_episodes(levels, range(nw[0]), k, 0.5, rng_seed=1)
        va = make_episodes(levels, range(nw[0], nw[0] + nw[1]), k, 0.5, rng_seed=2)
        te = make_episodes(levels, range(1000, 1000 + nw[2]), k, 0.5, rng_seed=3)
        acc = {c: {b: [] for b in list(BANDS) + ["all"]} for c in COND}
        for seed in range(a.seeds):
            p_tr, p_va, p_te = cross_fit_static_probs(tr, va, te, device, epochs, seed)
            f_tr, f_va, f_te = (correctness_flags(e, p) for e, p in ((tr, p_tr), (va, p_va), (te, p_te)))
            rnd = random.Random(seed)
            shuf = lambda fl: [rnd.sample(f, len(f)) for f in fl]
            sets = {"static": (episode_items(tr, "static"), episode_items(va, "static"), episode_items(te, "static"), "static"),
                    "asked": tuple(episode_items(x, "asked") for x in (tr, va, te)) + ("asked",),
                    "bin": (episode_items(tr, "bin", f_tr), episode_items(va, "bin", f_va), episode_items(te, "bin", f_te), "bin"),
                    "bin_shuf": (episode_items(tr, "bin", shuf(f_tr)), episode_items(va, "bin", shuf(f_va)), episode_items(te, "bin", shuf(f_te)), "bin"),
                    "fb": tuple(episode_items(x, "fb") for x in (tr, va, te)) + ("fb",)}
            for c in COND:
                a_tr, a_va, a_te, mode = sets[c]
                torch.manual_seed(seed)
                m = LogicJudge(64, 4, n_types=N_TYPES[mode]).to(device)
                fit(m, a_tr, a_va, device, epochs=epochs, seed=seed, log=lambda s: None)
                y = np.array([x.label for x in a_te])
                tt = np.array([x.t for x in a_te])
                right = (logits_of(m, a_te, device).argmax(1).cpu().numpy() == y).astype(float)
                for b, (lo, hi) in BANDS.items():
                    acc[c][b].append(float(right[(tt >= lo) & (tt <= hi)].mean()))
                acc[c]["all"].append(float(right.mean()))
            run.log(f"seed {seed}: " + ", ".join(f"{c} {acc[c]['all'][-1]:.1%}" for c in COND))
        L = ["# Feedback type in the sequential regime: binary vs full verdict", "", f"{a.seeds} seeds; {len(te)} test episodes x {k} claims; accuracy by position (mean, sd). Predictions F1-F3 are in the script.", "",
             "| condition | " + " | ".join(list(BANDS) + ["all"]) + " |", "| --- | " + " | ".join("---:" for _ in range(len(BANDS) + 1)) + " |"]
        for c in COND:
            L.append(f"| {c} | " + " | ".join(f"{100 * np.mean(acc[c][b]):.1f}% ({100 * np.std(acc[c][b]):.1f})" for b in list(BANDS) + ["all"]) + " |")
        L += ["", "## Paired differences (points; sd of the paired difference)", "", "| comparison | " + " | ".join(list(BANDS) + ["all"]) + " |", "| --- | " + " | ".join("---:" for _ in range(len(BANDS) + 1)) + " |"]
        for x, y_ in (("fb", "bin"), ("bin", "static"), ("bin", "bin_shuf"), ("bin", "asked"), ("fb", "static")):
            cells = []
            for b in list(BANDS) + ["all"]:
                d = 100 * (np.array(acc[x][b]) - np.array(acc[y_][b]))
                cells.append(f"{d.mean():+.1f} ({d.std():.1f})")
            L.append(f"| {x} minus {y_} | " + " | ".join(cells) + " |")
        run.finish(acc, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

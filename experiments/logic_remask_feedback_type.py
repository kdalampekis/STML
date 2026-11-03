"""Q35: does RE-MASKING (free data, Q17-Q18) rescue BINARY feedback? Q8: with 600 worlds, a right/wrong signal teaches about a fifth of what the verdict does (gain +2.8 vs +14.4 at claims 8-10).
The machine analogue of the human study's arm contrast (scalar vs diagnostic) assumed that gap to be structural. If the gap were just a data limit, 10 re-masked draws should shrink it.

Hidden-premise episodes (50% visible, 10 claims), 300 worlds per level, R = 1 or 10 draws. Binary feedback = whether the history-free judge's guess for the revealed claim was right (cross-fitted flags, as in logic_feedback_type);
verdict = the true verdict as an edge. Static = no feedback.
PREDICTIONS, written before running:
  Y1  at 10 draws the verdict-minus-binary gap at claims 7-10 is still >= +8 points (structural information difference).
  Y2  both binary and verdict gain from 10 draws, binary by less than the verdict (it has less to learn).
  Falsifier of Y1: gap at 10 draws below +5.

    python experiments/logic_remask_feedback_type.py --quick
    python experiments/logic_remask_feedback_type.py
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
from stml.model.logic_flags import correctness_flags, cross_fit_static_probs  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402


def late(m, items, device) -> float:
    y = np.array([x.label for x in items])
    t = np.array([x.t for x in items])
    return float(((logits_of(m, items, device).argmax(1).cpu().numpy() == y).astype(float))[t >= 7].mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, nv, nte, epochs, k = 300, 50, 100, 10, 10
    if a.quick:
        a.seeds, nw, nv, nte, epochs, k = 1, 8, 8, 12, 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_remask_feedback_type" + ("_quick" if a.quick else ""), vars(a)) as run:
        va = make_episodes(levels, range(10 ** 4, 10 ** 4 + nv), k, 0.5, rng_seed=2)
        te = make_episodes(levels, range(1000, 1000 + nte), k, 0.5, rng_seed=3)
        res = {(m, r): [] for m in ("static", "bin", "fb") for r in (1, 10)}
        for seed in range(a.seeds):
            for r in (1, 10):
                tr = sum((make_episodes(levels, range(nw), k, 0.5, rng_seed=100 + i) for i in range(r)), [])
                p_tr, p_va, p_te = cross_fit_static_probs(tr, va, te, device, epochs, seed)
                f_tr, f_va, f_te = (correctness_flags(e, p) for e, p in ((tr, p_tr), (va, p_va), (te, p_te)))
                sets = {"static": (episode_items(tr, "static"), episode_items(va, "static"), episode_items(te, "static")),
                        "bin": (episode_items(tr, "bin", f_tr), episode_items(va, "bin", f_va), episode_items(te, "bin", f_te)),
                        "fb": (episode_items(tr, "fb"), episode_items(va, "fb"), episode_items(te, "fb"))}
                for mode, (a_tr, a_va, a_te) in sets.items():
                    torch.manual_seed(seed)
                    m = LogicJudge(64, 4, n_types=N_TYPES[mode]).to(device)
                    fit(m, a_tr, a_va, device, epochs=epochs, seed=seed, log=lambda s: None)
                    res[(mode, r)].append(late(m, a_te, device))
            run.log(f"seed {seed}: " + ", ".join(f"{m}@R{r} {v[-1]:.1%}" for (m, r), v in res.items()))
        L = ["# Re-masking and the feedback type", "", f"{a.seeds} seeds; {len(te)} test episodes x {k} claims; accuracy at claims 7-10 (mean, sd). Predictions Y1-Y2 in the script.", "",
             "| feedback | 1 draw | 10 draws |", "| --- | ---: | ---: |"]
        for mode in ("static", "bin", "fb"):
            L.append(f"| {mode} | " + " | ".join(f"{100 * np.mean(res[(mode, r)]):.1f}% ({100 * np.std(res[(mode, r)]):.1f})" for r in (1, 10)) + " |")
        L += ["", "| comparison | 1 draw | 10 draws |", "| --- | ---: | ---: |"]
        for x, y_ in (("fb", "bin"), ("bin", "static"), ("fb", "static")):
            cells = []
            for r in (1, 10):
                d = 100 * (np.array(res[(x, r)]) - np.array(res[(y_, r)]))
                cells.append(f"{d.mean():+.1f} ({d.std():.1f})")
            L.append(f"| {x} minus {y_} | " + " | ".join(cells) + " |")
        run.finish({f"{m}@R{r}": v for (m, r), v in res.items()}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

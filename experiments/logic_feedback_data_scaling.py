"""Data scaling of the feedback learner, and a test of the DILUTION explanation for the null results of structured feedback (Q5).

Signed surprise (as 72 edge types) and 'which premises were used' (marks, 8 more premise edge types) both failed to add to the plain verdict, and slightly hurt it. A possible reason is
dilution: more edge types spread the same 600 worlds per level thinner. If so, the penalty should shrink, or the gain appear, with more training data.
Conditions at 600 and 2000 training worlds per level (levels 2-5, 10 claims per episode, 50% visible): static; fb (verdict); fb_why (verdict + premise marks).

PREDICTIONS, written before running:
  H1  the verdict learner's late-position accuracy (claims 7-10) rises by >= 3 points from 600 to 2000 worlds (it is data-limited; the sweep showed +2.4 points per tripling of worlds in the 'gain late').
  H2  (dilution) the gap fb_why - fb at late positions is higher at 2000 worlds than at 600 by >= 1 point.
  Falsifier of H2: the gap is unchanged or lower at 2000 worlds (then the nulls are not a data-dilution effect).

    python experiments/logic_feedback_data_scaling.py --quick
    python experiments/logic_feedback_data_scaling.py
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
from stml.model.logic_episodes import N_TYPES, episode_items, make_episodes, support_marks  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    sizes, epochs, k = (600, 2000), 10, 10
    if a.quick:
        a.seeds, sizes, epochs, k = 1, (24, 48), 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_feedback_data_scaling" + ("_quick" if a.quick else ""), {**vars(a), "sizes": sizes}) as run:
        nv, nt = (50, 100) if not a.quick else (8, 12)
        big = max(sizes)
        tr_all = make_episodes(levels, range(big), k, 0.5, rng_seed=1)                      # per level, seeds 0..big-1; a smaller set takes the first n seeds of each level
        va = make_episodes(levels, range(big, big + nv), k, 0.5, rng_seed=2)
        te = make_episodes(levels, range(5000, 5000 + nt), k, 0.5, rng_seed=3)
        per_level = {lv: [e for e in tr_all if e.level == lv] for lv in levels}
        mk_marks = lambda eps: [support_marks(e) for e in eps]
        va_m, te_m = mk_marks(va), mk_marks(te)
        R = {}
        for seed in range(a.seeds):
            for n in sizes:
                tr = [e for lv in levels for e in per_level[lv][:n]]
                tr_m = mk_marks(tr)
                for c, mode in (("static", "static"), ("fb", "fb"), ("fb_why", "fb_why")):
                    kw = lambda eps, m: {"marks": m} if mode == "fb_why" else {}
                    a_tr, a_va, a_te = episode_items(tr, mode, **kw(tr, tr_m)), episode_items(va, mode, **kw(va, va_m)), episode_items(te, mode, **kw(te, te_m))
                    torch.manual_seed(seed)
                    m = LogicJudge(64, 4, n_types=N_TYPES[mode]).to(device)
                    fit(m, a_tr, a_va, device, epochs=epochs, seed=seed, log=lambda s: None)
                    y = np.array([x.label for x in a_te])
                    tt = np.array([x.t for x in a_te])
                    right = (logits_of(m, a_te, device).argmax(1).cpu().numpy() == y).astype(float)
                    late = tt >= int(0.7 * k)
                    R.setdefault((c, n), {"late": [], "all": []})
                    R[(c, n)]["late"].append(float(right[late].mean()))
                    R[(c, n)]["all"].append(float(right.mean()))
            run.log(f"seed {seed}: " + ", ".join(f"{c}@{n} {R[(c, n)]['late'][-1]:.1%}" for c in ("static", "fb", "fb_why") for n in sizes))
        L = ["# Data scaling of the feedback learner and the dilution test", "", f"{a.seeds} seeds; accuracy at the late claims (7-10 of 10) on {len(te)} test episodes; training worlds per level in the columns. Predictions H1-H2 in the script.", "",
             "| condition | " + " | ".join(f"{n} worlds" for n in sizes) + " |", "| --- | " + " | ".join("---:" for _ in sizes) + " |"]
        for c in ("static", "fb", "fb_why"):
            L.append(f"| {c} | " + " | ".join(f"{100 * np.mean(R[(c, n)]['late']):.1f}% ({100 * np.std(R[(c, n)]['late']):.1f})" for n in sizes) + " |")
        L += ["", "| paired difference (points) | " + " | ".join(f"{n} worlds" for n in sizes) + " |", "| --- | " + " | ".join("---:" for _ in sizes) + " |"]
        for x, y_ in (("fb", "static"), ("fb_why", "fb")):
            L.append(f"| {x} minus {y_} | " + " | ".join(f"{(d := 100 * (np.array(R[(x, n)]['late']) - np.array(R[(y_, n)]['late']))).mean():+.1f} ({d.std():.1f})" for n in sizes) + " |")
        run.finish({f"{c}@{n}": v for (c, n), v in R.items()}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

"""Does telling the learner WHICH visible premises a revealed proof used add anything beyond the verdict? (Q8b; the 'why' part of the diagnostic arm)

Hidden-premise episodes as in logic_feedback_type. After each judgement the learner gets:
  static        nothing
  fb            the true verdict (added to the graph as an edge)
  fb_why        the verdict AND the marking of the visible premises that the minimal proof of every revealed follows/contradicted claim used (the premise edges change type);
                hidden premises used in a proof are not shown (the learner cannot see them)
  fb_why_rand   control: the same NUMBER of marked premises, chosen at random
PREDICTIONS, written before running:
  G1  fb_why beats fb by >= 1 point at late positions (claims 7-10) but by < 4 (the supervised H2 test found +0.6 to +1.4 for 'why' beyond the verdict).
  G2  fb_why_rand is not better than fb by more than the seed sd (a marking carries information only if it points at the right premises).
  Falsifier of G1: fb_why - fb within the seed sd at late positions.

    python experiments/logic_feedback_why.py --quick
    python experiments/logic_feedback_why.py
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

BANDS = {"t=0": (0, 0), "t=1-3": (1, 3), "t=4-6": (4, 6), "t=7-9": (7, 9)}
COND = ["static", "fb", "fb_why", "fb_why_rand"]


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
    with Run("logic_feedback_why" + ("_quick" if a.quick else ""), {**vars(a), "conditions": COND}) as run:
        tr = make_episodes(levels, range(nw[0]), k, 0.5, rng_seed=1)
        va = make_episodes(levels, range(nw[0], nw[0] + nw[1]), k, 0.5, rng_seed=2)
        te = make_episodes(levels, range(1000, 1000 + nw[2]), k, 0.5, rng_seed=3)
        marks = {n: [support_marks(e) for e in eps] for n, eps in (("tr", tr), ("va", va), ("te", te))}
        rmarks = {n: [support_marks(e, True, i) for i, e in enumerate(eps)] for n, eps in (("tr", tr), ("va", va), ("te", te))}
        share = np.mean([len(m[-1]) / max(1, len(e.visible)) for e, m in zip(te, marks["te"])])
        run.log(f"at the end of an episode {share:.0%} of the visible premises are marked on average")
        acc = {c: {b: [] for b in list(BANDS) + ["all"]} for c in COND}
        for seed in range(a.seeds):
            sets = {"static": tuple(episode_items(x, "static") for x in (tr, va, te)) + ("static",),
                    "fb": tuple(episode_items(x, "fb") for x in (tr, va, te)) + ("fb",),
                    "fb_why": tuple(episode_items(x, "fb_why", marks=marks[n]) for n, x in (("tr", tr), ("va", va), ("te", te))) + ("fb_why",),
                    "fb_why_rand": tuple(episode_items(x, "fb_why", marks=rmarks[n]) for n, x in (("tr", tr), ("va", va), ("te", te))) + ("fb_why",)}
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
        L = ["# Does 'which premises were used' add to the verdict?", "", f"{a.seeds} seeds; {len(te)} test episodes x {k} claims; accuracy by position (mean, sd). Predictions G1-G2 in the script. "
             f"At the end of an episode {share:.0%} of the visible premises are marked on average.", "",
             "| condition | " + " | ".join(list(BANDS) + ["all"]) + " |", "| --- | " + " | ".join("---:" for _ in range(len(BANDS) + 1)) + " |"]
        for c in COND:
            L.append(f"| {c} | " + " | ".join(f"{100 * np.mean(acc[c][b]):.1f}% ({100 * np.std(acc[c][b]):.1f})" for b in list(BANDS) + ["all"]) + " |")
        L += ["", "## Paired differences (points; sd of the paired difference)", "", "| comparison | " + " | ".join(list(BANDS) + ["all"]) + " |", "| --- | " + " | ".join("---:" for _ in range(len(BANDS) + 1)) + " |"]
        for x, y_ in (("fb_why", "fb"), ("fb_why_rand", "fb"), ("fb_why", "fb_why_rand"), ("fb", "static")):
            L.append(f"| {x} minus {y_} | " + " | ".join(f"{(d := 100 * (np.array(acc[x][b]) - np.array(acc[y_][b]))).mean():+.1f} ({d.std():.1f})" for b in list(BANDS) + ["all"]) + " |")
        run.finish(acc, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

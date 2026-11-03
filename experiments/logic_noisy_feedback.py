"""Q11: robustness to NOISY feedback. Each revealed verdict is replaced, with probability p, by one of the two wrong verdicts (a wrong teacher / a misread critic).

Hidden-premise episodes as in logic_feedback_type (verdict revealed as a graph edge). Conditions, accuracy at claims 7-10 on test episodes whose revealed verdicts are noisy with rate p_test:
  matched(p)    trained with the same noise rate p, p in {0, 0.1, 0.2, 0.3}
  clean_on(p)   trained on clean feedback (p = 0), tested with noise p  (a learner that trusts its teacher)
  static        no feedback (reference)
The learner is not told which verdicts are wrong. Targets (the claims to judge) are always the true verdicts.
PREDICTIONS, written before running:
  N1  the feedback gain over static falls with p: matched gain at p = 0.1 / 0.2 / 0.3 is about 85% / 65% / 45% of the clean gain (a wrong edge misleads about as often as a right one helps;
      allow +-15 points of percentage); it stays positive at p = 0.3.
  N2  a learner trained on clean feedback loses more than a matched one: matched(p) minus clean_on(p) >= +1.5 points at p = 0.2 and 0.3 (training on noise teaches it to discount the edges).
  Falsifier of N2: matched - clean_on within the seed sd at p = 0.3.
  Falsifier of N1: the gain at p = 0.3 is gone (<= 1 point) or is still > 90% of the clean gain.

    python experiments/logic_noisy_feedback.py --quick
    python experiments/logic_noisy_feedback.py
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

PS = [0.0, 0.1, 0.2, 0.3]


def late_acc(m, items, device) -> float:
    y = np.array([x.label for x in items])
    t = np.array([x.t for x in items])
    right = (logits_of(m, items, device).argmax(1).cpu().numpy() == y).astype(float)
    return float(right[t >= 6].mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--worlds", type=int, default=600, help="training worlds per level")
    ap.add_argument("--ps", default="0.0,0.1,0.2,0.3", help="noise rates; the first must be 0.0")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    PS[:] = [float(x) for x in a.ps.split(',')]
    nw, epochs, k = (a.worlds, 50, 100), 10, 10
    if a.quick:
        a.seeds, nw, epochs, k = 1, (24, 8, 12), 2, 8
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_noisy_feedback" + ("_quick" if a.quick else ""), {**vars(a), "ps": PS}) as run:
        tr = make_episodes(levels, range(nw[0]), k, 0.5, rng_seed=1)
        va = make_episodes(levels, range(nw[0], nw[0] + nw[1]), k, 0.5, rng_seed=2)
        te = make_episodes(levels, range(1000, 1000 + nw[2]), k, 0.5, rng_seed=3)
        res = {"static": [], **{f"matched_{p}": [] for p in PS}, **{f"clean_on_{p}": [] for p in PS[1:]}}
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            ms = LogicJudge(64, 4, n_types=N_TYPES["static"]).to(device)
            fit(ms, episode_items(tr, "static"), episode_items(va, "static"), device, epochs=epochs, seed=seed, log=lambda s: None)
            res["static"].append(late_acc(ms, episode_items(te, "static"), device))
            clean_model = None
            for p in PS:
                torch.manual_seed(seed)
                m = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
                fit(m, episode_items(noisy_feedback(tr, p, 10 * seed + 1), "fb"), episode_items(noisy_feedback(va, p, 10 * seed + 2), "fb"), device, epochs=epochs, seed=seed, log=lambda s: None)
                res[f"matched_{p}"].append(late_acc(m, episode_items(noisy_feedback(te, p, 10 * seed + 3), "fb"), device))
                if p == 0.0:
                    clean_model = m
                else:
                    res[f"clean_on_{p}"].append(late_acc(clean_model, episode_items(noisy_feedback(te, p, 10 * seed + 3), "fb"), device))
            run.log(f"seed {seed}: " + ", ".join(f"{k_} {v[-1]:.1%}" for k_, v in res.items()))
        L = ["# Noisy feedback", "", f"{a.seeds} seeds; {len(te)} test episodes x {k} claims; accuracy at claims 7-10 (mean, sd over seeds). Predictions N1-N2 in the script.", "", "| condition | accuracy |", "| --- | ---: |"]
        for c, v in res.items():
            L.append(f"| {c} | {100 * np.mean(v):.1f}% ({100 * np.std(v):.1f}) |")
        L += ["", "## Gain over static and share of the clean gain", "", "| p | matched gain | share of clean gain | clean-trained gain | matched minus clean-trained |", "| --- | ---: | ---: | ---: | ---: |"]
        st = np.array(res["static"])
        g0 = 100 * (np.array(res["matched_0.0"]) - st).mean()
        for p in PS:
            gm = 100 * (np.array(res[f"matched_{p}"]) - st)
            if p == 0.0:
                L.append(f"| 0.0 | {gm.mean():+.1f} ({gm.std():.1f}) | 100% | – | – |")
            else:
                gc = 100 * (np.array(res[f"clean_on_{p}"]) - st)
                d = 100 * (np.array(res[f"matched_{p}"]) - np.array(res[f"clean_on_{p}"]))
                L.append(f"| {p} | {gm.mean():+.1f} ({gm.std():.1f}) | {100 * gm.mean() / g0:.0f}% | {gc.mean():+.1f} ({gc.std():.1f}) | {d.mean():+.1f} ({d.std():.1f}) |")
        run.finish(res, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

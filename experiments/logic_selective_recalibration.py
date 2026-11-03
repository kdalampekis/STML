"""Q13: exploration-aware recalibration of the abstention threshold. A deployed learner only sees right/wrong for the answers it COMMITS to.

Offline replay on the saved scores of `2026-10-14_195306_logic_h3_thresholds` (5 judge seeds, small and extended training, trees and chains of 16/24/32 kinds; the scores are
temperature-calibrated confidences; labels are regenerated from the same seeds). Items arrive as a stream in random order; the learner commits when confidence > t, else abstains
(-0.2); right +1, wrong -lam. Strategies for t (re-fitted every 25 items from what has been observed):
  fixed        the Bayes threshold of a calibrated learner, t0 = (lam + 0.2) / (1 + lam): no adaptation
  selective_0  re-fit on committed items only, thresholds restricted to the observed score range (no information below it); no exploration
  selective_e  as selective_0 and, with probability e (0.05, 0.1, 0.2), commit to an item below t anyway to see its label (the exploration costs its realised reward)
  full_labels  re-fit on ALL items seen (right/wrong shown whether committed or not; an upper reference, not deployable)
  oracle       best fixed threshold in hindsight on the whole stream
Metric: mean reward per item over the stream (200 random orders per set and model seed), reported as the average over the 6 test sets x 5 models x lam in {1, 2, 4}, small and extended regimes.
PREDICTIONS, written before running:
  R1  Under shift to larger worlds the calibrated learner is OVERCONFIDENT, so the useful correction is to raise t, which committed items alone can reveal: selective_0 recovers >= 70% of the gain of
      full_labels over fixed (averaged over sets where full_labels beats fixed by > 0.02).
  R2  Exploration does not pay: selective_e (any e) is not better than selective_0 by more than 0.01.
  Falsifier of R1: selective_0 recovers < 50%.
Q13b (--margins 0.02,0.05,0.1): a re-fitted threshold is adopted only if it beats the current one by `margin` on the observed items (hysteresis against the ratchet seen in Q13).
  PREDICTION: margin 0.05 removes >= 70% of selective_0's loss in the calibrated cells (extended regime, lam <= 2) and keeps >= 80% of its gain in the 8 overconfident cells.

    python experiments/logic_selective_recalibration.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stml.model.logic_data as ld  # noqa: E402
from stml.model.logic_data import make_big_items  # noqa: E402
from stml.runlog import Run  # noqa: E402

K = 0.2
LAMS = (1.0, 2.0, 4.0)
BLOCK = 25
EPS = (0.05, 0.1, 0.2)
MARGINS = ()           # Q13b: hysteresis margins, e.g. --margins 0.02,0.05,0.1
SRC = ROOT / "runs" / "2026-10-14_195306_logic_h3_thresholds"


def reward(score, right, t, lam):
    return np.where(score > t, np.where(right == 1, 1.0, -lam), -K)


def refit(score, right, lam, floor):
    """Best threshold among observed scores >= floor, judged on the observed items only."""
    if len(score) < 20:
        return None
    cands = np.unique(np.quantile(score, np.linspace(0, 1, 51)))
    cands = cands[cands >= floor]
    if len(cands) == 0:
        return None
    rew = np.where(score[None, :] > cands[:, None], np.where(right == 1, 1.0, -lam)[None, :], -K).mean(1)
    return float(cands[int(rew.argmax())])


def stream(score, right, lam, mode, eps, rng, margin=0.0):
    """Mean reward of one pass; mode in fixed / selective / full."""
    t = (lam + K) / (1 + lam)
    obs_s, obs_r, tot = [], [], 0.0
    for start in range(0, len(score), BLOCK):
        s, r = score[start:start + BLOCK], right[start:start + BLOCK]
        commit = s > t
        explore = (~commit) & (rng.random(len(s)) < eps) if mode == "selective" else np.zeros(len(s), bool)
        act = commit | explore
        tot += float(np.where(act, np.where(r == 1, 1.0, -lam), -K).sum())
        seen = np.ones(len(s), bool) if mode == "full" else act
        obs_s += list(s[seen])
        obs_r += list(r[seen])
        if mode != "fixed":
            floor = -np.inf if mode == "full" else min(obs_s) if obs_s else np.inf
            new = refit(np.array(obs_s), np.array(obs_r), lam, floor)
            if new is not None and (margin == 0.0 or reward(np.array(obs_s), np.array(obs_r), new, lam).mean() - reward(np.array(obs_s), np.array(obs_r), t, lam).mean() > margin):
                t = new
    return tot / len(score)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--margins", default="", help="Q13b: comma-separated hysteresis margins (adds selective_h<m> strategies, no exploration)")
    a = ap.parse_args()
    global MARGINS
    MARGINS = tuple(float(x) for x in a.margins.split(',') if x)
    ld.SIZE_FREE = True
    if a.quick:
        a.reps = 5
    sizes, ntest, seeds = (16, 24, 32), 30, 5
    rng = np.random.default_rng(0)
    with Run("logic_selective_recalibration" + ("_quick" if a.quick else ""), {**vars(a), "src": str(SRC)}) as run:
        tests = {}
        cache = ROOT / "data" / "processed" / "selective_recalibration_labels.npz"      # SAT labelling of the 24/32-kind worlds takes minutes
        have = dict(np.load(cache)) if cache.exists() else {}
        for fam, chain in (("tree", 0.0), ("chain", 0.9)):
            for n in sizes:
                key = f"{fam}-{n}"
                if key not in have:
                    have[key] = np.array([it.label for it in make_big_items(n, range(1000, 1000 + ntest), rng_seed=n, chain=chain)])
                    np.savez_compressed(cache, **have)
                    run.log(f"labels for {key}: {len(have[key])}")
                tests[(fam, n)] = have[key]
        names = ["fixed", "selective_0"] + [f"selective_{e}" for e in EPS] + [f"selective_h{m}" for m in MARGINS] + ["full_labels", "oracle"]
        rows = []
        for regime in ("small", "extended"):
            z = np.load(SRC / f"scores_{regime}.npz")
            for (fam, n), y in tests.items():
                for lam in LAMS:
                    acc = {m: [] for m in names}
                    for s in range(1 if a.quick else seeds):
                        probs = z[f"{fam}-{n}-probs-{s}"]
                        conf = probs.max(1)
                        right = (probs.argmax(1) == y).astype(float)
                        for _ in range(a.reps):
                            p = rng.permutation(len(y))
                            sc, rt = conf[p], right[p]
                            acc["fixed"].append(stream(sc, rt, lam, "fixed", 0, rng))
                            acc["selective_0"].append(stream(sc, rt, lam, "selective", 0.0, rng))
                            for e in EPS:
                                acc[f"selective_{e}"].append(stream(sc, rt, lam, "selective", e, rng))
                            for m_ in MARGINS:
                                acc[f"selective_h{m_}"].append(stream(sc, rt, lam, "selective", 0.0, rng, m_))
                            acc["full_labels"].append(stream(sc, rt, lam, "full", 0, rng))
                        cands = np.unique(np.quantile(conf, np.linspace(0, 1, 101)))
                        acc["oracle"].append(max(float(reward(conf, right, c, lam).mean()) for c in cands))
                    rows.append({"regime": regime, "set": f"{fam}-{n}", "lam": lam, **{m: float(np.mean(v)) for m, v in acc.items()}})
            run.log(f"{regime} done")
        # aggregate
        g = np.array([[r[m] for m in names] for r in rows])
        gain_full = g[:, names.index("full_labels")] - g[:, names.index("fixed")]
        use = gain_full > 0.02
        rec = (g[use, names.index("selective_0")] - g[use, names.index("fixed")]).sum() / gain_full[use].sum() if use.any() else float("nan")
        L = ["# Selective recalibration (labels only for committed answers)", "", f"Offline replay of {len(rows)} (regime x set x lam) cells, {a.reps} random stream orders x {seeds} judge seeds each; mean reward per item. Predictions R1-R2 in the script.", "",
             "| strategy | mean reward (all cells) | mean gain over fixed |", "| --- | ---: | ---: |"]
        for j, m in enumerate(names):
            L.append(f"| {m} | {g[:, j].mean():+.3f} | {(g[:, j] - g[:, 0]).mean():+.3f} |")
        L += ["", f"Cells where full-label recalibration beats the fixed threshold by > 0.02: {int(use.sum())} of {len(rows)}. In those, selective_0 recovers {100 * rec:.0f}% of the full-label gain.", "",
              "## Per cell (lam, regime, set): fixed / selective_0 / selective_0.1 / full_labels / oracle", "", "| regime | set | lam | fixed | selective_0 | selective_0.1 | full | oracle |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in rows:
            L.append(f"| {r['regime']} | {r['set']} | {r['lam']:.0f} | {r['fixed']:+.3f} | {r['selective_0']:+.3f} | {r['selective_0.1']:+.3f} | {r['full_labels']:+.3f} | {r['oracle']:+.3f} |")
        if MARGINS:
            cal = np.array([r["regime"] == "extended" and r["lam"] <= 2 for r in rows])
            oc = use
            loss0 = (g[cal, 0] - g[cal, 1]).sum()
            L += ["", "## Hysteresis (Q13b)", "", f"Calibrated cells (extended, lam <= 2): {int(cal.sum())}; selective_0 loss vs fixed = {loss0 / cal.sum():+.3f} per cell. Overconfident cells: {int(oc.sum())}; selective_0 gain = {(g[oc, 1] - g[oc, 0]).sum() / oc.sum():+.3f} per cell.", "",
                  "| margin | share of the calibrated-cell loss removed | share of the overconfident-cell gain kept |", "| --- | ---: | ---: |"]
            for m_ in MARGINS:
                j = names.index(f"selective_h{m_}")
                L.append(f"| {m_} | {100 * (1 - (g[cal, 0] - g[cal, j]).sum() / loss0):.0f}% | {100 * (g[oc, j] - g[oc, 0]).sum() / (g[oc, 1] - g[oc, 0]).sum():.0f}% |")
        run.finish({"rows": rows, "recovery": float(rec)}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

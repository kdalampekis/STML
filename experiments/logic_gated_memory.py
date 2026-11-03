"""Q6: a surprise-GATED episodic store. Is a limited memory that keeps only the surprising verdicts as good as keeping all of them?

Hidden-premise episodes as in logic_feedback_type. After each judgement the true verdict is revealed, but the learner STORES it (adds it to the graph) only if
  fb_all         always (the reference: unlimited memory)
  store_surprise only if the history-free judge's guess was WRONG (a prediction error; about a third of the claims)
  store_random   a random subset of the SAME size per episode as store_surprise (capacity-matched control)
  store_expected only if the history-free judge's guess was RIGHT (the opposite gate)
  static         nothing stored
PREDICTIONS, written before running (accuracy at claims 7-10, points):
  M1  store_surprise beats store_random (same capacity) by >= 2: the wrongly-guessed claims are exactly where hidden structure sits.
  M2  store_surprise is below fb_all by >= 1 (gating discards information; it can only match fb_all if the unsurprising verdicts are redundant).
  M3  store_expected is not better than store_random by more than the seed sd.
  Falsifier of M1: store_surprise - store_random within the seed sd (the gate is no better than a random memory of the same size).

Q12 (--noise 0.2): each revealed verdict is wrong with probability 0.2 and 'surprise' is judged against the SHOWN verdict (a wrong verdict is therefore always surprising).
  PREDICTIONS, written before running: N3 store_surprise minus store_random at claims 7-10 is <= +1 (it was +2.2 with exact feedback) and may reverse (about 35% of the stored verdicts would be wrong, against 20% overall);
  falsifier: still >= +2 with noise.

    python experiments/logic_gated_memory.py --quick
    python experiments/logic_gated_memory.py
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
from stml.model.logic_episodes import N_TYPES, episode_items, make_episodes, noisy_feedback  # noqa: E402
from stml.model.logic_flags import cross_fit_static_probs  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

BANDS = {"t=0": (0, 0), "t=1-3": (1, 3), "t=4-6": (4, 6), "t=7-9": (7, 9)}
COND = ["static", "fb_all", "store_surprise", "store_random", "store_expected"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0, help="probability that a revealed verdict is wrong (Q12); surprise is then judged against the SHOWN verdict")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, epochs, k = (600, 50, 100), 10, 10
    if a.quick:
        a.seeds, nw, epochs, k = 1, (24, 8, 12), 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_gated_memory" + (f"_noise{a.noise}" if a.noise else "") + ("_quick" if a.quick else ""), {**vars(a), "conditions": COND}) as run:
        tr = make_episodes(levels, range(nw[0]), k, 0.5, rng_seed=1)
        va = make_episodes(levels, range(nw[0], nw[0] + nw[1]), k, 0.5, rng_seed=2)
        te = make_episodes(levels, range(1000, 1000 + nw[2]), k, 0.5, rng_seed=3)
        if a.noise > 0:
            tr, va, te = noisy_feedback(tr, a.noise, 11), noisy_feedback(va, a.noise, 12), noisy_feedback(te, a.noise, 13)
        acc = {c: {b: [] for b in list(BANDS) + ["all"]} for c in COND}
        stored = []
        for seed in range(a.seeds):
            p_tr, p_va, p_te = cross_fit_static_probs(tr, va, te, device, epochs, seed)
            def correctness_flags(eps, probs):                   # 1 = the history-free judge's guess matched the verdict the learner is SHOWN
                return [[int(probs[i * k + t].argmax() == (ep.shown or ep.labels)[t]) for t in range(k)] for i, ep in enumerate(eps)]
            f_tr, f_va, f_te = (correctness_flags(e, p) for e, p in ((tr, p_tr), (va, p_va), (te, p_te)))
            rnd = random.Random(seed)
            surprise = lambda fl: [[not x for x in f] for f in fl]
            expected = lambda fl: [[bool(x) for x in f] for f in fl]
            rand = lambda fl: [rnd.sample(m, len(m)) for m in surprise(fl)]       # same number kept per episode, random positions
            stored.append(float(np.mean([np.mean(m) for m in surprise(f_te)])))
            keeps = {"fb_all": (None, None, None), "store_surprise": tuple(map(surprise, (f_tr, f_va, f_te))),
                     "store_random": tuple(map(rand, (f_tr, f_va, f_te))), "store_expected": tuple(map(expected, (f_tr, f_va, f_te)))}
            for c in COND:
                mode = "static" if c == "static" else "fb"
                kk = keeps.get(c, (None, None, None))
                a_tr, a_va, a_te = (episode_items(x, mode, keeps=kp) for x, kp in zip((tr, va, te), kk))
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
        L = ["# Surprise-gated memory", "", f"{a.seeds} seeds; {len(te)} test episodes x {k} claims; accuracy by position (mean, sd). Predictions M1-M3 in the script. "
             f"Share of claims stored by the surprise gate: {100 * np.mean(stored):.0f}%.", "",
             "| condition | " + " | ".join(list(BANDS) + ["all"]) + " |", "| --- | " + " | ".join("---:" for _ in range(len(BANDS) + 1)) + " |"]
        for c in COND:
            L.append(f"| {c} | " + " | ".join(f"{100 * np.mean(acc[c][b]):.1f}% ({100 * np.std(acc[c][b]):.1f})" for b in list(BANDS) + ["all"]) + " |")
        L += ["", "## Paired differences (points; sd of the paired difference)", "", "| comparison | " + " | ".join(list(BANDS) + ["all"]) + " |", "| --- | " + " | ".join("---:" for _ in range(len(BANDS) + 1)) + " |"]
        for x, y_ in (("store_surprise", "store_random"), ("fb_all", "store_surprise"), ("store_expected", "store_random"), ("store_surprise", "static"), ("fb_all", "static")):
            cells = []
            for b in list(BANDS) + ["all"]:
                d = 100 * (np.array(acc[x][b]) - np.array(acc[y_][b]))
                cells.append(f"{d.mean():+.1f} ({d.std():.1f})")
            L.append(f"| {x} minus {y_} | " + " | ".join(cells) + " |")
        run.finish(acc, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

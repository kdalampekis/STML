"""Does an explicit SIGNED SURPRISE signal help a learner that learns from feedback? (C1, the input form; exact labels)

Setting (stml/model/logic_episodes.py): a game world (levels 2-5) of which the learner sees only ~50% of the premises. It judges 10 claims in
a row; after each, the TRUE verdict (from all premises) is revealed and added to the world graph as an extra edge. The learner is a graph model.
Conditions (same architecture family, data and training):
  static   no feedback (visible premises only)                         lower bound
  asked    claims asked but verdict not shown                          control: does the label matter?
  fb       verdict shown                                               feedback, no surprise
  fb_s     verdict shown + signed surprise flavour of the history-free judge (expected / underestimated / overestimated)   <- the test
  fb_shuf  fb_s with the flavours shuffled inside each episode          control: right distribution of flags, wrong claims
  oracle   all premises visible                                          upper bound
Flags for TRAINING claims come from 2-fold cross-fitted static judges (no leakage); for validation and test claims from the average of the same two judges (matched distribution).

PREDICTIONS, written before running:
  Q1  feedback helps: fb beats static by >= 5 accuracy points at late positions (t >= 6).
  Q2  signed surprise helps: fb_s beats fb by >= 2 points at t >= 6, and fb_shuf does not.
  Falsifier for Q2: fb_s - fb within the seed sd, or fb_shuf as good as fb_s (then any extra flag channel acts as a regulariser, as in the edit model).
  RESULT of the first run (conditions static..oracle): Q1 true (+9 / +12 points at late positions); Q2 false (fb_s - fb = -0.7), although fb_s beat its shuffled
  control by +1.7 - the information is there but splitting the edge types into 72 costs about as much as it gains. Follow-up written after seeing that:
  fb_n  delivers the same flags as 3 per-kind node features instead (24 edge types kept); fb_n_shuf is its shuffled control.
  Q3  fb_n beats fb by >= 2 points at t >= 6 and fb_n_shuf does not.

    python experiments/logic_surprise_input.py --quick
    python experiments/logic_surprise_input.py
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
from stml.model.logic_episodes import N_TYPES, NODE_DIM_N, episode_items, flavour_of, make_episodes, oracle_items  # noqa: E402
from stml.model.logic_judge import LogicJudge, ece, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

BANDS = {"t=0": (0, 0), "t=1-3": (1, 3), "t=4-6": (4, 6), "t=7-9": (7, 9)}
COND = ["static", "asked", "fb", "fb_s", "fb_shuf", "fb_n", "fb_n_shuf", "oracle"]


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def flavours_from(eps, static_items, probs) -> list[list[int]]:
    k = len(eps[0].queries)
    return [[flavour_of(probs[i * k + t], ep.labels[t]) for t in range(k)] for i, ep in enumerate(eps)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--vis", type=float, default=0.5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, epochs, k = (300, 50, 100), 10, 10
    if a.quick:
        a.seeds, nw, epochs, k = 1, (24, 8, 12), 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_surprise_input" + ("_quick" if a.quick else ""), {**vars(a), "conditions": COND, "k": k}) as run:
        tr = make_episodes(levels, range(nw[0]), k, a.vis, rng_seed=1)
        va = make_episodes(levels, range(nw[0], nw[0] + nw[1]), k, a.vis, rng_seed=2)
        te = make_episodes(levels, range(1000, 1000 + nw[2]), k, a.vis, rng_seed=3)
        run.log(f"episodes: train {len(tr)}, val {len(va)}, test {len(te)}; {k} claims each; visible fraction {a.vis}")
        acc = {c: {b: [] for b in list(BANDS) + ["all"]} for c in COND}
        ece_all = {c: [] for c in COND}
        for seed in range(a.seeds):
            # ---- history-free judges: 2-fold cross-fit for training flags, full model for val/test flags
            st_tr, st_va, st_te = (episode_items(x, "static") for x in (tr, va, te))
            half = [i % 2 for i in range(len(tr))]
            probs_tr = np.zeros((len(st_tr), 3))
            probs_va, probs_te = np.zeros((len(st_va), 3)), np.zeros((len(st_te), 3))
            for h in (0, 1):
                fit_eps = [e for e, g in zip(tr, half) if g != h]
                held = [i for i, g in enumerate(half) if g == h]
                torch.manual_seed(seed)
                m = LogicJudge(64, 4, n_types=N_TYPES["static"]).to(device)
                fit(m, episode_items(fit_eps, "static"), st_va, device, epochs=epochs, seed=seed, log=lambda s: None)
                idx = [i * k + t for i in held for t in range(k)]
                probs_tr[idx] = softmax(logits_of(m, [st_tr[j] for j in idx], device).cpu().numpy())
                probs_va += softmax(logits_of(m, st_va, device).cpu().numpy()) / 2       # val / test flags: average of the two half-data judges,
                probs_te += softmax(logits_of(m, st_te, device).cpu().numpy()) / 2       # so their distribution matches the cross-fitted training flags
            fl_tr, fl_va, fl_te = flavours_from(tr, st_tr, probs_tr), flavours_from(va, st_va, probs_va), flavours_from(te, st_te, probs_te)
            rnd = random.Random(seed)
            shuf = lambda fl: [rnd.sample(f, len(f)) for f in fl]
            sets = {"static": (st_tr, st_va, st_te, "static"),
                    "asked": tuple(episode_items(x, "asked") for x in (tr, va, te)) + ("asked",),
                    "fb": tuple(episode_items(x, "fb") for x in (tr, va, te)) + ("fb",),
                    "fb_s": (episode_items(tr, "fb_s", fl_tr), episode_items(va, "fb_s", fl_va), episode_items(te, "fb_s", fl_te), "fb_s"),
                    "fb_shuf": (episode_items(tr, "fb_s", shuf(fl_tr)), episode_items(va, "fb_s", shuf(fl_va)), episode_items(te, "fb_s", shuf(fl_te)), "fb_s"),
                    "fb_n": (episode_items(tr, "fb_n", fl_tr), episode_items(va, "fb_n", fl_va), episode_items(te, "fb_n", fl_te), "fb_n"),
                    "fb_n_shuf": (episode_items(tr, "fb_n", shuf(fl_tr)), episode_items(va, "fb_n", shuf(fl_va)), episode_items(te, "fb_n", shuf(fl_te)), "fb_n"),
                    "oracle": tuple(oracle_items(x) for x in (tr, va, te)) + ("static",)}
            for c in COND:
                a_tr, a_va, a_te, mode = sets[c]
                torch.manual_seed(seed)
                m = LogicJudge(64, 4, n_types=N_TYPES[mode], node_dim=NODE_DIM_N if mode == "fb_n" else None).to(device)
                fit(m, a_tr, a_va, device, epochs=epochs, seed=seed, log=lambda s: None)
                p = softmax(logits_of(m, a_te, device).cpu().numpy())
                y = np.array([it.label for it in a_te])
                tt = np.array([it.t for it in a_te])
                right = (p.argmax(1) == y).astype(float)
                for b, (lo, hi) in BANDS.items():
                    acc[c][b].append(float(right[(tt >= lo) & (tt <= hi)].mean()))
                acc[c]["all"].append(float(right.mean()))
                ece_all[c].append(ece(p.max(1), right))
            run.log(f"seed {seed} done: " + ", ".join(f"{c} {acc[c]['all'][-1]:.1%}" for c in COND))
        L = ["# Signed surprise as an input to a learner that learns from feedback (logic, hidden premises)", "",
             f"{a.seeds} seeds; {len(te)} test episodes x {k} claims; learner sees {a.vis:.0%} of the premises; accuracy by position t of the claim in the episode (mean, sd). "
             "Predictions Q1-Q2 are in the script docstring.", "",
             "| condition | " + " | ".join(list(BANDS) + ["all", "ECE"]) + " |", "| --- | " + " | ".join("---:" for _ in range(len(BANDS) + 2)) + " |"]
        for c in COND:
            L.append(f"| {c} | " + " | ".join(f"{np.mean(acc[c][b]):.1%} ({np.std(acc[c][b]):.1%})" for b in list(BANDS) + ["all"]) + f" | {np.mean(ece_all[c]):.3f} |")
        L += ["", "## Paired differences (points, mean over seeds; sd of the paired difference in brackets)", "",
              "| comparison | " + " | ".join(list(BANDS) + ["all"]) + " |", "| --- | " + " | ".join("---:" for _ in range(len(BANDS) + 1)) + " |"]
        for x, y_ in (("fb", "static"), ("fb", "asked"), ("fb_s", "fb"), ("fb_shuf", "fb"), ("fb_s", "fb_shuf"), ("fb_n", "fb"), ("fb_n_shuf", "fb"), ("fb_n", "fb_n_shuf"), ("oracle", "fb_n")):
            cells = []
            for b in list(BANDS) + ["all"]:
                d = 100 * (np.array(acc[x][b]) - np.array(acc[y_][b]))
                cells.append(f"{d.mean():+.1f} ({d.std():.1f})")
            L.append(f"| {x} minus {y_} | " + " | ".join(cells) + " |")
        run.finish({"acc": acc, "ece": ece_all}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

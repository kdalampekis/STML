"""Q39: fixing the bottleneck of the lemma scratchpad (Q15b): the judge's own lemmas get less precise as worlds grow (93% -> 82% -> 75% on 16 / 24 / 32-kind chains) and false lemmas cost 13-20 points.
Idea: accept a lemma only if SEVERAL independently trained judges all say 'follows' with probability > tau (agreement of an ensemble of 3), a verifier that needs no labels.

Setting as Q15/Q15b: fully visible worlds, 6-layer judges trained on 5/6/8 kinds (trees and chains, lemma-trained as 'B'), tested on 24- and 32-kind chains and trees; 12 fresh worlds per cell, 5 rounds.
Conditions (accuracy on the 21 balanced target claims): A plain judge (no lemmas); B round 0; single-model lemmas (tau 0.9); ensemble-agreement lemmas (min over 3 models of P(follows) > 0.9); the same with tau 0.97; true-lemma oracle.
PREDICTIONS, written before running (Q15b: 32-kind chains: single-model gain over A +1.1, lemma precision 75%; 24-kind chains +4.1, 82%):
  M1  ensemble agreement raises lemma precision by >= 5 points on 32-kind chains.
  M2  the gain over the plain judge A on 32-kind chains rises to >= +3 (from +1.1) and on 24-kind chains to >= +5.
  Falsifier of M2: the ensemble gain is within the seed sd of the single-model gain.

    python experiments/logic_lemma_ensemble.py --quick
    python experiments/logic_lemma_ensemble.py
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
from stml.model.logic_data import Item, claim_graph  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.logic_lemma import N_TYPES_LEMMA, build_world, item_for, picks, train_items  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

ROUNDS = 5


def rounds(models, w, targets, tau, device, agree: bool, oracle_rng=None):
    """Per round (accuracy on targets, lemma precision, lemma count). agree: lemma needs P(follows) > tau under EVERY model; else the first model alone."""
    ent = [i for i, l in enumerate(w.labels) if l == 0]
    lem = set(i for i in ent if oracle_rng.random() < 0.3) if oracle_rng is not None else set()
    out = []
    use = models if agree else models[:1]
    for r in range(ROUNDS + 1):
        items = [item_for(w, j, [w.pool[i] for i in lem if i != j]) for j in range(len(w.pool))]
        probs = np.stack([torch.softmax(logits_of(m, items, device), 1).cpu().numpy() for m in use])
        p0 = probs.mean(0)
        acc = float(np.mean([p0[j].argmax() == w.labels[j] for j in targets]))
        prec = float(np.mean([w.labels[i] == 0 for i in lem])) if lem else float("nan")
        out.append((acc, prec, len(lem)))
        if oracle_rng is not None:
            break
        lem = {i for i in range(len(w.pool)) if probs[:, i, 0].min() > tau}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--nw", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--members", type=int, default=3)
    a = ap.parse_args()
    ld.SIZE_FREE = True
    train_sizes, ntest = (5, 6, 8), 12
    cells = [("chain", 0.9, 24), ("chain", 0.9, 32), ("tree", 0.0, 24), ("tree", 0.0, 32)]
    if a.quick:
        a.seeds, a.nw, a.epochs, a.members, train_sizes, ntest = 1, 6, 2, 2, (5, 6), 3
        cells = [("chain", 0.9, 10), ("tree", 0.0, 10)]
    device = pick_device(None)
    conds = ["A_plain", "B_round0", "single_0.9", "ensemble_0.9", "ensemble_0.97", "oracle"]
    with Run("logic_lemma_ensemble" + ("_quick" if a.quick else ""), {**vars(a), "cells": cells}) as run:
        fams = (("tree", 0.0), ("chain", 0.9))
        tr_w = [build_world(n, s, c) for n in train_sizes for _, c in fams for s in range(a.nw)]
        va_w = [build_world(n, 5000 + s, c) for n in train_sizes for _, c in fams for s in range(max(3, a.nw // 12))]
        tests = {(f, n): [build_world(n, 1000 + s, c) for s in range(ntest)] for f, c, n in cells}
        tpicks = {k: [picks(w, random.Random(7 + i)) for i, w in enumerate(ws)] for k, ws in tests.items()}
        acc = {k: {c: [] for c in conds} for k in tests}
        prec = {k: {c: [] for c in ("single_0.9", "ensemble_0.9", "ensemble_0.97")} for k in tests}
        for seed in range(a.seeds):
            rng = random.Random(seed)
            torch.manual_seed(seed)
            mA = LogicJudge(64, 6, n_types=8).to(device)
            fit(mA, train_items(tr_w, random.Random(seed), False), train_items(va_w, random.Random(seed + 1), False), device, epochs=a.epochs, seed=seed, log=lambda s: None)
            members = []
            for k in range(a.members):
                torch.manual_seed(100 * seed + k)
                m = LogicJudge(64, 6, n_types=N_TYPES_LEMMA).to(device)
                fit(m, train_items(tr_w, random.Random(seed * 10 + k), True), train_items(va_w, random.Random(seed * 10 + k + 1), True), device, epochs=a.epochs, seed=100 * seed + k, log=lambda s: None)
                members.append(m)
            run.log(f"seed {seed}: trained A and {a.members} lemma judges")
            for key, ws in tests.items():
                per = {c: [] for c in conds}
                pr = {c: [] for c in prec[key]}
                for w, tg in zip(ws, tpicks[key]):
                    items = [Item(w.n, w.seed, w.n, w.pool[j], w.labels[j], 0, claim_graph(w.n, w.premises, w.pool[j]), len(w.premises), (), ()) for j in tg]
                    per["A_plain"].append(float((logits_of(mA, items, device).argmax(1).cpu().numpy() == np.array([w.labels[j] for j in tg])).mean()))
                    s1 = rounds(members, w, tg, 0.9, device, False)
                    e9 = rounds(members, w, tg, 0.9, device, True)
                    e97 = rounds(members, w, tg, 0.97, device, True)
                    orc = rounds(members, w, tg, 0.9, device, False, oracle_rng=rng)
                    per["B_round0"].append(s1[0][0])
                    per["single_0.9"].append(s1[-1][0])
                    per["ensemble_0.9"].append(e9[-1][0])
                    per["ensemble_0.97"].append(e97[-1][0])
                    per["oracle"].append(orc[0][0])
                    pr["single_0.9"].append(s1[-1][1])
                    pr["ensemble_0.9"].append(e9[-1][1])
                    pr["ensemble_0.97"].append(e97[-1][1])
                for c in conds:
                    acc[key][c].append(float(np.mean(per[c])))
                for c in pr:
                    prec[key][c].append(float(np.nanmean(pr[c])) if not np.all(np.isnan(pr[c])) else float("nan"))
            run.log(f"seed {seed}: " + ", ".join(f"{k[0][0]}{k[1]}: A {acc[k]['A_plain'][-1]:.0%} single {acc[k]['single_0.9'][-1]:.0%} ens {acc[k]['ensemble_0.9'][-1]:.0%}" for k in tests))
        L = ["# Lemma scratchpad with ensemble agreement", "", f"{a.seeds} seeds; {a.members}-model ensembles; {ntest} fresh test worlds per cell; accuracy on the 21 balanced target claims after {ROUNDS} rounds (mean, sd over seeds). Predictions M1-M2 in the script.", "",
             "| test set | " + " | ".join(conds) + " |", "| --- | " + " | ".join("---:" for _ in conds) + " |"]
        for k in tests:
            L.append(f"| {k[0]}-{k[1]} | " + " | ".join(f"{100 * np.mean(acc[k][c]):.1f}% ({100 * np.std(acc[k][c]):.1f})" for c in conds) + " |")
        L += ["", "| test set | single minus A | ensemble 0.9 minus A | ensemble 0.97 minus A | ensemble 0.9 minus single | precision single / ens 0.9 / ens 0.97 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
        for k in tests:
            d = lambda x, y: 100 * (np.array(acc[k][x]) - np.array(acc[k][y]))
            f = lambda v: f"{v.mean():+.1f} ({v.std():.1f})"
            L.append(f"| {k[0]}-{k[1]} | {f(d('single_0.9', 'A_plain'))} | {f(d('ensemble_0.9', 'A_plain'))} | {f(d('ensemble_0.97', 'A_plain'))} | {f(d('ensemble_0.9', 'single_0.9'))} | "
                     + " / ".join(f"{100 * np.nanmean(prec[k][c]):.0f}%" for c in ("single_0.9", "ensemble_0.9", "ensemble_0.97")) + " |")
        run.finish({f"{k[0]}-{k[1]}": v for k, v in acc.items()}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

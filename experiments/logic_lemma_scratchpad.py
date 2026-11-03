"""Q15 (out of the box): LEMMAS AS A SCRATCHPAD. Can the judge reason deeper than it was trained for by writing down its own confident conclusions as new edges and re-reading the graph?

Bottleneck targeted: proof depth / larger worlds (judges trained on small worlds collapse on 12-16 kind chains: 56-69%). Idea (chain-of-thought for graphs, cf. Wei et al. 2022 for text;
bootstrapping intermediate facts as in Alphageometry's auxiliary constructions): after judging every claim of a world, the confident 'follows' claims become LEMMA edges (8 extra edge types) and
the world is judged again, so a long proof is assembled from short, reliable steps. All fully visible worlds, no hidden premises; trained on 5-8 kinds (trees and chains), tested on 8, 12, 16 kinds.
Models: A = plain judge (no lemma edges); B = same data/steps, each training item carries a random subset (rate 0 / .15 / .3 / .6) of the world's TRUE lemmas plus ~10% false ones.
Test conditions for B: round 0 (no lemmas), self-lemmas for 1-3 rounds (confidence > 0.9), a random-false-lemma control (same count), and oracle lemmas (a random 30% of the true entailed claims, upper bound).
PREDICTIONS, written before running (accuracy on balanced target claims):
  P1  oracle lemmas raise B well above round 0 on the out-of-range chains (>= +10 points at 12 and 16 kinds): the information is useful when it is right.
  P2  self-lemmas (3 rounds) beat B round 0 by >= 3 points on 12-kind chains and beat A by >= 3, with lemma precision >= 90% in round 1.
  P3  the gain on trees is at most half of that on chains (their proofs are shallow).
  P4  the random-false-lemma control is NOT better than round 0 (lemmas help only when they are true).
  Falsifier of P2: self rounds within the seed sd of round 0, or accuracy falls with rounds (error compounding).

Q15b (--test-sizes 16,24,32 --ntest 12 --rounds 5 --taus 0.8,0.9,0.97 --no-random): same models, larger worlds, more rounds, three thresholds.
  PREDICTIONS, written before running: the self-lemma gain over the plain judge grows with size: >= +6 at 24-kind chains and >= +8 at 32-kind chains; tau 0.97 gives higher lemma precision but a smaller gain than 0.9;
  accuracy does not fall with rounds (no error compounding) up to 5 rounds. Falsifier: gain at 32-kind chains <= the 16-kind gain (+4.9), or accuracy falling with rounds at tau 0.8.

    python experiments/logic_lemma_scratchpad.py --quick
    python experiments/logic_lemma_scratchpad.py
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
from stml.model.logic_judge import LogicJudge, fit  # noqa: E402
from stml.model.logic_lemma import N_TYPES_LEMMA, build_world, picks, self_lemma_rounds, train_items  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

ROUNDS = 3
TAU = 0.9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--nw", type=int, default=150, help="training worlds per size and family")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--test-sizes", default="8,12,16")
    ap.add_argument("--ntest", type=int, default=25)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--taus", default="0.9", help="comma-separated confidence thresholds for self-lemmas (Q15b)")
    ap.add_argument("--no-random", action="store_true", help="skip the random-false-lemma control")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    global ROUNDS
    ROUNDS = a.rounds
    train_sizes, test_sizes, ntest = (5, 6, 8), tuple(int(x) for x in a.test_sizes.split(',')), a.ntest
    taus = [float(x) for x in a.taus.split(',')]
    if a.quick:
        a.seeds, a.nw, a.epochs, train_sizes, test_sizes, ntest = 1, 6, 2, (5, 6), (8, 10), 4
    device = pick_device(None)
    fams = (("tree", 0.0), ("chain", 0.9))
    with Run("logic_lemma_scratchpad" + ("_quick" if a.quick else ""), {**vars(a), "train_sizes": train_sizes, "test_sizes": test_sizes}) as run:
        tr_w = [build_world(n, s, c) for n in train_sizes for _, c in fams for s in range(a.nw)]
        va_w = [build_world(n, 5000 + s, c) for n in train_sizes for _, c in fams for s in range(max(3, a.nw // 12))]
        run.log(f"train worlds {len(tr_w)}, val worlds {len(va_w)}")
        tests = {(f, n): [build_world(n, 1000 + s, c) for s in range(ntest)] for f, c in fams for n in test_sizes}
        tpicks = {k: [picks(w, random.Random(7 + i)) for i, w in enumerate(ws)] for k, ws in tests.items()}
        run.log("test worlds built")
        tl = lambda t: f"@{t}"
        conds = ["A_plain", "B_round0"] + [f"B_self{r}{tl(t)}" for t in taus for r in range(1, ROUNDS + 1)] + ([] if a.no_random else [f"B_random{r}" for r in range(1, ROUNDS + 1)]) + ["B_oracle"]
        acc = {k: {c: [] for c in conds} for k in tests}
        prec = {k: {(t, r): [] for t in taus for r in range(1, ROUNDS + 1)} for k in tests}
        from stml.model.logic_data import Item, claim_graph
        from stml.model.logic_judge import logits_of
        for seed in range(a.seeds):
            rng = random.Random(seed)
            torch.manual_seed(seed)
            models = {}
            for name, use in (("A", False), ("B", True)):
                trn = train_items(tr_w, random.Random(seed), use)
                van = train_items(va_w, random.Random(seed + 1), use)
                m = LogicJudge(64, 6, n_types=N_TYPES_LEMMA if use else 8).to(device)
                fit(m, trn, van, device, epochs=a.epochs, seed=seed, log=lambda s: None)
                models[name] = m
            run.log(f"seed {seed}: trained")
            for k, ws in tests.items():
                per = {c: [] for c in conds}
                pr = {(t, r): [] for t in taus for r in range(1, ROUNDS + 1)}
                for w, tg in zip(ws, tpicks[k]):
                    items = [Item(w.n, w.seed, w.n, w.pool[j], w.labels[j], 0, claim_graph(w.n, w.premises, w.pool[j]), len(w.premises), (), ()) for j in tg]
                    per["A_plain"].append(float((logits_of(models["A"], items, device).argmax(1).cpu().numpy() == np.array([w.labels[j] for j in tg])).mean()))
                    for t in taus:
                        s_self = self_lemma_rounds(models["B"], w, tg, ROUNDS, t, device, "self", rng)
                        per["B_round0"].append(s_self[0][0]) if t == taus[0] else None
                        for r in range(1, ROUNDS + 1):
                            per[f"B_self{r}{tl(t)}"].append(s_self[r][0])
                            pr[(t, r)].append(s_self[r][1])
                    if not a.no_random:
                        s_rand = self_lemma_rounds(models["B"], w, tg, ROUNDS, taus[0], device, "random", rng)
                        for r in range(1, ROUNDS + 1):
                            per[f"B_random{r}"].append(s_rand[r][0])
                    per["B_oracle"].append(self_lemma_rounds(models["B"], w, tg, 0, taus[0], device, "oracle", rng)[0][0])
                for c in conds:
                    acc[k][c].append(float(np.mean(per[c])))
                for key in pr:
                    prec[k][key].append(float(np.nanmean(pr[key])) if not np.all(np.isnan(pr[key])) else float("nan"))
            run.log(f"seed {seed}: " + ", ".join(f"{k[0][0]}{k[1]}: A {acc[k]['A_plain'][-1]:.0%} r0 {acc[k]['B_round0'][-1]:.0%} self{ROUNDS} {acc[k][f'B_self{ROUNDS}{tl(taus[0])}'][-1]:.0%}" for k in tests))
        L = ["# Lemmas as a scratchpad", "", f"{a.seeds} seeds; trained on {train_sizes} kinds (trees and chains), {a.nw} worlds each; tested on {ntest} fresh worlds per cell, 21 balanced target claims per world; accuracy (sd over seeds); taus {taus}, {ROUNDS} rounds. Predictions in the script.", "",
             "| test set | " + " | ".join(conds) + " |", "| --- | " + " | ".join("---:" for _ in conds) + " |"]
        for k in tests:
            L.append(f"| {k[0]}-{k[1]} | " + " | ".join(f"{100 * np.mean(acc[k][c]):.1f}% ({100 * np.std(acc[k][c]):.1f})" for c in conds) + " |")
        L += ["", "## Paired differences at the last round (points; sd) and lemma precision per round", "", "| test set | tau | self - round0 | self - A | oracle - round0 | round0 - A | lemma precision r1 / r2 / last |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for k in tests:
            d = lambda x, y: 100 * (np.array(acc[k][x]) - np.array(acc[k][y]))
            f = lambda v: f"{v.mean():+.1f} ({v.std():.1f})"
            for t in taus:
                last = f"B_self{ROUNDS}{tl(t)}"
                L.append(f"| {k[0]}-{k[1]} | {t} | {f(d(last, 'B_round0'))} | {f(d(last, 'A_plain'))} | {f(d('B_oracle', 'B_round0'))} | {f(d('B_round0', 'A_plain'))} | "
                         + " / ".join(f"{100 * np.nanmean(prec[k][(t, r)]):.0f}%" for r in (1, 2, ROUNDS)) + " |")
        run.finish({"acc": {f"{k[0]}-{k[1]}": v for k, v in acc.items()}}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

"""Q27 (out of the box, transfers the logic finding to ARC): FREE DATA by RULE-PRESERVING AUGMENTATION of the training puzzles. The pure STML learner overfits H-ARC (RL restarts 62-68% on
training puzzles, 52-56% on unseen; Q17-Q18 in logic: diversity of training episodes, not number of steps, is what helps).

An augmented copy of a puzzle applies ONE consistent transformation to every grid of the puzzle (train inputs and outputs, test input, correct output and every candidate answer): a random
rotation/reflection (8 dihedral) and a random permutation of the colours 1-9 (background 0 fixed). The rule of the puzzle is transformed with it; candidate indices, people's choices and feedback carry over unchanged.
Only the TRAINING puzzles of a split are augmented; evaluation is on the unaugmented test puzzles. Pure family (graph encoder + history transformer + output-match input), width 128, 15 epochs, same splits (seeds 0-2) as harc_h1_stml_v2.
Conditions (solved in 3 attempts on unseen puzzles):
  H_base      imitate human learning, H-ARC steps of the training puzzles
  H_base_x4   the same, 4x the epochs (equal number of gradient steps as H_aug)
  H_aug       the same steps replicated on 3 augmented copies of every training puzzle (4x the puzzles)
  A_base      imitate answers (same step budget as the human steps)
  A_aug       imitate answers on the original + 3 augmented copies (4x the steps)
PREDICTIONS, written before running:
  A1  H_aug beats H_base by >= 3 points solved (the learner is data-starved / overfits).
  A2  H_base_x4 gains <= 1 point over H_base (it is diversity, not steps).
  A3  H1 stays unsupported: H_aug minus A_aug is within +-2 points (augmentation lifts both).
  Falsifier of A1: H_aug - H_base within the seed sd.

    python experiments/harc_augmentation.py --quick
    python experiments/harc_augmentation.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.model.graphs import ARC_EDGE_TYPES, ARC_NODE_DIM, grid_graph, parse_grid  # noqa: E402
from stml.model.harc_data import (Puzzle, Step, answer_steps, attach_pools, build_graphs, human_steps,  # noqa: E402
                                  load_puzzles, read_records)
from stml.model.network import STMLLearner, pick_device  # noqa: E402
from stml.model.train import evaluate, fit_imitation  # noqa: E402
from stml.runlog import Run  # noqa: E402

K_AUG = 3


def transform_str(s: str, rot: int, flip: bool, perm: np.ndarray) -> str:
    g = parse_grid(s)
    g = np.rot90(g, rot)
    if flip:
        g = np.fliplr(g)
    g = perm[g]
    return "|" + "|".join("".join(str(int(c)) for c in row) for row in g) + "|"


def augmented(p: Puzzle, idx: int, rng: random.Random) -> Puzzle:
    perm = np.array([0] + rng.sample(range(1, 10), 9), dtype=np.int8)
    rot, flip = rng.randrange(4), rng.random() < 0.5
    tf = lambda s: transform_str(s, rot, flip, perm)
    q = Puzzle(f"{p.name}#a{idx}", [(tf(a), tf(b)) for a, b in p.train], tf(p.test_input), tf(p.target), [tf(c) for c in p.candidates])
    q.graph_train = [(grid_graph(a), grid_graph(b)) for a, b in q.train]
    q.graph_test_in = grid_graph(q.test_input)
    q.graph_cands = [grid_graph(c) for c in q.candidates]
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.d = 1, 2, 64
    device = pick_device(None)
    conds = ["H_base", "H_base_x4", "H_aug", "A_base", "A_aug"]
    with Run("harc_augmentation" + ("_quick" if a.quick else ""), {**vars(a), "conditions": conds, "k_aug": K_AUG}) as run:
        records = read_records(ROOT / "data/processed/harc.jsonl")
        puzzles = attach_pools(load_puzzles(ROOT / "data/raw/harc/arc_data"), records)
        if a.quick:
            puzzles = {k: puzzles[k] for k in sorted(puzzles)[:40]}
        build_graphs(puzzles, cache=None if a.quick else ROOT / "data/processed/harc_graphs.pkl", log=run.log)
        names = sorted(puzzles)
        cache: dict[tuple[str, int], Puzzle] = {}
        per_split = {c: [] for c in conds}

        def new_model(seed: int) -> STMLLearner:
            torch.manual_seed(seed)
            return STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=a.d, out_match=True).to(device)

        for seed in range(a.seeds):
            rnd = random.Random(seed)
            order = names[:]
            rnd.shuffle(order)
            train, test = order[: len(order) // 2], order[len(order) // 2:]
            for n in train:
                for i in range(K_AUG):
                    if (n, i) not in cache:
                        cache[(n, i)] = augmented(puzzles[n], i, random.Random(zlib.crc32(f"{n}:{i}".encode())))
            aug_names = [f"{n}#a{i}" for n in train for i in range(K_AUG)]
            allp = dict(puzzles)
            allp.update({q.name: q for (n, i), q in cache.items() if n in set(train)})
            h_train, h_test = human_steps(records, puzzles, set(train)), human_steps(records, puzzles, set(test))
            h_aug = h_train + [Step(f"{s.puzzle}#a{i}", s.tried, s.choice, s.correct) for s in h_train for i in range(K_AUG)]
            budget = len(h_train)
            run.log(f"--- split {seed + 1}/{a.seeds}: {len(train)} training, {len(test)} test, {budget} human steps, {len(h_aug)} with augmentation")

            def record(c: str, m: STMLLearner) -> None:
                r = evaluate(m, allp, test, device, human=h_test)
                per_split[c].append(r)
                run.log(f"{c}: solved {r['solved']:.1%}, first try {r['first']:.1%}")

            fit = lambda m, steps, ep=a.epochs: fit_imitation(m, allp, steps, device, epochs=ep, seed=seed, log=lambda s: None)
            record("H_base", fit(new_model(seed), h_train))
            record("H_base_x4", fit(new_model(seed), h_train, a.epochs * (K_AUG + 1)))
            record("H_aug", fit(new_model(seed), h_aug))
            record("A_base", fit(new_model(seed), answer_steps(puzzles, set(train), budget)))
            record("A_aug", fit(new_model(seed), answer_steps(allp, set(train) | set(aug_names), budget * (K_AUG + 1))))
            (run.dir / "partial_per_split.json").write_text(json.dumps({c: [{k: v for k, v in r.items() if not k.startswith('_')} for r in rs] for c, rs in per_split.items()}))
        sv = {c: np.array([r["solved"] for r in per_split[c]]) for c in conds}
        fs = {c: np.array([r["first"] for r in per_split[c]]) for c in conds}
        L = ["# Rule-preserving augmentation of the training puzzles (pure STML learner)", "", f"{a.seeds} splits; solved in 3 attempts on unseen puzzles (mean, sd over splits). Predictions A1-A3 in the script.", "",
             "| condition | solved in 3 | right first |", "| --- | ---: | ---: |"]
        for c in conds:
            L.append(f"| {c} | {100 * sv[c].mean():.1f}% ({100 * sv[c].std():.1f}) | {100 * fs[c].mean():.1f}% ({100 * fs[c].std():.1f}) |")
        L += ["", "| comparison | solved difference (points; sd) |", "| --- | ---: |"]
        for x, y in (("H_aug", "H_base"), ("H_base_x4", "H_base"), ("H_aug", "H_base_x4"), ("A_aug", "A_base"), ("H_aug", "A_aug"), ("H_base", "A_base")):
            d = 100 * (sv[x] - sv[y])
            L.append(f"| {x} minus {y} | {d.mean():+.1f} ({d.std():.1f}) |")
        run.finish({c: [{k: v for k, v in r.items() if not k.startswith('_')} for r in rs] for c, rs in per_split.items()}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

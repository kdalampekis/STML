"""Q28 (follow-up of Q27): does FREE DATA (rule-preserving augmentation) change the H1 comparison on ARC, now for the RL conditions?

Same puzzles, splits (seeds 0-2), augmentation (3 copies per training puzzle: one dihedral transform + one colour permutation applied to every grid and candidate) and pure learner as harc_augmentation.py.
Conditions, solved in 3 attempts on unseen puzzles (RL: 6000 episodes, 2 restarts, the restart chosen on the original TRAINING puzzles only; the RL step budget is the same with and without augmentation):
  R_base   RL from scratch on the original training puzzles
  R_aug    RL from scratch on the training puzzles and their augmented copies
  S_aug    STML pipeline: imitate human learning on the augmented steps (as H_aug in Q27), then 6000 RL episodes on training + augmented puzzles
PREDICTIONS, written before running:
  B1  R_aug beats R_base by >= 3 points (RL overfits too: 62-68% on training puzzles vs 52-56% on unseen).
  B2  S_aug is within +-2 points of R_aug (H1: human histories add nothing over RL once both see the same data).
  Falsifier of B1: R_aug - R_base within the seed sd.

    python experiments/harc_augmentation_rl.py --quick
    python experiments/harc_augmentation_rl.py
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from harc_augmentation import K_AUG, augmented  # noqa: E402
from stml.model.graphs import ARC_EDGE_TYPES, ARC_NODE_DIM  # noqa: E402
from stml.model.harc_data import Step, attach_pools, build_graphs, human_steps, load_puzzles, read_records  # noqa: E402
from stml.model.network import STMLLearner, pick_device  # noqa: E402
from stml.model.train import evaluate, fit_imitation, train_rl  # noqa: E402
from stml.runlog import Run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--rl-episodes", type=int, default=6000)
    ap.add_argument("--rl-restarts", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.rl_episodes, a.rl_restarts, a.d = 1, 2, 160, 1, 64
    device = pick_device(None)
    conds = ["R_base", "R_aug", "S_aug"]
    with Run("harc_augmentation_rl" + ("_quick" if a.quick else ""), {**vars(a), "conditions": conds, "k_aug": K_AUG}) as run:
        records = read_records(ROOT / "data/processed/harc.jsonl")
        puzzles = attach_pools(load_puzzles(ROOT / "data/raw/harc/arc_data"), records)
        if a.quick:
            puzzles = {k: puzzles[k] for k in sorted(puzzles)[:40]}
        build_graphs(puzzles, cache=None if a.quick else ROOT / "data/processed/harc_graphs.pkl", log=run.log)
        names = sorted(puzzles)
        cache: dict[tuple[str, int], object] = {}
        per_split = {c: [] for c in conds}

        def new_model(seed: int) -> STMLLearner:
            torch.manual_seed(seed)
            return STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=a.d, out_match=True).to(device)

        def best_rl(allp, pool, train, seed, tag):
            best = None
            for k in range(a.rl_restarts):
                m = new_model(1000 * seed + k)
                train_rl(m, allp, pool, device, episodes=a.rl_episodes, seed=1000 * seed + k, log=lambda s: None)
                score = evaluate(m, allp, train, device)["solved"]          # restart chosen on the ORIGINAL training puzzles
                run.log(f"{tag} restart {k + 1}: training-puzzle solve rate {score:.1%}")
                if best is None or score > best[0]:
                    best = (score, m)
            return best[1]

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
            h_train = human_steps(records, puzzles, set(train))
            h_aug = h_train + [Step(f"{s.puzzle}#a{i}", s.tried, s.choice, s.correct) for s in h_train for i in range(K_AUG)]
            h_test = human_steps(records, puzzles, set(test))
            run.log(f"--- split {seed + 1}/{a.seeds}: {len(train)} training, {len(test)} test puzzles")

            def record(c, m):
                r = evaluate(m, allp, test, device, human=h_test)
                per_split[c].append(r)
                run.log(f"{c}: solved {r['solved']:.1%}, first try {r['first']:.1%}")

            record("R_base", best_rl(allp, train, train, seed, "R_base"))
            record("R_aug", best_rl(allp, train + aug_names, train, seed, "R_aug"))
            m_s = fit_imitation(new_model(seed), allp, h_aug, device, epochs=a.epochs, seed=seed, log=lambda s: None)
            train_rl(m_s, allp, train + aug_names, device, episodes=a.rl_episodes, seed=seed, log=lambda s: None)
            record("S_aug", m_s)
            (run.dir / "partial_per_split.json").write_text(json.dumps({c: [{k: v for k, v in r.items() if not k.startswith('_')} for r in rs] for c, rs in per_split.items()}))
        sv = {c: np.array([r["solved"] for r in per_split[c]]) for c in conds}
        fs = {c: np.array([r["first"] for r in per_split[c]]) for c in conds}
        L = ["# Augmentation x RL (pure STML learner)", "", f"{a.seeds} splits; solved in 3 attempts on unseen puzzles (mean, sd over splits). Predictions B1-B2 in the script.", "",
             "| condition | solved in 3 | right first |", "| --- | ---: | ---: |"]
        for c in conds:
            L.append(f"| {c} | {100 * sv[c].mean():.1f}% ({100 * sv[c].std():.1f}) | {100 * fs[c].mean():.1f}% ({100 * fs[c].std():.1f}) |")
        L += ["", "| comparison | solved difference (points; sd) |", "| --- | ---: |"]
        for x, y in (("R_aug", "R_base"), ("S_aug", "R_aug"), ("S_aug", "R_base")):
            d = 100 * (sv[x] - sv[y])
            L.append(f"| {x} minus {y} | {d.mean():+.1f} ({d.std():.1f}) |")
        L += ["", "Reference (Q27, same splits): imitate human learning with augmentation 57.3%, imitate answers with augmentation 59.3%."]
        run.finish({c: [{k: v for k, v in r.items() if not k.startswith('_')} for r in rs] for c, rs in per_split.items()}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

"""Learning to learn across puzzles on H-ARC (goal 5).

People get better over their first few puzzles (first-attempt accuracy about 50% -> 58%, 1st to 5th, random
order). Does a learner that carries its experience (attempts + right/wrong feedback) from puzzle to puzzle
do the same? Test: the learner plays random sequences of 5 UNSEEN puzzles with its own context; accuracy is
reported by position in the sequence. A model with context switched off is the control (it should be flat).

Hybrid family (graph encoder + hand features, d=128), the strongest learner so far. Conditions:
  RL, no context               meta-RL sequences, but every puzzle treated alone (control)
  RL with context              meta-RL (RL^2): the transformer sees the earlier puzzles of the sequence
  Imitate human sequences      Stage A on people's own puzzle sequences (train puzzles only), with / without context
  STML: humans then RL         the Stage A models continued with RL, with / without context

    python experiments/harc_learning_to_learn.py --quick
    python experiments/harc_learning_to_learn.py --seeds 5
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.model.graphs import ARC_EDGE_TYPES, ARC_NODE_DIM  # noqa: E402
from stml.model.hand_features import N_HAND, attach_hand  # noqa: E402
from stml.model.harc_data import attach_pools, build_graphs, load_puzzles, read_records  # noqa: E402
from stml.model.network import STMLLearner, n_params, pick_device  # noqa: E402
from stml.model.train_seq import evaluate_seq, fit_imitation_seq, human_sequences, train_rl_seq  # noqa: E402
from stml.runlog import Run  # noqa: E402

CONDS = ["RL, no context", "RL with context", "Imitate human sequences, no context",
         "Imitate human sequences, with context", "STML: humans then RL, no context",
         "STML: humans then RL, with context"]


def human_curve(records: list[dict]) -> list[float]:
    by = {}
    for r in records:
        if r["task"]["order"] <= 5:      # positions 1-5 (most people did five puzzles; 182 did ten)
            by.setdefault(r["task"]["order"], []).append(bool(r["attempts"][0]["feedback"]["correct"]))
    return [float(np.mean(by[k])) for k in sorted(by)]


def slope(curve: list[float]) -> float:
    return float(np.polyfit(np.arange(len(curve)), curve, 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--rl-episodes", type=int, default=6000)
    ap.add_argument("--rl-restarts", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.rl_episodes, a.rl_restarts, a.d = 1, 2, 200, 1, 64
    device = pick_device(a.device)
    with Run("harc_learning_to_learn" + ("_quick" if a.quick else ""), {**vars(a), "device_used": str(device), "conditions": CONDS}) as run:
        records = read_records(ROOT / "data/processed/harc.jsonl")
        arc_dir = ROOT / "data/raw/harc/arc_data"
        puzzles = attach_pools(load_puzzles(arc_dir), records)
        if a.quick:
            puzzles = {k: puzzles[k] for k in sorted(puzzles)[:80]}
        build_graphs(puzzles, cache=None if a.quick else ROOT / "data/processed/harc_graphs.pkl", log=run.log)
        attach_hand(puzzles, arc_dir, records)
        raw_hand = {n: p.hand for n, p in puzzles.items()}
        names = sorted(puzzles)
        hc = human_curve(records)
        run.log(f"people, first-attempt accuracy by position in their session: {[round(x, 3) for x in hc]}")
        per_split = {c: [] for c in CONDS}

        def new_model(seed: int) -> STMLLearner:
            torch.manual_seed(seed)
            return STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=a.d, out_match=True, hand_dim=N_HAND, wide=True).to(device)

        run.log(f"model size: {n_params(new_model(0)):,} parameters")
        for seed in range(a.seeds):
            rnd = random.Random(seed)
            order = names[:]
            rnd.shuffle(order)
            train, test = order[: len(order) // 2], order[len(order) // 2:]
            X = np.concatenate([raw_hand[n] for n in train])
            mu, sd = X.mean(0), X.std(0)
            sd = np.where(sd < 1e-6, 1.0, sd)
            for n in names:
                puzzles[n].hand = ((raw_hand[n] - mu) / sd).astype(np.float32)
            seqs = human_sequences(records, puzzles, set(train))
            run.log(f"--- split {seed + 1}/{a.seeds}: {len(train)} train, {len(test)} test puzzles; "
                    f"{len(seqs)} human sequences ({sum(map(len, seqs))} puzzle episodes, mean length {np.mean(list(map(len, seqs))):.1f})")
            metric = lambda d, s=seed: run.metric({"split": s, **d})

            def record(cond: str, model: STMLLearner, ctx: bool) -> None:
                r = evaluate_seq(model, puzzles, test, device, use_context=ctx, seed=seed)
                per_split[cond].append(r)
                run.log(f"{cond}: solved {r['solved']:.1%}, first {r['first']:.1%}; first-try by position "
                        f"{[round(x, 3) for x in r['first_by_position']]}")
                metric({"condition": cond, **r})

            best = {}
            for ctx, label in ((False, "RL, no context"), (True, "RL with context")):
                run.log(f"[{label}] best of {a.rl_restarts} restarts, chosen on TRAINING puzzles")
                top = None
                for k in range(a.rl_restarts):
                    m = new_model(1000 * seed + k)
                    train_rl_seq(m, puzzles, train, device, use_context=ctx, episodes=a.rl_episodes, seed=1000 * seed + k,
                                 log=run.log, metric=metric)
                    score = evaluate_seq(m, puzzles, train[: len(test)], device, use_context=ctx, repeats=2, seed=seed)["solved"]
                    run.log(f"restart {k + 1}: training-puzzle solve rate {score:.1%}")
                    if top is None or score > top[0]:
                        top = (score, m)
                record(label, top[1], ctx)
            for ctx, tag in ((False, "no context"), (True, "with context")):
                run.log(f"[imitate human sequences, {tag}]")
                m = fit_imitation_seq(new_model(seed), puzzles, seqs, device, use_context=ctx, epochs=a.epochs, seed=seed,
                                      log=run.log, metric=metric)
                record(f"Imitate human sequences, {tag}", m, ctx)
                run.log(f"[STML: humans then RL, {tag}]")
                m2 = copy.deepcopy(m)
                train_rl_seq(m2, puzzles, train, device, use_context=ctx, episodes=a.rl_episodes, seed=seed, log=run.log, metric=metric)
                record(f"STML: humans then RL, {tag}", m2, ctx)
            (run.dir / "partial_per_split.json").write_text(json.dumps(per_split))

        # report
        L = 5
        lines = ["# Learning to learn across puzzles (H-ARC, hybrid family)", "",
                 f"{a.seeds} splits; random sequences of {L} unseen puzzles, played with the learner's own context. "
                 "Accuracy by position in the sequence (right on the first try).", "",
                 f"People, first-attempt accuracy by position in their session (all puzzles): {', '.join(f'{x:.1%}' for x in hc)}", "",
                 "| Condition | Solved in 3 | First try | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | Slope per position (pts) |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        results = {}
        for c in CONDS:
            rs = per_split[c]
            fb = np.mean([r["first_by_position"] for r in rs], 0)
            sl = [slope(r["first_by_position"]) * 100 for r in rs]
            results[c] = {"solved": float(np.mean([r["solved"] for r in rs])), "first": float(np.mean([r["first"] for r in rs])),
                          "first_by_position": fb.tolist(), "slope_pts_per_split": sl}
            lines.append(f"| {c} | {results[c]['solved']:.1%} | {results[c]['first']:.1%} | " +
                         " | ".join(f"{x:.1%}" for x in fb) + f" | {np.mean(sl):+.2f} (sd {np.std(sl):.2f}) |")
        lines += ["", "Slope = least-squares change in first-try accuracy per position, in points, per split (mean and sd over splits). "
                      "The no-context rows are the control: any slope there is noise or an artifact of the sequences.", ""]
        run.finish({"per_split": per_split, "summary": results, "people_curve": hc}, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

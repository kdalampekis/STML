"""H1 on H-ARC, round 2: the STML learner with ARCTraj, in two model families.

  --family pure     graph encoder + history transformer + output-match input. No hand-made features.
  --family hybrid   the same, plus the 12 hand-made features of baseline_v1 as an input and as a linear
                    shortcut to the choice score and the confidence (wide-and-deep).

Same protocol, splits (seeds 0-4) and test puzzles as harc_h1_stml.py and baseline_v1, so all three
are comparable. Conditions:
  Imitate human learning                  H-ARC histories only
  Imitate human + ARCTraj                 + ARCTraj first-attempt choices and the progress regression
  Imitate answers                         same step budget as the H-ARC human steps
  RL from scratch (best of restarts)      restarts chosen on TRAINING puzzles
  Imitate RL learning history             same step budget
  STML: imitate humans, then RL           H-ARC only
  STML: imitate humans + ARCTraj, then RL
ARCTraj is only ever used on the split's training puzzles. Run the two families as separate processes:
    python experiments/harc_h1_stml_v2.py --family pure
    python experiments/harc_h1_stml_v2.py --family hybrid
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.model.arctraj_data import arctraj_steps, attach_progress, load_progress_grids  # noqa: E402
from stml.model.graphs import ARC_EDGE_TYPES, ARC_NODE_DIM  # noqa: E402
from stml.model.hand_features import N_HAND, attach_hand  # noqa: E402
from stml.model.harc_data import (answer_steps, attach_pools, build_graphs, human_steps,  # noqa: E402
                                  load_puzzles, read_records)
from stml.model.network import STMLLearner, n_params, pick_device  # noqa: E402
from stml.model.train import evaluate, fit_imitation, people_reference, random_reference, train_rl  # noqa: E402
from stml.runlog import Run, table  # noqa: E402

CONDITIONS = ["Imitate human learning", "Imitate human + ARCTraj", "Imitate answers", "Imitate RL learning history",
              "RL from scratch (best of restarts)", "STML: imitate humans, then RL",
              "STML: imitate humans + ARCTraj, then RL"]
AUX_WEIGHT = 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["pure", "hybrid"], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--rl-episodes", type=int, default=6000)
    ap.add_argument("--rl-restarts", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.rl_episodes, a.rl_restarts, a.d = 1, 2, 160, 1, 64
    hybrid = a.family == "hybrid"
    device = pick_device(a.device)
    config = {**vars(a), "device_used": str(device), "conditions": CONDITIONS, "aux_weight": AUX_WEIGHT,
              "protocol": "choose among people's answers; 3 attempts; right/wrong feedback; 50/50 puzzle split per seed"}
    with Run(f"harc_h1_stml_v2_{a.family}" + ("_quick" if a.quick else ""), config) as run:
        run.log(f"family: {a.family}; device: {device}")
        records = read_records(ROOT / "data/processed/harc.jsonl")
        arc_records = read_records(ROOT / "data/processed/arctraj.jsonl")
        arc_dir = ROOT / "data/raw/harc/arc_data"
        puzzles = attach_pools(load_puzzles(arc_dir), records)
        if a.quick:
            puzzles = {k: puzzles[k] for k in sorted(puzzles)[:60]}
        run.log(f"{len(puzzles)} puzzles whose correct answer is among people's answers")
        build_graphs(puzzles, cache=None if a.quick else ROOT / "data/processed/harc_graphs.pkl", log=run.log)
        grids = load_progress_grids(ROOT / "data/raw/arctraj/ARCTraj.csv", puzzles,
                                    cache=ROOT / "data/processed/arctraj_progress_grids.pkl")
        attach_progress(puzzles, grids, cache=ROOT / "data/processed/arctraj_progress_graphs.pkl", log=run.log)
        if hybrid:
            attach_hand(puzzles, arc_dir, records)
        raw_hand = {n: p.hand for n, p in puzzles.items()} if hybrid else None
        names = sorted(puzzles)
        per_split: dict[str, list[dict]] = {c: [] for c in CONDITIONS + ["Random choice", "People (reference)"]}

        def new_model(seed: int) -> STMLLearner:
            torch.manual_seed(seed)
            return STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=a.d, out_match=True,
                               hand_dim=N_HAND if hybrid else 0, wide=hybrid).to(device)

        run.log(f"model size: {n_params(new_model(0)):,} parameters")
        for seed in range(a.seeds):
            rnd = random.Random(seed)
            order = names[:]
            rnd.shuffle(order)
            train, test = order[: len(order) // 2], order[len(order) // 2:]
            if hybrid:   # standardize with TRAINING puzzles' candidates only; features with no spread keep unit scale
                X = np.concatenate([raw_hand[n] for n in train])
                mu, sd = X.mean(0), X.std(0)
                sd = np.where(sd < 1e-6, 1.0, sd)
                for n in names:
                    puzzles[n].hand = ((raw_hand[n] - mu) / sd).astype(np.float32)
            h_train, h_test = human_steps(records, puzzles, set(train)), human_steps(records, puzzles, set(test))
            extra = arctraj_steps(arc_records, puzzles, set(train))
            budget = len(h_train)
            run.log(f"--- split {seed + 1}/{a.seeds}: {len(train)} training, {len(test)} test puzzles, "
                    f"{budget} human steps, {len(extra)} ARCTraj steps")
            metric = lambda d, s=seed: run.metric({"split": s, **d})

            def record(cond: str, model: STMLLearner) -> None:
                r = evaluate(model, puzzles, test, device, human=h_test)
                per_split[cond].append(r)
                run.log(f"{cond}: solved {r['solved']:.1%}, first try {r['first']:.1%}, "
                        f"human-likeness {r['human_loglik']:.2f}")
                metric({"condition": cond, **{k: v for k, v in r.items() if not k.startswith('_')}})

            fit = lambda m, steps, **kw: fit_imitation(m, puzzles, steps, device, epochs=a.epochs, seed=seed,
                                                       log=run.log, metric=metric, **kw)
            run.log("[1/7] imitate human learning")
            m_h = fit(new_model(seed), h_train)
            record("Imitate human learning", m_h)

            run.log("[2/7] imitate human learning + ARCTraj")
            m_ha = fit(new_model(seed), h_train + extra, aux_weight=AUX_WEIGHT)
            record("Imitate human + ARCTraj", m_ha)

            run.log("[3/7] imitate answers")
            record("Imitate answers", fit(new_model(seed), answer_steps(puzzles, set(train), budget)))

            run.log("[4/7] RL from scratch, best of restarts chosen on TRAINING puzzles")
            best = None
            for k in range(a.rl_restarts):
                m = new_model(1000 * seed + k)
                hist = train_rl(m, puzzles, train, device, episodes=a.rl_episodes, seed=1000 * seed + k,
                                log=run.log, metric=metric)
                score = evaluate(m, puzzles, train, device)["solved"]
                run.log(f"restart {k + 1}: training-puzzle solve rate {score:.1%}")
                if best is None or score > best[0]:
                    best = (score, m, hist)
            record("RL from scratch (best of restarts)", best[1])

            run.log("[5/7] imitate the RL agent's learning history (same step budget)")
            take = np.random.default_rng(seed).choice(len(best[2]), size=min(budget, len(best[2])), replace=False)
            record("Imitate RL learning history", fit(new_model(seed), [best[2][i] for i in take]))

            run.log("[6/7] STML: imitate humans, then RL")
            m_s = copy.deepcopy(m_h)
            train_rl(m_s, puzzles, train, device, episodes=a.rl_episodes, seed=seed, log=run.log, metric=metric)
            record("STML: imitate humans, then RL", m_s)

            run.log("[7/7] STML: imitate humans + ARCTraj, then RL")
            m_sa = copy.deepcopy(m_ha)
            train_rl(m_sa, puzzles, train, device, episodes=a.rl_episodes, seed=seed, log=run.log, metric=metric)
            record("STML: imitate humans + ARCTraj, then RL", m_sa)

            per_split["Random choice"].append(random_reference(puzzles, test, h_test))
            per_split["People (reference)"].append(people_reference(records, set(test)))
            import json   # progress after every split, so a killed run keeps what it finished
            (run.dir / "partial_per_split.json").write_text(json.dumps(
                {c: [{k: v for k, v in r.items()} for r in rs] for c, rs in per_split.items()}))

        results, report = summarize(per_split, a, device)
        run.finish(results, report)


def summarize(per_split, a, device):
    keys = ["solved", "first", "recover", "human_loglik", "human_loglik_revisions", "brier", "ece"]
    mean = {c: {k: float(np.nanmean([r[k] for r in rs])) for k in keys} for c, rs in per_split.items() if rs}
    sd = {c: {k: float(np.nanstd([r[k] for r in rs])) for k in keys} for c, rs in per_split.items() if rs}
    rng = np.random.default_rng(0)
    paired = {}
    for base_name in ("Imitate human learning", "Imitate human + ARCTraj"):
        base = per_split[base_name]
        for other in CONDITIONS:
            if other == base_name:
                continue
            diffs = [x["solved"] - y["solved"] for x, y in zip(base, per_split[other])]
            boots = []
            for _ in range(2000):
                tot = []
                for x, y in zip(base, per_split[other]):
                    sx, sy = np.array(x["_solved"]), np.array(y["_solved"])
                    idx = rng.integers(len(sx), size=len(sx))
                    tot.append(sx[idx].mean() - sy[idx].mean())
                boots.append(np.mean(tot))
            paired[f"{base_name} minus {other}"] = {"diff": float(np.mean(diffs)),
                                                    "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    cols = [("solved", "Solved in 3", "{:.1%}"), ("first", "Right first", "{:.1%}"), ("recover", "Recovers", "{:.1%}"),
            ("human_loglik", "Human-likeness", "{:.2f}"), ("human_loglik_revisions", "(revisions)", "{:.2f}"),
            ("brier", "Brier", "{:.3f}"), ("ece", "ECE", "{:.3f}")]
    lines = [f"# H1 on H-ARC, STML learner round 2 ({a.family} family, with ARCTraj)", "",
             f"Splits: {a.seeds} · Stage A epochs: {a.epochs} · RL episodes: {a.rl_episodes} · "
             f"RL restarts: {a.rl_restarts} · width d={a.d} · device: {device}", "",
             "Same unseen-puzzle protocol and splits as baseline_v1 and harc_h1_stml. "
             "ARCTraj is used on training puzzles only.", "", "## Results (mean over splits)", "", table(mean, cols), "",
             "## Paired differences in solved-in-3 (95% CI over puzzles)", ""]
    lines += [f"- {k}: {v['diff']:+.1%} [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]" for k, v in paired.items()]
    lines += ["", "## Solved in 3, per split", ""]
    lines += [f"- {c}: " + ", ".join(f"{r['solved']:.1%}" for r in rs) for c, rs in per_split.items() if rs]
    lines += ["", "References: baseline_v1 (12 hand features, linear): 65.8-67.1% solved, 34% right first; "
                  "harc_h1_stml (pure GNN, round 1): 51-54% solved, 22-25% right first."]
    results = {"mean": mean, "sd": sd, "paired": paired,
               "per_split": {c: [{k: v for k, v in r.items() if not k.startswith('_')} for r in rs]
                             for c, rs in per_split.items()}}
    return results, "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

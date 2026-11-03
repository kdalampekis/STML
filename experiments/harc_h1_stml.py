"""H1 on H-ARC with the STML learner (graph encoder + history transformer).

Question: on unseen ARC puzzles (3 attempts, right/wrong feedback, choosing among
the answers people gave), is the STML learner better when it is pretrained by
imitating HUMAN learning histories than by imitating answers or an RL agent's
learning history? And does the full STML pipeline (imitate humans, then RL) win?

Run on the Mac (uses Metal automatically):
    python experiments/harc_h1_stml.py --quick          # 2-minute smoke test first
    python experiments/harc_h1_stml.py --seeds 5        # the real run

Needs data/processed/harc.jsonl (python scripts/convert_datasets.py harc ...)
and data/raw/harc/arc_data/. Every run writes runs/<time>_harc_h1_stml/.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.model.graphs import ARC_EDGE_TYPES, ARC_NODE_DIM  # noqa: E402
from stml.model.harc_data import (answer_steps, attach_pools, build_graphs, human_steps,  # noqa: E402
                                  load_puzzles, read_records)
from stml.model.network import STMLLearner, n_params, pick_device  # noqa: E402
from stml.model.train import (evaluate, fit_imitation, people_reference, random_reference,  # noqa: E402
                              train_rl)
from stml.runlog import Run, table  # noqa: E402

CONDITIONS = ["Imitate human learning", "Imitate answers", "Imitate RL learning history",
              "RL from scratch (best of restarts)", "STML: imitate humans, then RL"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "harc.jsonl"))
    ap.add_argument("--arc", default=str(ROOT / "data" / "raw" / "harc" / "arc_data"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--rl-episodes", type=int, default=6000)
    ap.add_argument("--rl-restarts", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--device", default=None, help="mps, cuda or cpu (default: best available)")
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.rl_episodes, a.rl_restarts, a.d = 1, 2, 160, 1, 64

    device = pick_device(a.device)
    config = {**vars(a), "device_used": str(device), "conditions": CONDITIONS,
              "protocol": "choose among people's answers; 3 attempts; right/wrong feedback; 50/50 puzzle split per seed"}
    with Run("harc_h1_stml" + ("_quick" if a.quick else ""), config) as run:
        run.log(f"device: {device}")
        records = read_records(a.data)
        puzzles = attach_pools(load_puzzles(a.arc), records)
        if a.quick:
            keep = sorted(puzzles)[:60]
            puzzles = {k: puzzles[k] for k in keep}
        run.log(f"{len(puzzles)} puzzles whose correct answer is among people's answers")
        build_graphs(puzzles, cache=None if a.quick else ROOT / "data" / "processed" / "harc_graphs.pkl", log=run.log)
        names = sorted(puzzles)
        per_split: dict[str, list[dict]] = {c: [] for c in CONDITIONS + ["Random choice", "People (reference)"]}

        def new_model(seed: int) -> STMLLearner:
            import torch
            torch.manual_seed(seed)
            return STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=a.d).to(device)

        run.log(f"model size: {n_params(new_model(0)):,} parameters")
        for seed in range(a.seeds):
            rnd = random.Random(seed)
            order = names[:]
            rnd.shuffle(order)
            train, test = order[: len(order) // 2], order[len(order) // 2:]
            h_train = human_steps(records, puzzles, set(train))
            h_test = human_steps(records, puzzles, set(test))
            budget = len(h_train)
            run.log(f"--- split {seed + 1}/{a.seeds}: {len(train)} training puzzles, {len(test)} test puzzles, "
                    f"{budget} human steps")
            metric = lambda d, s=seed: run.metric({"split": s, **d})

            def record(cond: str, model: STMLLearner) -> None:
                r = evaluate(model, puzzles, test, device, human=h_test)
                per_split[cond].append(r)
                run.log(f"{cond}: solved {r['solved']:.1%}, first try {r['first']:.1%}, "
                        f"human-likeness {r['human_loglik']:.2f}")
                metric({"condition": cond, **{k: v for k, v in r.items() if not k.startswith('_')}})

            run.log("[1/5] imitate human learning")
            m_h = fit_imitation(new_model(seed), puzzles, h_train, device, epochs=a.epochs, seed=seed,
                                log=run.log, metric=metric)
            record("Imitate human learning", m_h)

            run.log("[2/5] imitate answers")
            m_a = fit_imitation(new_model(seed), puzzles, answer_steps(puzzles, set(train), budget), device,
                                epochs=a.epochs, seed=seed, log=run.log, metric=metric)
            record("Imitate answers", m_a)

            run.log("[3/5] RL from scratch, best of restarts chosen on TRAINING puzzles")
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

            run.log("[4/5] imitate the RL agent's learning history (same step budget)")
            rng = np.random.default_rng(seed)
            take = rng.choice(len(best[2]), size=min(budget, len(best[2])), replace=False)
            m_r = fit_imitation(new_model(seed), puzzles, [best[2][i] for i in take], device, epochs=a.epochs,
                                seed=seed, log=run.log, metric=metric)
            record("Imitate RL learning history", m_r)

            run.log("[5/5] STML pipeline: imitate humans, then RL")
            m_s = copy.deepcopy(m_h)
            train_rl(m_s, puzzles, train, device, episodes=a.rl_episodes, seed=seed, log=run.log, metric=metric)
            record("STML: imitate humans, then RL", m_s)

            per_split["Random choice"].append(random_reference(puzzles, test, h_test))
            per_split["People (reference)"].append(people_reference(records, set(test)))

        results, report = summarize(per_split, a, device)
        run.finish(results, report)


def summarize(per_split: dict[str, list[dict]], a, device) -> tuple[dict, str]:
    keys = ["solved", "first", "recover", "human_loglik", "human_loglik_revisions", "brier", "ece"]
    mean = {c: {k: float(np.nanmean([r[k] for r in rs])) for k in keys} for c, rs in per_split.items() if rs}
    sd = {c: {k: float(np.nanstd([r[k] for r in rs])) for k in keys} for c, rs in per_split.items() if rs}
    paired = {}
    base = per_split["Imitate human learning"]
    rng = np.random.default_rng(0)
    for other in CONDITIONS[1:]:
        diffs, boots = [], []
        for x, y in zip(base, per_split[other]):
            diffs.append(x["solved"] - y["solved"])
        for _ in range(2000):
            tot = []
            for x, y in zip(base, per_split[other]):
                sx, sy = np.array(x["_solved"]), np.array(y["_solved"])
                idx = rng.integers(len(sx), size=len(sx))
                tot.append(sx[idx].mean() - sy[idx].mean())
            boots.append(np.mean(tot))
        paired[other] = {"diff": float(np.mean(diffs)), "ci95": [float(np.percentile(boots, 2.5)),
                                                                float(np.percentile(boots, 97.5))]}
    cols = [("solved", "Solved in 3", "{:.1%}"), ("first", "Right first", "{:.1%}"), ("recover", "Recovers", "{:.1%}"),
            ("human_loglik", "Human-likeness", "{:.2f}"), ("human_loglik_revisions", "(revisions)", "{:.2f}"),
            ("brier", "Brier", "{:.3f}"), ("ece", "ECE", "{:.3f}")]
    lines = [
        "# H1 on H-ARC with the STML learner",
        "",
        f"Splits: {a.seeds} · Stage A epochs: {a.epochs} · RL episodes: {a.rl_episodes} · "
        f"RL restarts: {a.rl_restarts} · width d={a.d} · device: {device}",
        "",
        "Unseen puzzles; 3 attempts with right/wrong feedback; the learner chooses among the answers people gave. "
        "Human-likeness = mean log-likelihood of people's actual choices (higher is more human-like). "
        "Brier and ECE measure how well the confidence head's P(right) matches reality (lower is better).",
        "",
        "## Results (mean over splits)",
        "",
        table(mean, cols),
        "",
        "## Imitate human learning minus each alternative (solved in 3, 95% CI over puzzles)",
        "",
    ]
    for k, v in paired.items():
        lines.append(f"- {k}: {v['diff']:+.1%} [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]")
    lines += ["", "## Solved in 3, per split", ""]
    for c, rs in per_split.items():
        if rs:
            lines.append(f"- {c}: " + ", ".join(f"{r['solved']:.1%}" for r in rs))
    lines += ["", "Compare with experiments/baseline_v1 (hand-made features, all conditions 65.8-67.1%, "
                  "people 71.4%, random 45.8%)."]
    results = {"mean": mean, "sd": sd, "paired_vs_human_learning": paired,
               "per_split": {c: [{k: v for k, v in r.items() if not k.startswith('_')} for r in rs]
                             for c, rs in per_split.items()}}
    return results, "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

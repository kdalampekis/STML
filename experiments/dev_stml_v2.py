"""Development ablation for the STML learner: does ARCTraj (and a smaller model) help?

Model selection must not touch the final test puzzles. Each dev split takes a random
half of the puzzles as "training"; that half is cut again into fit (80%) and validation
(20%). Variants are trained on fit only (Stage A, imitate human learning) and scored on
validation. The dev splits use seeds >= 100, and the final run (harc_h1_stml_v2.py) scores
on separate seeds 0-4, whose test halves are never used here.

    python experiments/dev_stml_v2.py --splits 2
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.model.arctraj_data import arctraj_steps, attach_progress, load_progress_grids  # noqa: E402
from stml.model.hand_features import N_HAND, attach_hand  # noqa: E402
from stml.model.graphs import ARC_EDGE_TYPES, ARC_NODE_DIM  # noqa: E402
from stml.model.harc_data import attach_pools, build_graphs, human_steps, load_puzzles, read_records  # noqa: E402
from stml.model.network import STMLLearner, pick_device  # noqa: E402
from stml.model.train import evaluate, fit_imitation  # noqa: E402
from stml.runlog import Run, table  # noqa: E402

# name: (d, dropout, use ARCTraj steps, aux weight, output-match input, hand features)
VARIANTS = {
    "A base (d128)": (128, 0.1, False, 0.0, False, False),
    "B + ARCTraj steps": (128, 0.1, True, 0.0, False, False),
    "C + ARCTraj progress": (128, 0.1, False, 1.0, False, False),
    "D + steps + progress": (128, 0.1, True, 1.0, False, False),
    "E small (d64, dropout .3)": (64, 0.3, False, 0.0, False, False),
    "F small + steps + progress": (64, 0.3, True, 1.0, False, False),
    "G output-match input": (128, 0.1, False, 0.0, True, False),
    "H hand features (hybrid)": (128, 0.1, False, 0.0, False, True),
    "I output-match + hand": (128, 0.1, False, 0.0, True, True),
    "J I + steps + progress": (128, 0.1, True, 1.0, True, True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default="", help="comma-separated variant letters, e.g. A,G,H")
    a = ap.parse_args()
    device = pick_device(a.device)
    if a.only:
        keep = set(a.only.split(","))
        VARIANTS_RUN = {k: v for k, v in VARIANTS.items() if k.split()[0] in keep}
    else:
        VARIANTS_RUN = dict(VARIANTS)
    with Run("dev_stml_v2", {**vars(a), "variants": {k: list(v) for k, v in VARIANTS_RUN.items()}}) as run:
        records = read_records(ROOT / "data/processed/harc.jsonl")
        arc_records = read_records(ROOT / "data/processed/arctraj.jsonl")
        puzzles = attach_pools(load_puzzles(ROOT / "data/raw/harc/arc_data"), records)
        build_graphs(puzzles, cache=ROOT / "data/processed/harc_graphs.pkl", log=run.log)
        grids = load_progress_grids(ROOT / "data/raw/arctraj/ARCTraj.csv", puzzles,
                                    cache=ROOT / "data/processed/arctraj_progress_grids.pkl")
        run.log(f"progress grids: {sum(map(len, grids.values()))} over {len(grids)} puzzles")
        attach_progress(puzzles, grids, cache=ROOT / "data/processed/arctraj_progress_graphs.pkl", log=run.log)
        attach_hand(puzzles, ROOT / "data/raw/harc/arc_data", records)
        names = sorted(puzzles)
        res = {k: [] for k in VARIANTS_RUN}
        for s in range(a.splits):
            rnd = random.Random(100 + s)
            order = names[:]
            rnd.shuffle(order)
            train = order[: len(order) // 2]
            fit, val = train[: int(0.8 * len(train))], train[int(0.8 * len(train)):]
            h_fit, h_val = human_steps(records, puzzles, set(fit)), human_steps(records, puzzles, set(val))
            extra = arctraj_steps(arc_records, puzzles, set(fit))
            run.log(f"--- dev split {s + 1}/{a.splits}: fit {len(fit)}, val {len(val)}, human steps {len(h_fit)}, "
                    f"ARCTraj steps {len(extra)}")
            for k, (d, dr, use_steps, aux_w, om, hand) in VARIANTS_RUN.items():
                import torch
                torch.manual_seed(s)
                m = STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=d, dropout=dr, out_match=om,
                                hand_dim=N_HAND if hand else 0).to(device)
                fit_imitation(m, puzzles, h_fit + (extra if use_steps else []), device, epochs=a.epochs, seed=s,
                              log=lambda _: None, aux_weight=aux_w)
                r = evaluate(m, puzzles, val, device, human=h_val)
                res[k].append(r)
                run.log(f"{k}: val solved {r['solved']:.1%}, first {r['first']:.1%}, human-likeness "
                        f"{r['human_loglik']:.2f}, Brier {r['brier']:.3f}")
                run.metric({"split": s, "variant": k, **{x: y for x, y in r.items() if not x.startswith('_')}})
        keys = ["solved", "first", "recover", "human_loglik", "human_loglik_revisions", "brier", "ece"]
        mean = {k: {x: float(np.nanmean([r[x] for r in rs])) for x in keys} for k, rs in res.items()}
        cols = [("solved", "Val solved in 3", "{:.1%}"), ("first", "Right first", "{:.1%}"),
                ("human_loglik", "Human-likeness", "{:.2f}"), ("brier", "Brier", "{:.3f}")]
        rep = ("# Dev ablation (validation puzzles only, Stage A imitate human learning)\n\n" + table(mean, cols) +
               f"\n\nMean over {a.splits} dev splits; each validation set is ~79 puzzles, so differences "
               "under about 5 points are within noise.\n")
        run.finish({"mean": mean, "per_split": {k: [{x: y for x, y in r.items() if not x.startswith('_')} for r in rs]
                                                for k, rs in res.items()}}, rep)


if __name__ == "__main__":
    main()

"""Baseline v1: the first H1 test, with a small linear choice model and hand-made features.

Kept as the reference point for the STML learner (experiments/harc_h1_stml.py).
It is the version with the feature-scaling bug fixed (see CLAUDE.md, "First
experiment"). Run on the Mac:
    python experiments/baseline_v1/run.py --seeds 5     # about 8 minutes
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from choice_model import attach_candidates, load_puzzles, run  # noqa: E402
from stml.runlog import Run, table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "harc.jsonl"))
ap.add_argument("--arc", default=str(ROOT / "data" / "raw" / "harc" / "arc_data"))
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()

with Run("baseline_v1_choice_model", vars(a)) as r:
    records = [json.loads(l) for l in open(a.data)]
    puzzles = attach_candidates(load_puzzles(a.arc), records)
    r.log(f"{len(puzzles)} puzzles")
    res = run(records, puzzles, seeds=a.seeds, out=r.dir / "raw_results.json")
    rows = {k: {m: v[0] for m, v in d.items()} for k, d in res["summary"].items()}
    cols = [("solved", "Solved in 3", "{:.1%}"), ("first", "Right first", "{:.1%}"), ("recover", "Recovers", "{:.1%}"),
            ("fit_all", "Human-likeness", "{:.2f}"), ("fit_revision", "(revisions)", "{:.2f}")]
    report = "# Baseline v1: linear choice model, hand-made features\n\n" + table(rows, cols) + "\n\n"
    report += "\n".join(f"- Imitate human learning minus {k}: {v['diff']:+.1%} [{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]"
                        for k, v in res["paired_vs_human_learning"].items()) + "\n"
    r.finish(res, report)

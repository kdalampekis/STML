"""Convert human-learning datasets to stml-traj/1 and print their learning profile.

Usage:
  python scripts/convert_datasets.py harc --data path/to/h-arc/data --arc path/to/h-arc/arc_data
  python scripts/convert_datasets.py arctraj --arc data/raw/harc/arc_data
  python scripts/convert_datasets.py game logs/pilot-01
  python scripts/convert_datasets.py profile datasets/harc.jsonl datasets/game.jsonl

Output goes to datasets/<source>.jsonl. The profile is domain-free: how often
people are right on attempt 1, 2, 3; how much they change after being wrong;
how often they explain themselves. That is the shape of human learning the
model will imitate, and it lets us compare domains side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.datasets.common import learning_events, read_jsonl, write_jsonl  # noqa: E402

OUT = ROOT / "data" / "processed"


def profile(records: list[dict]) -> dict:
    by_attempt: dict[int, list[bool]] = defaultdict(list)
    change_after_wrong, think_after_wrong = [], []
    explained = attempts = 0
    for rec in records:
        ev = learning_events(rec)
        for i, e in enumerate(ev):
            attempts += 1
            explained += e["explained"]
            if e["correct"] is not None:
                by_attempt[e["attempt"]].append(bool(e["correct"]))
            if i > 0 and ev[i - 1]["correct"] is False:
                if e["change"] is not None:
                    change_after_wrong.append(e["change"])
                think_after_wrong.append(e["think_ms"])
    return {
        "records": len(records),
        "learners": len({r["learner"]["id"] for r in records}),
        "tasks": len({r["task"]["id"] for r in records}),
        "attempts": attempts,
        "solved_share": round(mean(r["outcome"]["solved"] for r in records), 3) if records else None,
        "right_by_attempt": {k + 1: round(mean(v), 3) for k, v in sorted(by_attempt.items()) if len(v) >= 1},
        "n_by_attempt": {k + 1: len(v) for k, v in sorted(by_attempt.items())},
        "median_change_after_wrong": median(change_after_wrong) if change_after_wrong else None,
        "median_seconds_after_wrong": round(median(think_after_wrong) / 1000, 1) if think_after_wrong else None,
        "explained_share": round(explained / attempts, 3) if attempts else None,
    }


def print_profile(name: str, p: dict) -> None:
    print(f"\n{name}: {p['records']} learner-task records, {p['learners']} learners, {p['tasks']} tasks, "
          f"{p['attempts']} attempts")
    print(f"  solved: {p['solved_share']:.0%}" if p["solved_share"] is not None else "  solved: -")
    for k, v in p["right_by_attempt"].items():
        print(f"  right on attempt {k}: {v:.0%}  (n={p['n_by_attempt'][k]})")
    print(f"  after a wrong answer: median change {p['median_change_after_wrong']}, "
          f"median {p['median_seconds_after_wrong']} s before the next answer")
    if p["explained_share"] is not None:
        print(f"  attempts with the learner's own explanation: {p['explained_share']:.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harc")
    h.add_argument("--data", required=True, help="folder with data.csv and summary_data.csv")
    h.add_argument("--arc", required=True, help="h-arc/arc_data folder")
    h.add_argument("--all-participants", action="store_true", help="keep participants who did not finish")
    a = sub.add_parser("arctraj")
    a.add_argument("--csv", default=str(ROOT / "data/raw/arctraj/ARCTraj.csv"))
    a.add_argument("--arc", required=True, help="h-arc/arc_data folder (for the target outputs)")
    g = sub.add_parser("game")
    g.add_argument("paths", nargs="+", help="log files or folders")
    pr = sub.add_parser("profile")
    pr.add_argument("files", nargs="+")
    args = ap.parse_args()

    if args.cmd == "harc":
        from stml.datasets.harc import convert
        recs = list(convert(args.data, args.arc, complete_only=not args.all_participants))
        n = write_jsonl(recs, OUT / "harc.jsonl")
        print(f"wrote {n} records to {OUT / 'harc.jsonl'}")
        print_profile("H-ARC", profile(recs))
    elif args.cmd == "arctraj":
        from stml.datasets.arctraj import convert
        recs = list(convert(args.csv, args.arc))
        n = write_jsonl(recs, OUT / "arctraj.jsonl")
        print(f"wrote {n} records to {OUT / 'arctraj.jsonl'}")
        print_profile("ARCTraj", profile(recs))
    elif args.cmd == "game":
        from stml.datasets.stml_game import convert
        docs = []
        for p in map(Path, args.paths):
            for f in (sorted(p.rglob("*.json")) if p.is_dir() else [p]):
                obj = json.loads(f.read_text())
                docs.extend(obj if isinstance(obj, list) else [obj])
        recs = list(convert(docs))
        n = write_jsonl(recs, OUT / "game.jsonl")
        print(f"wrote {n} records to {OUT / 'game.jsonl'}")
        print_profile("Wugs and Daxes", profile(recs))
    else:
        for f in args.files:
            print_profile(Path(f).stem, profile(list(read_jsonl(f))))


if __name__ == "__main__":
    main()

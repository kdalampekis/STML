"""Print the learning curve from one or more game logs.

Usage:  python scripts/analyze_log.py logs/pilot-01            (a folder of logs)
        python scripts/analyze_log.py stml-log-pilot-01-*.json  (downloaded files)
Accepts the game's download (one session), the per-world files written by
scripts/play_local.py, or documents exported from the page's research store.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.analysis import ERROR_TYPES  # noqa: E402
from stml.trajectories import load_events, summary, to_trajectories  # noqa: E402


def main(paths: list[str]) -> None:
    docs: list = []
    files: list[Path] = []
    for p in map(Path, paths):
        files.extend(sorted(p.rglob("*.json")) if p.is_dir() else [p])
    for f in files:
        obj = json.loads(f.read_text())
        docs.extend(obj if isinstance(obj, list) else [obj])
    trajs = to_trajectories(load_events(docs))
    rows = summary(trajs)
    print(f"{'world':7} {'claims':>6} {'found':>9} {'precision':>9} {'brier':>6} {'reveals':>7}  errors")
    for r in rows:
        errs = ", ".join(f"{k}x{v}" for k, v in sorted(r["errors"].items(), key=lambda kv: -kv[1]))
        brier = "-" if r["brier"] is None else f"{r['brier']:.2f}"
        print(f"{r['episode']:7} {r['claims']:>6} {r['found']:>4}/{r['total']:<4} {r['precision']:>9.2f} {brier:>6} {r['reveals']:>7}  {errs}")
    totals: dict[str, int] = {}
    for r in rows:
        for k, v in r["errors"].items():
            totals[k] = totals.get(k, 0) + v
    if totals:
        print("\nError types:")
        for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"  {v:3}  {k}: {ERROR_TYPES[k]}")
    probe_rows = [(r["episode"], r["probe"]) for r in rows if r["probe"]]
    if probe_rows:
        from stml.metacog import items_from_probe_events, metrics
        print("\nTest rounds (knowing what you do not know):")
        print(f"{'world':7} {'answered':>8} {'right when answered':>20} {'passed':>7} {'brier':>6}")
        for eid, m in probe_rows:
            acc = "-" if m["selective_accuracy"] is None else f"{m['selective_accuracy']:.2f}"
            br = "-" if m["brier"] is None else f"{m['brier']:.2f}"
            print(f"{eid:7} {m['coverage']:>8.2f} {acc:>20} {m['abstain_rate']:>7.2f} {br:>6}")
        allm = metrics(items_from_probe_events(e for t in trajs for e in t.probes))
        print(f"overall: answered {allm['coverage']:.0%}, right when answered "
              f"{(allm['selective_accuracy'] or 0):.0%}, honest score {allm['honest_score']:.2f}")
    reflections = [(t.episode, t.reflection) for t in trajs if t.reflection]
    if reflections:
        print("\nReflections:")
        for eid, text in reflections:
            print(f"  {eid}: {text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])

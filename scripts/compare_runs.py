"""Compare finished runs side by side: one table per run, same metrics.

    python scripts/compare_runs.py                  # every run in runs/ that has results.json
    python scripts/compare_runs.py runs/<a> runs/<b>

Reads runs/<run>/results.json (baseline_v1 stores [mean, sd] per metric under
'summary'; harc_h1_stml stores plain means under 'mean'). Writes runs/COMPARISON.md.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLS = [("solved", "Solved in 3"), ("first", "Right first"), ("recover", "Recovers"),
        ("human_loglik", "Human-likeness"), ("human_loglik_revisions", "(revisions)"),
        ("brier", "Brier"), ("ece", "ECE")]
ALIAS = {"fit_all": "human_loglik", "fit_revision": "human_loglik_revisions"}


def load(run: Path) -> dict[str, dict[str, float]]:
    d = json.loads((run / "results.json").read_text())
    src = d.get("mean") or d.get("summary") or {}
    out = {}
    for cond, ms in src.items():
        out[cond] = {ALIAS.get(k, k): (v[0] if isinstance(v, list) else v) for k, v in ms.items()}
    return out


def fmt(k, v):
    if v is None or v != v:
        return "-"
    return f"{v:.1%}" if k in ("solved", "first", "recover") else f"{v:.2f}" if k.startswith("human") else f"{v:.3f}"


def main():
    runs = [Path(p) for p in sys.argv[1:]] or sorted(p.parent for p in (ROOT / "runs").glob("*/results.json"))
    md = ["# Run comparison\n"]
    for run in runs:
        try:
            t = load(run)
        except Exception as e:
            md.append(f"## {run.name}\n\n(unreadable: {e})\n")
            continue
        cols = [c for c in COLS if any(c[0] in m for m in t.values())]
        md.append(f"## {run.name}\n")
        md.append("| Condition | " + " | ".join(n for _, n in cols) + " |")
        md.append("| --- | " + " | ".join("---:" for _ in cols) + " |")
        for cond, m in t.items():
            md.append(f"| {cond} | " + " | ".join(fmt(k, m.get(k)) for k, _ in cols) + " |")
        md.append("")
    out = ROOT / "runs" / "COMPARISON.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"wrote {out}")


main()

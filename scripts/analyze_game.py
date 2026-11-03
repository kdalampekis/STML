"""Analyse game logs: the planned analyses E1-E3, learning curves, sizing update, comparison with the simulation.

Usage:  python scripts/analyze_game.py logs/            (all participants under logs/)
        python scripts/analyze_game.py logs/pilot-01 logs/pilot-02
Writes a run folder runs/<time>_game_analysis/ (report.md, results.json). Never modifies the logs.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.game_analysis import analyze, by_participant, load_docs, render  # noqa: E402
from stml.runlog import Run  # noqa: E402


def sim_assumptions() -> dict | None:
    runs = sorted((ROOT / "runs").glob("*_sim_study_power/results.json"))
    if not runs:
        return None
    s = json.loads(runs[-1].read_text())["scenarios"]["hc=1.0|gain=0.35"]["summary"]
    return {"acc_all": f"{s['acc_all_scalar']:.3f}", "abstain": f"{s['abstain_scalar']:.3f}", "n_errors": f"{s['errors_per_person']:.1f}", "auroc": "0.64"}


def main() -> None:
    paths = sys.argv[1:] or [str(ROOT / "logs")]
    docs = load_docs(paths)
    if not docs:
        sys.exit("no log files found")
    with Run("game_analysis", {"paths": paths, "n_docs": len(docs)}) as run:
        res = analyze(by_participant(docs))
        run.log(f"{res['n_participants']} participants, {res['n_completed']} finished")
        run.finish(res, render(res, sim_assumptions()))


if __name__ == "__main__":
    main()

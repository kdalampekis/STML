"""Convert ARCTraj (Kim et al., GIST DSLab; Hugging Face SejinKimm/ARCTraj) to stml-traj/1.

Real layout (checked on the downloaded ARCTraj.csv, 236 MB):
  columns logId, userId, taskId (8-char ARC training-task hash), actionSequence (JSON),
  startedAt, endedAt. 10,670 sessions, 101 users, 401 task ids.
  Each action: category, operation (SelectCell, SelectGrid, Paint, Paste, Copy, Move,
  ResizeGrid, SelectObject, Flip, Rotate, Undo, Redo, Submit), the full grid after
  the action, optional position/color/direction/source, an ISO timestamp.
  Every session has exactly ONE Submit, and its special.succeed flag is False on all
  10,669 of them, so the flag is unusable: correctness is recomputed here against
  the first test output of the ARC training task (from the H-ARC task files).

Consequence for research: there is no wrong-answer -> feedback -> revision loop.
Each record is a single attempt (feedback is the recomputed right/wrong, not
something the person saw). Use it for action-level Stage A pretraining, not for
learning-from-feedback (H1) tests.

Cleaning: unparseable sessions and sessions without a Submit are dropped; times are
relative to the session's first action; task.order = number of earlier sessions by
the same user (by startedAt).
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from .harc import grid_distance, load_targets


def _s(grid) -> str:
    return "|" + "|".join("".join(str(c) for c in row) for row in grid) + "|"


def _ts(x):
    from datetime import datetime
    return datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp() * 1000


def convert(csv_path: str | Path, arc_dir: str | Path) -> Iterator[dict]:
    csv.field_size_limit(sys.maxsize)
    targets = {name.removesuffix(".json"): g for (tt, name), g in load_targets(arc_dir).items() if tt == "training"}
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    order: dict[str, int] = defaultdict(int)
    for r in sorted(rows, key=lambda r: r["startedAt"]):
        try:
            acts = json.loads(r["actionSequence"])
        except json.JSONDecodeError:
            continue
        subs = [i for i, a in enumerate(acts) if a.get("operation") == "Submit"]
        target = targets.get(r["taskId"])
        if not subs or target is None:
            continue
        acts = acts[: subs[0] + 1]
        t0 = _ts(acts[0]["timestamp"])
        actions, best = [], None
        for a in acts[:-1]:
            g = _s(a["grid"])
            d = grid_distance(g, target)
            if d is not None:
                best = d if best is None else min(best, d)
            args = {k: a[k] for k in ("position", "color", "direction", "source") if k in a}
            actions.append({"t": max(0, int(_ts(a["timestamp"]) - t0)), "op": a["operation"], "args": args, "d": d})
        sub = acts[-1]
        ans = _s(sub["grid"])
        d_sub = grid_distance(ans, target)
        ok = d_sub == 0
        uid = r["userId"]
        n_cells = sum(len(x) for x in target.strip("|").split("|"))
        yield {
            "schema": "stml-traj/1", "source": "arctraj", "domain": "arc",
            "learner": {"id": f"arctraj-{uid}", "kind": "human"},
            "task": {"id": r["taskId"] + ".json", "order": order[uid]},
            "attempts": [{
                "index": 0, "actions": actions,
                "commit": {"t": max(0, int(_ts(sub["timestamp"]) - t0)), "answer": ans, "confidence": None},
                "feedback": {"correct": ok, "kind": "binary", "detail": {"recomputed": True}},
                "explanation": None,
                "analysis": {"cells_wrong": d_sub, "change": None, "best_during": best,
                             "n_actions": len(actions), "target_cells": n_cells},
            }],
            "explanations": [], "judgments": [],
            "outcome": {"solved": ok, "n_attempts": 1},
            "meta": {"log_id": r["logId"], "n_undo": sum(a["operation"] == "Undo" for a in acts),
                     "n_redo": sum(a["operation"] == "Redo" for a in acts), "target": target},
        }
        order[uid] += 1

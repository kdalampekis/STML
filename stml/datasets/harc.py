"""Convert H-ARC (LeGris, Vong, Lake & Gureckis, Scientific Data 2025) to stml-traj/1.

Inputs, from the OSF archive (https://osf.io/bh8yq) and the GitHub repo
(https://github.com/Le-Gris/h-arc):
  data/data.csv             one row per action, with the full output grid after the action
  data/summary_data.csv     one row per attempt, with the participant's rule descriptions
  arc_data/ARC_*_tasks.csv  the original tasks; the first test output is the target

What the real data looks like (checked on the October 2026 OSF archive):
  - 586,266 actions, 8,899 participant x task pairs, 1,729 participants, 800 tasks.
  - An attempt ends with an explicit 'submit' action; 'solved' on that row is the
    right/wrong feedback the participant saw.
  - The first rule description is written during attempt 1, before any feedback
    ('write_first_description'). After the final submission people either write a
    revised description ('write_last_description') or keep the first
    ('no_last_description'). The texts are in summary_data.csv, one pair per task.
  - Grids are strings like '|077|777|077|'.

What each piece becomes:
  participant x task          -> one record
  actions of an attempt       -> actions; each carries 'd' = cells still wrong
                                 after it (None while the grid has the wrong size)
  'submit'                    -> commit (answer = grid) + binary feedback
  first description           -> attempts[0].explanation and explanations[0]
  revised description         -> explanations[1] (after the last attempt)
  analysis per attempt        -> cells_wrong at submit, change since the previous
                                 submit, best distance reached during the attempt

Cleaning: rows sorted by attempt and action_id; times relative to the task's first
action; only participants who finished the experiment unless asked otherwise;
attempts without a submit (people who quit mid-attempt) are dropped; descriptions
of fewer than 4 words, or stock phrases, are kept but flagged low-quality.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import pandas as pd

DESCRIBE_OPS = {"write_first_description", "write_last_description", "no_last_description", "write_description"}
_LOW_QUALITY = re.compile(r"^\s*(yes|no|done|ok|okay|easy|hard|idk|i don'?t know|none|n/?a|good|nice|fun)\b", re.I)


def parse_grid(s) -> list[str] | None:
    """'|077|777|' -> ['077', '777']"""
    if not isinstance(s, str) or "|" not in s:
        return None
    rows = s.strip().strip("|").split("|")
    return rows if rows and rows != [""] else None


def grid_distance(a, b) -> int | None:
    """Cells that differ; None if either is missing or the shapes differ."""
    ga, gb = parse_grid(a), parse_grid(b)
    if ga is None or gb is None or len(ga) != len(gb) or any(len(x) != len(y) for x, y in zip(ga, gb)):
        return None
    return sum(ca != cb for ra, rb in zip(ga, gb) for ca, cb in zip(ra, rb))


def description_quality(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return "missing"
    words = text.split()
    if len(words) < 4 or (_LOW_QUALITY.match(text) and len(words) < 8):
        return "low"
    return "ok"


def load_targets(arc_dir: str | Path) -> dict[tuple[str, str], str]:
    """(task_type, task_name) -> first test output grid."""
    frames = [pd.read_csv(f) for f in sorted(Path(arc_dir).glob("ARC_*_tasks.csv"))]
    t = pd.concat(frames)
    t = t[t.example_type == "test"].sort_values("example_number").drop_duplicates(["task_type", "task_name"])
    return {(a, b): g for a, b, g in zip(t.task_type, t.task_name, t.output_grid)}


def _ms(series: pd.Series) -> pd.Series:
    t = pd.to_datetime(series, errors="coerce", format="mixed")
    return (t - t.min()).dt.total_seconds().mul(1000).round().fillna(0).astype(int)


def convert(data_dir: str | Path, arc_dir: str | Path, complete_only: bool = True,
            data: pd.DataFrame | None = None) -> Iterator[dict]:
    data_dir = Path(data_dir)
    targets = load_targets(arc_dir)
    summ = pd.read_csv(data_dir / "summary_data.csv")
    if data is None:
        data = pd.read_csv(data_dir / "data.csv", usecols=lambda c: c not in (
            "test_input_grid", "selected_data", "copy_paste_data"))
    if complete_only:
        data = data[data["complete"].astype(bool)]
        summ = summ[summ["complete"].astype(bool)]
    task_type = summ.drop_duplicates("joint_id_task").set_index("joint_id_task")["task_type"].to_dict()
    texts = summ.drop_duplicates("joint_id_task").set_index("joint_id_task")[
        ["first_written_solution", "last_written_solution", "exp_name"]].to_dict("index")
    data = data.sort_values(["joint_id_task", "attempt_number", "action_id"], kind="stable")

    for jid, rows in data.groupby("joint_id_task", sort=False):
        ttype = task_type.get(jid) or (rows["task_type"].iloc[0] if "task_type" in rows.columns else None)
        task = rows["task_name"].iloc[0]
        target = targets.get((ttype, task))
        t = _ms(rows["time"])
        rows = rows.assign(tms=t.values)
        txt = texts.get(jid, {})
        attempts, explanations = [], []
        prev_grid, solved_any = None, False
        cur_actions: list[dict] = []
        best = None
        first_desc_t = revised_t = None
        revised_written = False
        for r in rows.itertuples(index=False):
            op = r.action if isinstance(r.action, str) else "unknown"
            if op in DESCRIBE_OPS:
                if op == "write_first_description" and first_desc_t is None:
                    first_desc_t = int(r.tms)
                elif op == "write_last_description":
                    revised_t, revised_written = int(r.tms), True
                elif op == "no_last_description":
                    revised_t = int(r.tms)
                continue
            grid = r.test_output_grid
            if op == "submit":
                if solved_any:
                    continue  # nothing after a correct answer is part of learning this task
                ok = bool(r.solved)
                d_sub = grid_distance(grid, target)
                attempts.append({
                    "index": len(attempts),
                    "actions": cur_actions,
                    "commit": {"t": int(r.tms), "answer": grid, "confidence": None},
                    "feedback": {"correct": ok, "kind": "binary", "detail": {}},
                    "explanation": None,
                    "analysis": {"cells_wrong": d_sub,
                                 "change": grid_distance(prev_grid, grid) if prev_grid is not None else None,
                                 "best_during": best,
                                 "n_actions": len(cur_actions),
                                 "target_cells": None if parse_grid(target) is None
                                 else sum(len(x) for x in parse_grid(target))},
                })
                prev_grid, cur_actions, best = grid, [], None
                solved_any = solved_any or ok
                continue
            if solved_any:
                continue
            d = grid_distance(grid, target)
            if d is not None:
                best = d if best is None else min(best, d)
            args = {}
            if pd.notna(r.action_x):
                args["x"], args["y"] = int(r.action_x), int(r.action_y)
            if pd.notna(r.selected_symbol):
                args["color"] = int(r.selected_symbol)
            if isinstance(r.selected_tool, str):
                args["tool"] = r.selected_tool
            cur_actions.append({"t": int(r.tms), "op": op, "args": args, "d": d})
        if not attempts:
            continue
        first_txt, last_txt = txt.get("first_written_solution"), txt.get("last_written_solution")
        if isinstance(first_txt, str) and first_txt.strip():
            attempts[0]["explanation"] = first_txt.strip()
            explanations.append({"t": first_desc_t, "after_attempt": None, "kind": "initial",
                                 "text": first_txt.strip(), "quality": description_quality(first_txt)})
        if revised_written and isinstance(last_txt, str) and last_txt.strip() and last_txt != first_txt:
            explanations.append({"t": revised_t, "after_attempt": len(attempts) - 1, "kind": "revised",
                                 "text": last_txt.strip(), "quality": description_quality(last_txt)})
        yield {
            "schema": "stml-traj/1",
            "source": "harc",
            "domain": "arc",
            "learner": {"id": rows["hashed_id"].iloc[0], "kind": "human"},
            "task": {"id": task, "order": int(rows["task_number"].iloc[0])},
            "attempts": attempts,
            "explanations": explanations,
            "judgments": [],
            "outcome": {"solved": solved_any, "n_attempts": len(attempts)},
            "meta": {"task_type": ttype, "exp_name": txt.get("exp_name"),
                     "target": target, "kept_first_description": not revised_written},
        }

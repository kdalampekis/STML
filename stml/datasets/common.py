"""A common format for learning trajectories from any source.

The premise (itself a hypothesis to test): the human learning loop has the same
shape in every domain. Someone forms a hypothesis, acts on it, commits to an
answer, gets feedback, and revises. So a Sudoku solve, an ARC puzzle and a
world in Wugs and Daxes can all be written as the same kind of record, and one
model can learn from all of them.

Schema "stml-traj/1", one JSON object per learner x task:

{
  "schema": "stml-traj/1",
  "source": "harc" | "stml-game" | "arctraj" | "sudoku-ctc" | "synthetic" | "rl-agent",
  "domain": "arc" | "syllogistic" | "sudoku" | ...,
  "learner": {"id": str, "kind": "human" | "machine"},
  "task":    {"id": str, "order": int | null},      # order = how many tasks this learner had done before
  "attempts": [
     {"index": 0,
      "actions": [{"t": ms, "op": str, "args": {...}}],   # fine-grained, domain-specific
      "commit":  {"t": ms, "answer": any, "confidence": 0..1 | null} | null,
      "feedback": {"correct": bool | null, "kind": "binary" | "diagnostic",
                   "detail": {...}},                       # diagnostic sources fill detail
      "explanation": str | null,                           # the learner's own words, if any
      "analysis": {...}}                                   # computed by the converter for research
                                                           # (e.g. revision size); never shown to a learner
  ],
  "explanations": [ {"t": ms, "after_attempt": int | null, "kind": "initial" | "revised" | "reflection",
                     "text": str, "quality": "ok" | "low" | "missing"} ],   # timed, in the learner's words
  "judgments": [ {"item": str, "answer": "true"|"false"|"open"|"unknown",
                  "confidence": 0..1 | null, "truth": str} ],  # epistemic probes, if any
  "outcome": {"solved": bool, "n_attempts": int},
  "meta": {...}
}

Times are milliseconds from the start of the task. Every converter must
produce records that pass validate().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA = "stml-traj/1"
SOURCES = {"harc", "stml-game", "arctraj", "sudoku-ctc", "synthetic", "rl-agent"}
JUDGMENTS = {"true", "false", "open", "unknown"}


class SchemaError(ValueError):
    pass


def validate(rec: dict) -> dict:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise SchemaError(msg)

    need(rec.get("schema") == SCHEMA, "schema must be stml-traj/1")
    need(rec.get("source") in SOURCES, f"unknown source {rec.get('source')!r}")
    need(isinstance(rec.get("domain"), str) and rec["domain"], "domain required")
    lr = rec.get("learner") or {}
    need(isinstance(lr.get("id"), str) and lr.get("kind") in ("human", "machine"), "bad learner")
    need(isinstance((rec.get("task") or {}).get("id"), str), "task.id required")
    atts = rec.get("attempts")
    need(isinstance(atts, list), "attempts must be a list")
    last_t = -1
    for i, a in enumerate(atts):
        need(a.get("index") == i, f"attempt {i} has index {a.get('index')}")
        for act in a.get("actions", []):
            need(isinstance(act.get("op"), str), "action.op required")
            need(isinstance(act.get("t"), (int, float)) and act["t"] >= 0, "action.t must be >= 0")
        c = a.get("commit")
        if c is not None:
            need(isinstance(c.get("t"), (int, float)), "commit.t required")
            conf = c.get("confidence")
            need(conf is None or 0 <= conf <= 1, "confidence must be in [0, 1]")
            need(c["t"] >= last_t, "commits must be in time order")
            last_t = c["t"]
        fb = a.get("feedback")
        if fb is not None:
            need(fb.get("kind") in ("binary", "diagnostic"), "feedback.kind must be binary or diagnostic")
    for j in rec.get("judgments", []):
        need(j.get("answer") in JUDGMENTS, "bad judgment answer")
    out = rec.get("outcome") or {}
    need(isinstance(out.get("solved"), bool), "outcome.solved required")
    need(out.get("n_attempts") == len(atts), "outcome.n_attempts must equal len(attempts)")
    return rec


def write_jsonl(records: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(validate(rec), separators=(",", ":")) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def learning_events(rec: dict) -> list[dict]:
    """The domain-free view: one row per commit, the loop a model imitates.

    For each attempt: was it right, how much effort went in, how long since
    the last feedback, and how much the answer changed after being told it
    was wrong (revision size, filled in by the converter as analysis['change']).
    """
    rows = []
    prev_t = 0
    for a in rec["attempts"]:
        c, fb = a.get("commit"), a.get("feedback") or {}
        if c is None:
            continue
        rows.append({
            "attempt": a["index"],
            "correct": fb.get("correct"),
            "n_actions": len(a.get("actions", [])),
            "think_ms": c["t"] - prev_t,
            "change": (a.get("analysis") or {}).get("change"),
            "explained": bool(a.get("explanation")),
            "confidence": c.get("confidence"),
        })
        prev_t = c["t"]
    return rows

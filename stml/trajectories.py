"""Load human learning logs from the game and turn them into trajectories.

Log schema ("stml-log/1"): a JSON object
  {"schema": "stml-log/1",
   "session": {"id", "participant", "started_at", "game_version"},
   "events": [ ... ]}
or a JSON list of such events. Each event has "type", "t" (ms since the
session started) and "episode" (episode id, e.g. "L2-1"), plus:

  episode_start  {}
  claim          {"claim": key, "confidence": 1-5, "used": [premise idx],
                  "why": free text, "status": critic status, "rt_ms": int}
  reveal         {"claim": key}          person asked to see the full answer
  probe          {"claim": key, "answer": "true"|"false"|"open"|"unknown",
                  "confidence": 1-5 or null, "truth": referee status, "rt_ms": int}
  episode_end    {"reason": "complete"|"skip", "found": int, "total": int,
                  "reflection": free text}

A trajectory is the ordered list of claim steps in one episode, each with
the critic's verdict and the error types it shows. That is exactly the
learning history the imitation model is trained on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .analysis import classify_error
from .critic import Status
from .episodes import world_for_episode
from .logic import Claim
from .metacog import items_from_probe_events, metrics

SCHEMA = "stml-log/1"


@dataclass
class Step:
    t: int
    claim: Claim
    status: Status
    confidence: int | None
    used: list[int]
    why: str
    rt_ms: int | None
    revealed: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.status.correct


@dataclass
class EpisodeTrajectory:
    episode: str
    steps: list[Step]
    found: int
    total: int
    reason: str
    reflection: str
    probes: list[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return sum(s.correct for s in self.steps) / len(self.steps) if self.steps else 0.0

    @property
    def recall(self) -> float:
        return self.found / self.total if self.total else 0.0


def load_events(path_or_obj) -> list[dict]:
    if isinstance(path_or_obj, (str, Path)):
        obj = json.loads(Path(path_or_obj).read_text())
    else:
        obj = path_or_obj
    docs = [obj] if isinstance(obj, dict) else list(obj)
    events: list[dict] = []
    for d in docs:
        if "events" in d:  # a log document (whole session download, or one episode doc from the store)
            if d.get("schema") != SCHEMA:
                raise ValueError(f"unknown log schema {d.get('schema')!r}")
            events.extend(d["events"])
        else:  # a bare event
            events.append(d)
    # the same event can appear in a full download and an episode doc
    seen, unique = set(), []
    for e in events:
        k = (e.get("t"), e.get("type"), e.get("episode"), e.get("claim"), e.get("answer"))
        if k not in seen:
            seen.add(k)
            unique.append(e)
    return sorted(unique, key=lambda e: e.get("t", 0))


def to_trajectories(events: list[dict]) -> list[EpisodeTrajectory]:
    by_episode: dict[str, dict] = {}
    order: list[str] = []
    for ev in events:
        eid = ev.get("episode")
        if eid is None:
            continue
        if eid not in by_episode:
            by_episode[eid] = {"steps": [], "end": None, "probes": []}
            order.append(eid)
        rec = by_episode[eid]
        if ev["type"] == "claim":
            claim = Claim.from_key(ev["claim"])
            rec["steps"].append(Step(
                t=ev.get("t", 0), claim=claim, status=Status(ev["status"]),
                confidence=ev.get("confidence"), used=list(ev.get("used", [])),
                why=ev.get("why", ""), rt_ms=ev.get("rt_ms")))
        elif ev["type"] == "reveal":
            for step in reversed(rec["steps"]):
                if step.claim.key() == ev["claim"]:
                    step.revealed = True
                    break
        elif ev["type"] == "probe":
            rec["probes"].append(ev)
        elif ev["type"] == "episode_end":
            rec["end"] = ev

    out = []
    for eid in order:
        rec = by_episode[eid]
        world = world_for_episode(eid)
        for s in rec["steps"]:
            if not s.correct:
                s.errors = classify_error(world, s.claim)
        end = rec["end"] or {}
        out.append(EpisodeTrajectory(
            episode=eid, steps=rec["steps"],
            found=end.get("found", sum(1 for s in rec["steps"] if s.status == Status.TARGET)),
            total=end.get("total", len(world.targets)),
            reason=end.get("reason", "unfinished"),
            reflection=end.get("reflection", ""), probes=rec["probes"]))
    return out


def summary(trajs: list[EpisodeTrajectory]) -> list[dict]:
    """One row per episode: the learning curve across the session."""
    rows = []
    for tr in trajs:
        conf = [s for s in tr.steps if s.confidence is not None]
        # Brier score of confidence (1..5 mapped to 0..1) against correctness
        brier = (sum(((s.confidence - 1) / 4 - s.correct) ** 2 for s in conf) / len(conf)) if conf else None
        errors: dict[str, int] = {}
        for s in tr.steps:
            for e in s.errors:
                errors[e] = errors.get(e, 0) + 1
        rows.append({
            "episode": tr.episode,
            "claims": len(tr.steps),
            "found": tr.found,
            "total": tr.total,
            "precision": round(tr.precision, 3),
            "recall": round(tr.recall, 3),
            "brier": None if brier is None else round(brier, 3),
            "reveals": sum(s.revealed for s in tr.steps),
            "errors": errors,
            "probe": metrics(items_from_probe_events(tr.probes)) if tr.probes else None,
        })
    return rows

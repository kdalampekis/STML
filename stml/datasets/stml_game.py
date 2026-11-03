"""Convert Wugs and Daxes game logs (stml-log/1) to stml-traj/1.

Each world a person plays becomes one record. Each claim is an attempt; the
critic's verdict is diagnostic feedback (status, which premises, which rule,
whether a countermodel exists). Test-round answers become judgments.
"""

from __future__ import annotations

from typing import Iterator

from ..logic import Claim
from ..trajectories import load_events, to_trajectories

CORRECT = {"target", "weak", "premise", "trivial"}


def _change(prev: Claim | None, cur: Claim) -> int | None:
    """How many of (form, first kind, second kind) changed since the last claim."""
    if prev is None:
        return None
    return int(prev.form != cur.form) + int(prev.a != cur.a) + int(prev.b != cur.b)


def convert(logs) -> Iterator[dict]:
    """logs: anything trajectories.load_events accepts (paths, dicts, lists)."""
    events = load_events(logs)
    session = next((e for e in events if e.get("type") == "session_start"), {})
    participant = session.get("participant") or "unknown"
    from ..episodes import world_for_episode
    for order, tr in enumerate(to_trajectories(events)):
        world = world_for_episode(tr.episode)
        crit_by_claim = {}
        try:
            from ..critic import SocraticCritic
            critic = SocraticCritic(world)
            crit_by_claim = {s.claim: critic.critique(s.claim) for s in tr.steps}
        except Exception:  # noqa: BLE001  (a log from an older game version)
            pass
        attempts, prev, t0 = [], None, (tr.steps[0].t - (tr.steps[0].rt_ms or 0)) if tr.steps else 0
        for s in tr.steps:
            c = crit_by_claim.get(s.claim)
            attempts.append({
                "index": len(attempts),
                "actions": [{"t": max(0, s.t - t0), "op": "mark_premise", "args": {"premise": i}} for i in s.used],
                "commit": {"t": max(0, s.t - t0), "answer": s.claim.key(),
                           "confidence": None if s.confidence is None else (s.confidence - 1) / 4},
                "feedback": {"correct": s.status.value in CORRECT, "kind": "diagnostic",
                             "detail": {"status": s.status.value,
                                        "support": c.support if c else None,
                                        "rules": c.rules if c else None,
                                        "countermodel": bool(c.countermodel) if c else None,
                                        "revealed": s.revealed}},
                "explanation": s.why or None,
                "analysis": {"change": _change(prev, s.claim), "errors": s.errors},
            })
            prev = s.claim
        judgments = [{"item": p["claim"], "answer": p["answer"],
                      "confidence": None if p.get("confidence") is None else (p["confidence"] - 1) / 4,
                      "truth": p["truth"]} for p in tr.probes]
        yield {
            "schema": "stml-traj/1",
            "source": "stml-game",
            "domain": "syllogistic",
            "learner": {"id": participant, "kind": "human"},
            "task": {"id": tr.episode, "order": order},
            "attempts": attempts,
            "judgments": judgments,
            "outcome": {"solved": tr.total > 0 and tr.found >= tr.total, "n_attempts": len(attempts)},
            "meta": {"found": tr.found, "total": tr.total, "end": tr.reason, "reflection": tr.reflection},
        }

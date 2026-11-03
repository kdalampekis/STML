"""Turn a synthetic participant (stml/synthetic.py) into a game log in the real `stml-log/1` format.

Purpose: test the log -> trajectory -> analysis pipeline end to end BEFORE real logs exist. The emitted documents have the same event types and fields as the
game writes (session_start, episode_start, claim, probe, episode_end). Never write them into logs/ (that folder is for real data).
"""

from __future__ import annotations

import numpy as np

from .episodes import GAME_VERSION
from .synthetic import Participant, game_worlds

SCHEMA = "stml-log/1"


def participant_log(p: Participant, pid: str, seed: int = 0, started_at: str = "2026-10-20T10:00:00Z") -> dict:
    rng = np.random.default_rng(seed)
    worlds = game_worlds()
    events = [{"type": "session_start", "t": 0, "participant": pid, "arm": p.arm}]
    t = 1000
    by_world: dict[int, dict] = {}
    for rec in p.proposals:
        by_world.setdefault(rec[0], {"claims": [], "probes": []})["claims"].append(rec)
    for rec in p.probes:
        by_world.setdefault(rec[0], {"claims": [], "probes": []})["probes"].append(rec)
    last_world = max(by_world) if by_world else -1
    for wi in sorted(by_world):
        w = worlds[wi]
        events.append({"type": "episode_start", "t": t, "episode": w.eid})
        found = 0
        for (_, kind, etype, conf, key, status) in by_world[wi]["claims"]:
            rt = int(rng.integers(4000, 20000))
            t += rt
            found += kind == "correct"
            events.append({"type": "claim", "t": t, "episode": w.eid, "claim": key, "confidence": int(conf), "used": [], "why": "",
                           "status": status, "rt_ms": rt})
        unfinished = (not p.completed) and wi == last_world and not by_world[wi]["probes"]
        if by_world[wi]["probes"]:
            for (_, truth, answer, conf, correct, key) in by_world[wi]["probes"]:
                rt = int(rng.integers(3000, 15000))
                t += rt
                events.append({"type": "probe", "t": t, "episode": w.eid, "claim": key, "answer": answer,
                               "confidence": None if conf is None else int(conf), "truth": truth, "rt_ms": rt})
        if not unfinished:
            events.append({"type": "episode_end", "t": t + 500, "episode": w.eid, "reason": "complete" if found >= len(w.targets) else "skip",
                           "found": found, "total": len(w.targets), "reflection": ""})
            t += 1000
    return {"schema": SCHEMA, "session": {"id": f"synthetic-{pid}", "participant": pid, "started_at": started_at,
                                          "game_version": GAME_VERSION, "arm": p.arm}, "events": events}

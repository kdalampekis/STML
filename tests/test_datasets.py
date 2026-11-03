"""Converters: a miniature H-ARC built to the documented column layout, and a
game log, must both come out as valid stml-traj/1 with the right learning loop."""

import csv

import pytest

from stml.datasets.common import SchemaError, learning_events, validate
from stml.datasets.harc import convert as convert_harc, grid_distance
from stml.episodes import episode_payload

DATA_COLS = ["exp_name", "task_type", "hashed_id", "joint_id_task", "task_name", "task_number", "time",
             "attempt_number", "action_id", "solved", "done", "test_input_grid", "test_input_size_x",
             "test_input_size_y", "test_output_grid", "test_output_size_x", "test_output_size_y", "action",
             "action_x", "action_y", "select_loc", "selected_data", "selected_symbol", "selected_tool",
             "copy_paste_data", "complete"]
SUMMARY_COLS = ["exp_name", "task_type", "hashed_id", "joint_id_task", "task_name", "task_number",
                "attempt_number", "num_actions", "solved", "test_output_grid", "first_written_solution",
                "last_written_solution", "complete"]


def _write(path, cols, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


@pytest.fixture
def mini_harc(tmp_path):
    """Three people on one task, laid out like the real data: an explicit submit per
    attempt, the full grid after every action, timed description events."""
    data, arc = tmp_path / "data", tmp_path / "arc_data"
    data.mkdir(), arc.mkdir()
    target = "|12|34|"
    plan = {  # participant -> list of (grids after each edit, solved at submit)
        "p1": [(["|10|00|", "|00|00|"], False), (["|12|00|"], False), (["|12|30|", target], True)],
        "p2": [(["|12|00|", target], True)],
        "p3": [(["|00|00|"], False)],   # did not finish the experiment
    }
    drows, srows = [], []
    for pid, attempts in plan.items():
        sec, aid = 0, 0
        def row(att, action, grid, solved=False, x=None):
            nonlocal sec, aid
            sec += 2; aid += 1
            return {"exp_name": "x", "task_type": "training", "hashed_id": pid, "joint_id_task": f"{pid}_abc.json",
                    "task_name": "abc.json", "task_number": 1, "time": f"2023-08-03T04:22:{sec:02d}.000000",
                    "attempt_number": att, "action_id": aid, "solved": solved, "done": True,
                    "test_output_grid": grid, "action": action, "action_x": x, "action_y": 0 if x is not None else None,
                    "selected_symbol": 1, "selected_tool": "edit", "complete": pid != "p3"}
        for att, (grids, ok) in enumerate(attempts, 1):
            for k, g in enumerate(grids):
                drows.append(row(att, "edit", g, x=k))
                if att == 1 and k == 0:
                    drows.append(row(att, "write_first_description", g))
            drows.append(row(att, "submit", grids[-1], solved=ok))
            srows.append({"exp_name": "x", "task_type": "training", "hashed_id": pid,
                          "joint_id_task": f"{pid}_abc.json", "task_name": "abc.json", "task_number": 1,
                          "attempt_number": att, "num_actions": len(grids) + 1, "solved": ok,
                          "test_output_grid": grids[-1], "first_written_solution": "copy the input colours",
                          "last_written_solution": "put 1 2 on top and 3 4 below" if pid == "p1" else "copy the input colours",
                          "complete": pid != "p3"})
        drows.append(row(len(attempts) + (0 if attempts[-1][1] else 1),
                         "write_last_description" if pid == "p1" else "no_last_description", attempts[-1][0][-1]))
    _write(data / "data.csv", DATA_COLS, drows)
    _write(data / "summary_data.csv", SUMMARY_COLS, srows)
    _write(arc / "ARC_training_tasks.csv",
           ["task_type", "task_number", "task_name", "example_type", "example_number", "input_grid", "output_grid"],
           [{"task_type": "training", "task_number": 1, "task_name": "abc.json", "example_type": "test",
             "example_number": 1, "input_grid": "|00|00|", "output_grid": target}])
    return data, arc


def test_harc_conversion(mini_harc):
    data, arc = mini_harc
    recs = {r["learner"]["id"]: r for r in convert_harc(data, arc)}
    assert set(recs) == {"p1", "p2"}  # p3 did not finish
    p1 = validate(recs["p1"])
    assert [a["feedback"]["correct"] for a in p1["attempts"]] == [False, False, True]
    assert [a["analysis"]["cells_wrong"] for a in p1["attempts"]] == [4, 2, 0]
    assert [a["analysis"]["change"] for a in p1["attempts"]] == [None, 2, 2]
    assert p1["attempts"][0]["actions"][0]["t"] == 0 and p1["attempts"][0]["actions"][0]["d"] == 3
    assert all(a["op"] != "write_first_description" for a in p1["attempts"][0]["actions"])
    assert [e["kind"] for e in p1["explanations"]] == ["initial", "revised"]
    assert p1["explanations"][0]["t"] < p1["attempts"][0]["commit"]["t"]  # hypothesis before feedback
    assert p1["outcome"] == {"solved": True, "n_attempts": 3}
    assert recs["p2"]["meta"]["kept_first_description"] is True and len(recs["p2"]["explanations"]) == 1
    ev = learning_events(p1)
    assert [e["correct"] for e in ev] == [False, False, True] and ev[1]["think_ms"] > 0
    assert len(list(convert_harc(data, arc, complete_only=False))) == 3


def test_grid_distance():
    assert grid_distance("|12|34|", "|12|30|") == 1
    assert grid_distance("|12|", "|12|34|") is None


def test_game_conversion():
    from stml.datasets.stml_game import convert as convert_game
    p = episode_payload(1, 0)
    eid = p["id"]
    wrong = next(k for k, c in p["critiques"].items() if c["status"] == "undetermined")
    right = p["targets"][0]
    events = [
        {"type": "session_start", "t": 0, "episode": eid, "participant": "pilot-x"},
        {"type": "episode_start", "t": 5, "episode": eid},
        {"type": "claim", "t": 4000, "episode": eid, "claim": wrong, "confidence": 5, "used": [0],
         "why": "reversed it", "status": "undetermined", "rt_ms": 3995},
        {"type": "claim", "t": 9000, "episode": eid, "claim": right, "confidence": 3, "used": [0, 1],
         "why": "", "status": "target", "rt_ms": 5000},
        {"type": "probe", "t": 9500, "episode": eid, "claim": p["probes"][0]["key"], "answer": "unknown",
         "confidence": None, "truth": p["probes"][0]["truth"], "rt_ms": 400},
        {"type": "episode_end", "t": 9900, "episode": eid, "reason": "skip", "found": 1,
         "total": len(p["targets"]), "reflection": "direction matters"},
    ]
    rec = validate(next(convert_game(events)))
    assert rec["learner"]["id"] == "pilot-x"
    a0, a1 = rec["attempts"]
    assert a0["feedback"]["kind"] == "diagnostic" and a0["feedback"]["correct"] is False
    assert a0["feedback"]["detail"]["countermodel"] is True
    assert a0["commit"]["confidence"] == 1.0 and a0["explanation"] == "reversed it"
    assert a0["analysis"]["errors"] and a1["feedback"]["correct"] is True
    assert rec["judgments"][0]["answer"] == "unknown"


def test_validate_rejects_bad_records():
    with pytest.raises(SchemaError):
        validate({"schema": "stml-traj/1", "source": "nowhere"})


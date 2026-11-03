"""End-to-end test of the game-log pipeline on synthetic logs in the real stml-log/1 format (never written to logs/)."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.datasets.common import validate  # noqa: E402
from stml.datasets.stml_game import convert  # noqa: E402
from stml.game_analysis import analyze, by_participant, load_docs, participant_endpoints, render  # noqa: E402
from stml.synthetic import Population, simulate_participant  # noqa: E402
from stml.synthetic_logs import participant_log  # noqa: E402


def _study(tmp_path, n_per_arm=30, gain=1.0, seed=0):
    rng = np.random.default_rng(seed)
    pop = Population()
    for arm in ("scalar", "diagnostic"):
        for i in range(n_per_arm):
            p = simulate_participant(rng, arm, pop, gain)
            (tmp_path / f"{arm}-{i:03d}.json").write_text(json.dumps(participant_log(p, f"{arm[:3]}-{i:03d}", seed=i)))
    return tmp_path


def test_synthetic_logs_go_through_the_real_loader_and_converter(tmp_path):
    p = simulate_participant(np.random.default_rng(1), "diagnostic", Population(), 1.0)
    doc = participant_log(p, "sim-1")
    assert doc["schema"] == "stml-log/1" and {e["type"] for e in doc["events"]} >= {"session_start", "episode_start", "claim", "probe", "episode_end"}
    recs = list(convert(doc))
    assert recs and all(validate(r) for r in recs)


def test_endpoints_recomputed_from_the_log_match_the_simulation(tmp_path):
    from stml.synthetic import endpoints
    p = simulate_participant(np.random.default_rng(4), "scalar", Population(), 1.0)
    sim, log = endpoints(p), participant_endpoints([participant_log(p, "x")])
    assert abs(sim["acc_all"] - log["acc_all"]) < 1e-9 and abs(sim["abstain"] - log["abstain"]) < 1e-9
    assert (np.isnan(sim["auroc"]) and np.isnan(log["auroc"])) or abs(sim["auroc"] - log["auroc"]) < 1e-9
    assert sim["n_errors"] == log["n_errors"]


def test_full_analysis_runs_and_recovers_the_built_in_effects(tmp_path):
    folder = _study(tmp_path, n_per_arm=30)
    res = analyze(by_participant(load_docs([folder])))
    assert res["n_participants"] == 60 and res["arms"] == {"diagnostic": 30, "scalar": 30}
    assert res["E1_metacognition"]["mean"] > 0.05 and res["E1_metacognition"]["p"] < 0.001      # confidence is informative by construction
    e3 = res["E3_feedback_arm"]
    assert e3["diff_points"] > 3 and e3["n_scalar"] >= 15                                       # a built-in feedback effect of about 8 points
    assert "sizing" in res and res["sizing"]["n_per_arm_80pct"]["8"] < res["sizing"]["n_per_arm_80pct"]["4"]
    text = render(res, {"acc_all": "0.585"})
    assert "Planned analyses" in text and "E3 feedback arm" in text


def test_analysis_degrades_gracefully_with_almost_no_data(tmp_path):
    p = simulate_participant(np.random.default_rng(2), "scalar", Population(), 1.0)
    (tmp_path / "one.json").write_text(json.dumps(participant_log(p, "solo")))
    res = analyze(by_participant(load_docs([tmp_path])))
    assert res["E1_metacognition"] is None and "E3_feedback_arm" not in res
    assert "not enough data" in render(res)

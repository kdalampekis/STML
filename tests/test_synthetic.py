"""Tests for the synthetic human-like learner and the sizing helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from stml.synthetic import Population, auroc, endpoints, game_worlds, simulate_participant  # noqa: E402


def _acc(pop, arm="scalar", gain=1.0, n=150, seed=0):
    rng = np.random.default_rng(seed)
    e = [endpoints(simulate_participant(rng, arm, pop, gain)) for _ in range(n)]
    return float(np.nanmean([x["acc_all"] for x in e]))


def test_game_items_are_the_real_ones():
    w = game_worlds()
    assert len(w) == 15 and all(len(x.probes) == 4 for x in w)
    assert {p.truth for x in w for p in x.probes} == {"follows", "contradicted", "open"}
    assert all(p.support >= 1 for x in w for p in x.probes if p.truth != "open")


def test_same_seed_same_participant():
    a = simulate_participant(np.random.default_rng(5), "diagnostic", Population(), 1.0)
    b = simulate_participant(np.random.default_rng(5), "diagnostic", Population(), 1.0)
    assert a.probes == b.probes and a.proposals == b.proposals


def test_deeper_reasoners_score_higher():
    assert _acc(Population(depth_mean=2.6)) > _acc(Population(depth_mean=0.8)) + 0.05


def test_no_benefit_setting_gives_no_arm_difference_but_real_gain_does():
    null = _acc(Population(), "diagnostic", 0.35, n=400) - _acc(Population(), "scalar", n=400, seed=1)
    gain = _acc(Population(), "diagnostic", 1.0, n=400) - _acc(Population(), "scalar", n=400, seed=1)
    assert abs(null) < 0.03 and gain > 0.05


def test_dropout_rate_matches_the_setting():
    rng = np.random.default_rng(0)
    done = np.mean([simulate_participant(rng, "scalar").completed for _ in range(600)])
    assert 0.75 < done < 0.88


def test_endpoints_are_well_formed():
    e = endpoints(simulate_participant(np.random.default_rng(2), "scalar"))
    for k in ("acc_all", "abstain"):
        assert 0 <= e[k] <= 1
    assert e["n_errors"] >= 0 and (np.isnan(e["auroc"]) or 0 <= e["auroc"] <= 1)


def test_auroc_with_ties_and_perfect_separation():
    assert auroc(np.array([3, 4, 5, 1, 2.0]), np.array([1, 1, 1, 0, 0.0])) == 1.0
    assert auroc(np.array([1, 1, 1, 1.0]), np.array([1, 1, 0, 0.0])) == 0.5      # all tied
    assert np.isnan(auroc(np.array([1, 2.0]), np.array([1, 1.0])))               # one class only


def test_power_helpers_type_one_error_and_power():
    from sim_study_power import onesample_p, welch_p
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, (500, 40)), rng.normal(0, 1, (500, 40))
    assert (welch_p(a, b) < 0.05).mean() < 0.09                                  # about 5% false positives
    assert (welch_p(a, b + 1.0) < 0.05).mean() > 0.95                            # a 1 sd shift is detected
    x = rng.normal(0, 1, (500, 40))
    assert (onesample_p(x) < 0.05).mean() < 0.09 and (onesample_p(x + 1.0) < 0.05).mean() > 0.95

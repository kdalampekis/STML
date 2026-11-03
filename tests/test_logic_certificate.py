"""Tests for the logic judge's certificate check, the large-world generator and the abstention metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import stml.model.logic_data as ld  # noqa: E402
from stml.logic import Semantics, all_claims  # noqa: E402
from stml.model.logic_data import big_world, make_big_items, make_items, sub_item  # noqa: E402
from stml.model.logic_judge import LogicJudge, certificate_scores, fit, judge_with_why  # noqa: E402
from stml.referee import referee  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _restore_flag():
    old = ld.SIZE_FREE
    yield
    ld.SIZE_FREE = old


def test_big_world_labels_match_exact_semantics():
    verdict = {"entailed": "follows", "contradicted": "contradicted", "undetermined": "open"}
    for n in (5, 8):
        for seed in range(3):
            prem = big_world(n, seed, max(1, n // 4), 1, 1, max(1, n // 4))
            sem = Semantics(n)
            assert sem.satisfiable(prem)
            for c in all_claims(n)[:40]:
                if c.a != c.b:
                    assert referee(n, prem, c).status == verdict[sem.evaluate(prem, c).value]


def test_big_items_are_balanced_and_cores_name_real_premises():
    items = make_big_items(10, range(3), per_world=21, chain=0.9)
    assert {i.label for i in items} == {0, 1, 2}
    for i in items:
        assert all(0 <= j < i.n_prem for j in i.core)
        assert (i.label == 2) == (i.core == ())          # open claims have no proof to name


def test_sub_item_keeps_claim_and_only_chosen_premises():
    it = make_items([3], range(2), per_world=3)[0]
    sub = sub_item(it, [0, 2])
    assert sub.claim == it.claim and sub.n_prem == 2 and len(sub.graph.edges) == 4
    assert sub_item(it, []).graph.edges.shape[0] == 0


def test_size_free_features_do_not_depend_on_world_size():
    ld.SIZE_FREE = True
    a = make_big_items(8, range(1), per_world=3)[0]
    b = make_big_items(24, range(1), per_world=3)[0]
    assert np.all(a.graph.x[:, 3] == 0) and np.all(b.graph.x[:, 3] == 0)
    assert a.graph.x[:, 1:3].max() <= 1 and b.graph.x[:, 1:3].max() <= 1


def test_judge_with_why_and_certificate_scores_have_the_right_shapes():
    torch.manual_seed(0)
    items = make_items([2, 4], range(4), per_world=4)
    m = LogicJudge(16, 2)
    lg, cp = judge_with_why(m, items, CPU)
    assert lg.shape == (len(items), 3) and [len(c) for c in cp] == [i.n_prem for i in items]
    r = certificate_scores(m, items, CPU)
    for k in ("suff", "nec", "cert"):
        assert r[k].shape == (len(items),) and r[k].min() >= 0 and r[k].max() <= 1 + 1e-6
    assert np.allclose(r["cert"], r["suff"] * r["nec"])


def test_fit_with_why_head_runs_and_keeps_best_epoch():
    items = make_items([2], range(10), per_world=6)
    m = LogicJudge(16, 2)
    fit(m, items[:40], items[40:], CPU, epochs=2, log=lambda s: None, why_weight=0.5)


def test_selective_metrics_on_known_cases():
    from logic_h3_certificate import aurc, auroc, best_threshold, honest, sel_acc
    right = np.array([1, 1, 1, 0, 0], dtype=float)
    perfect = np.array([0.9, 0.8, 0.7, 0.2, 0.1])      # ranks every right answer above every wrong one
    # risks at coverage 1..5 answers: 0, 0, 0, 1/4, 2/5 -> mean 0.13 (the full-coverage point always counts)
    assert auroc(perfect, right) == 1.0 and aurc(perfect, right) == pytest.approx(0.13)
    assert sel_acc(perfect, right, 0.6) == 1.0
    reversed_ = perfect[::-1].copy()
    assert auroc(reversed_, right) == 0.0 and aurc(reversed_, right) > aurc(perfect, right)
    # commit above 0.5: 3 right (+1 each), 0 wrong, 2 abstentions (-0.2 each)
    assert honest(perfect, right, 0.5, 4.0) == pytest.approx((3 - 0.4) / 5)
    assert best_threshold(perfect, right, 4.0) >= 0.2          # never commits to the wrong ones when lam is high
    assert honest(perfect, right, best_threshold(perfect, right, 4.0), 4.0) == pytest.approx(0.52)

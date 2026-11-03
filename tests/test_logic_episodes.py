"""Tests for hidden-premise episodes with feedback."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stml.model.logic_data as ld  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_graph, episode_items, flavour_of, make_episodes, oracle_items  # noqa: E402
from stml.model.logic_judge import LogicJudge, logits_of  # noqa: E402


@pytest.fixture(autouse=True)
def _flag():
    old = ld.SIZE_FREE
    ld.SIZE_FREE = True
    yield
    ld.SIZE_FREE = old


def test_episodes_hide_some_premises_and_labels_use_all_of_them():
    eps = make_episodes([3, 4], range(6), k=5)
    for e in eps:
        assert 1 <= len(e.visible) < len(e.full) or len(e.full) == 1
        assert len(e.queries) == len(e.labels) == 5 and all(l in (0, 1, 2) for l in e.labels)
        assert set(map(repr, e.visible)) <= set(map(repr, e.full))


def test_revealed_claims_become_edges_only_in_feedback_modes():
    e = make_episodes([4], range(1), k=6)[0]
    n_static = len(episode_graph(e, 4, "static").edges)
    for mode in ("asked", "fb"):
        assert len(episode_graph(e, 4, mode).edges) == n_static + 8          # 4 revealed claims x 2 directed edges
    assert len(episode_graph(e, 0, "fb").edges) == n_static                   # nothing revealed at t = 0
    g = episode_graph(e, 5, "fb_s", [0, 1, 2, 0, 1, 2])
    assert g.edges[:, 2].max() < N_TYPES["fb_s"] and g.n_types == N_TYPES["fb_s"]


def test_query_flags_mark_the_asked_claim_and_not_the_future():
    e = make_episodes([3], range(1), k=4)[0]
    g = episode_graph(e, 2, "fb")
    q = e.queries[2]
    assert g.x[q.a, 4] == 1 and g.x[q.b, 5] == 1
    # future claims never appear: the graph at t=2 has exactly 2 revealed claims
    assert len(episode_graph(e, 2, "fb").edges) == len(episode_graph(e, 0, "fb").edges) + 4


def test_flavours_cover_the_signed_cases():
    assert flavour_of(np.array([0.9, 0.05, 0.05]), 0) == 0                 # expected
    assert flavour_of(np.array([0.1, 0.1, 0.8]), 0) == 1                   # thought open, it follows
    assert flavour_of(np.array([0.8, 0.1, 0.1]), 2) == 2                   # thought derivable, it is open


def test_every_mode_runs_through_the_judge_and_oracle_sees_all_premises():
    eps = make_episodes([2, 3], range(3), k=4)
    fl = [[0, 1, 2, 0]] * len(eps)
    for mode in ("static", "asked", "fb", "fb_s"):
        items = episode_items(eps, mode, fl if mode == "fb_s" else None)
        out = logits_of(LogicJudge(16, 2, n_types=N_TYPES[mode]), items, torch.device("cpu"))
        assert out.shape == (len(items), 3) and all(hasattr(i, "t") for i in items)
    orc = oracle_items(eps)
    assert sum(len(i.graph.edges) for i in orc) >= sum(len(i.graph.edges) for i in episode_items(eps, "static"))


def test_knowable_means_a_proof_exists_from_what_was_seen():
    from stml.logic import Claim, Form
    from stml.model.logic_episodes import Episode, knowable
    # visible: All 0 are 1, All 1 are 2.  Query "All 0 are 2" follows by chaining -> knowable. "All 2 are 0" is open -> never knowable.
    vis = [Claim(Form.ALL, 0, 1), Claim(Form.ALL, 1, 2)]
    ep = Episode(2, 0, 4, vis, [Claim(Form.ALL, 0, 2), Claim(Form.ALL, 2, 0)], [0, 2], vis)
    assert knowable(ep, 0) and not knowable(ep, 1)
    # a hidden premise: "All 3 are 0" is not visible, so "All 3 are 1" is unknowable until the learner is TOLD something that implies it
    ep2 = Episode(2, 0, 4, vis, [Claim(Form.ALL, 3, 0), Claim(Form.ALL, 3, 1)], [0, 0], vis + [Claim(Form.ALL, 3, 0)])
    assert not knowable(ep2, 0)
    assert knowable(ep2, 1)                      # after being told "All 3 are 0" is true, "All 3 are 1" follows from the visible chain


def test_query_after_asks_matches_the_sequential_graph_and_infogain_counts_new_proofs():
    from stml.model.logic_episodes import newly_knowable, query_after_asks
    e = make_episodes([4], range(1), k=8)[0]
    seq = episode_items([e], "fb")
    # asking claims 0..2 then judging claim 3 is exactly the sequential item at t = 3
    q = query_after_asks(e, [0, 1, 2], 3)
    assert q.label == seq[3].label and q.t == 3
    assert len(q.graph.edges) == len(seq[3].graph.edges) and (q.graph.edges == seq[3].graph.edges).all()
    # information gain is never negative: revealing a true fact cannot make a claim less provable
    assert all(newly_knowable(e, [0], c, [5, 6, 7]) >= 0 for c in (1, 2, 3, 4))


def test_support_marks_are_cumulative_visible_and_never_show_the_future():
    from stml.model.logic_episodes import support_marks
    eps = make_episodes([3, 4], range(6), k=6)
    for e in eps:
        m = support_marks(e)
        assert len(m) == 6 and m[0] == set()                              # nothing is marked before anything is revealed
        assert all(a <= b for a, b in zip(m, m[1:]))                       # marks only accumulate
        assert all(x < len(e.visible) for s in m for x in s)               # only visible premises can be marked
        r = support_marks(e, randomise=True, seed=1)
        assert [len(a) for a in r][0] == 0 and len(r) == 6
    g = episode_graph(eps[0], 3, "fb_why", marks=support_marks(eps[0]))
    assert g.edges[:, 2].max() < N_TYPES["fb_why"]


def test_keep_mask_limits_memory():
    from stml.model.logic_episodes import episode_graph, make_episodes
    ep = make_episodes([2], [0], k=6, vis_frac=0.5)[0]
    full = episode_graph(ep, 5, "fb")
    none = episode_graph(ep, 5, "fb", keep=[False] * 5)
    some = episode_graph(ep, 5, "fb", keep=[True, False, False, False, False])
    stat = episode_graph(ep, 5, "static")
    assert len(none.edges) == len(stat.edges)
    assert len(stat.edges) < len(some.edges) < len(full.edges) or len(some.edges) <= len(full.edges)
    assert len(some.edges) == len(stat.edges) + 2


def test_noisy_feedback_changes_shown_not_labels():
    from stml.model.logic_episodes import episode_graph, make_episodes, noisy_feedback
    eps = make_episodes([2, 3], range(20), k=8, vis_frac=0.5)
    nz = noisy_feedback(eps, 0.5, seed=1)
    flips = sum(a != b for e, n in zip(eps, nz) for a, b in zip(e.labels, n.shown))
    tot = sum(len(e.labels) for e in eps)
    assert [e.labels for e in eps] == [n.labels for n in nz]
    assert 0.35 < flips / tot < 0.65
    assert all(a == b for e, n in zip(eps, noisy_feedback(eps, 0.0)) for a, b in zip(e.labels, n.shown))
    g_clean, g_noisy = episode_graph(eps[0], 7, "fb"), episode_graph(nz[0], 7, "fb")
    assert len(g_clean.edges) == len(g_noisy.edges)


def test_query_after_asks_keeps_noisy_verdicts():
    from stml.model.logic_episodes import make_episodes, noisy_feedback, query_after_asks
    ep = noisy_feedback(make_episodes([3], [0], k=8, vis_frac=0.5), 1.0, seed=0)[0]      # every shown verdict is wrong
    clean = query_after_asks(make_episodes([3], [0], k=8, vis_frac=0.5)[0], [0, 1], 5)
    noisy = query_after_asks(ep, [0, 1], 5)
    assert noisy.label == clean.label
    assert (noisy.graph.edges != clean.graph.edges).any()

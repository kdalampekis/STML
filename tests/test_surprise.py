"""Tests for surprise-weighted training."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stml.model.logic_data as ld  # noqa: E402
from stml.model.logic_data import make_items  # noqa: E402
from stml.model.logic_judge import LogicJudge, logits_of  # noqa: E402
from stml.model.surprise import MODES, fit_weighted  # noqa: E402


@pytest.fixture(autouse=True)
def _flag():
    old = ld.SIZE_FREE
    yield
    ld.SIZE_FREE = old


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_trains_and_keeps_outputs_finite(mode):
    items = make_items([2], range(12), per_world=8)
    train, val = items[:60], items[60:]
    torch.manual_seed(0)
    m = LogicJudge(16, 2)
    fit_weighted(m, train, val, torch.device("cpu"), mode=mode, epochs=2, batch=16)
    assert torch.isfinite(logits_of(m, val, torch.device("cpu"))).all()


def test_plain_mode_matches_unweighted_training_given_the_same_seed():
    items = make_items([2], range(12), per_world=8)
    train, val = items[:60], items[60:]
    out = []
    for _ in range(2):
        torch.manual_seed(1)
        m = LogicJudge(16, 2)
        fit_weighted(m, train, val, torch.device("cpu"), mode="plain", epochs=2, batch=16, seed=3)
        out.append(logits_of(m, val, torch.device("cpu")))
    assert torch.allclose(out[0], out[1])


def test_unknown_mode_is_rejected():
    with pytest.raises(AssertionError):
        fit_weighted(LogicJudge(16, 2), [], [], torch.device("cpu"), mode="nonsense")

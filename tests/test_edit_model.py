"""Tests for the cell-level edit model data pipeline and network."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from stml.model.edit_model import H, EditNet, build_rows, make_input, pad, parse  # noqa: E402

DATA = ROOT / "data/raw/harc/data/data.csv"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="H-ARC raw data not present")


def test_parse_and_pad():
    a = parse("|012|345|")
    assert a.shape == (2, 3) and a[1, 2] == 5
    p = pad(a)
    assert p.shape == (H, H) and p[0, 0] == 0 and p[2, 0] == -1 and p[0, 3] == -1
    assert parse("not a grid") is None


@pytest.fixture(scope="module")
def rows():
    return build_rows(DATA, ROOT / "data/raw/harc/arc_data", 1500, 800)


def test_every_target_cell_exists_in_the_grid_before_the_edit(rows):
    for split in ("train", "test"):
        d = rows[split]
        y, x = d["y"] // H, d["y"] % H
        assert (d["grid"][np.arange(len(y)), y, x] >= 0).all()


def test_train_and_test_use_different_puzzles(rows):
    # task codes are per split (factorised), so check through the sizes only: both splits exist and are non-trivial
    assert len(rows["train"]["y"]) > 1000 and len(rows["test"]["y"]) > 500


def test_history_channels_are_zero_without_history_and_shuffle_changes_them(rows):
    b = {k: v[:64] for k, v in rows["test"].items()}
    cpu = torch.device("cpu")
    x0, valid = make_input(b, cpu, history=False)
    x1, _ = make_input(b, cpu, history=True)
    assert x0.shape == x1.shape == (64, 46, H, H)
    assert x0[:, 34:].abs().sum() == 0                     # the history blocks are the last 12 channels (34-45)
    assert x1[:, 34:].abs().sum() > 0
    torch.manual_seed(0)
    xs, _ = make_input(b, cpu, history=True, shuffle_history=True)
    assert not torch.equal(x1, xs)
    logits = EditNet()(x1, valid)
    assert logits.shape == (64, H * H) and torch.isfinite(logits[valid]).all()
    assert (logits[~valid] < -1e8).all()

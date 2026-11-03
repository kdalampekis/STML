"""The 12 hand-made candidate features of baseline_v1 (shape rule, palette, density, symmetry, ...),
attached to STML puzzles as an optional extra input. Same candidate order (sorted pool) as baseline_v1.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N_HAND = 12


def _load_cm():
    if "baseline_choice_model" in sys.modules:
        return sys.modules["baseline_choice_model"]
    spec = importlib.util.spec_from_file_location("baseline_choice_model", ROOT / "experiments/baseline_v1/choice_model.py")
    cm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cm
    spec.loader.exec_module(cm)
    return cm


def hand_for_grids(train, test_input, target, candidates) -> "np.ndarray":
    """Raw (unstandardised) hand features [n_candidates, 12] for a puzzle given as grid strings, e.g. an augmented copy."""
    cm = _load_cm()
    p = cm.Puzzle("aug", [(a, b) for a, b in train], test_input, target, list(candidates))     # grid strings; the baseline code parses them itself
    p.grids = [cm._arr(c) for c in candidates]
    return cm.static_features(p)


def attach_hand(puzzles: dict, arc_dir, records: list[dict]) -> None:
    cm = _load_cm()
    base = cm.attach_candidates(cm.load_puzzles(str(arc_dir)), records)
    for name, p in puzzles.items():
        assert base[name].candidates == p.candidates, name
        p.hand = base[name].feats

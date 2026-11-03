"""ARCTraj as extra training data for the STML learner. Only ever used on TRAINING puzzles.

Two uses:
  steps      each ARCTraj session is a first-attempt human choice (the answer the person
             submitted, right/wrong recomputed). Kept only if that answer is in the puzzle's
             H-ARC candidate pool, so the test protocol and the pools do not change.
  progress   ARCTraj records the full grid after every action. A few intermediate grids per
             session, each labelled with the share of cells that are wrong against the target,
             teach the encoder what "close to right" looks like (auxiliary regression).
"""

from __future__ import annotations

import csv
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

from .graphs import grid_graph, parse_grid
from .harc_data import Puzzle, Step


def arctraj_steps(records: list[dict], puzzles: dict[str, Puzzle], tasks: set[str]) -> list[Step]:
    steps = []
    index = {}
    for r in records:
        name = r["task"]["id"]
        if name not in tasks or name not in puzzles:
            continue
        idx = index.setdefault(name, {c: i for i, c in enumerate(puzzles[name].candidates)})
        i = idx.get(r["attempts"][0]["commit"]["answer"])
        if i is not None:
            steps.append(Step(name, [], i, bool(r["attempts"][0]["feedback"]["correct"])))
    return steps


def frac_wrong(grid: np.ndarray, target: np.ndarray) -> float:
    return 1.0 if grid.shape != target.shape else float((grid != target).mean())


def load_progress_grids(csv_path: str | Path, puzzles: dict[str, Puzzle], per_session: int = 4,
                        seed: int = 0, cache: str | Path | None = None) -> dict[str, list[tuple[str, float]]]:
    """task -> [(grid string, share of cells wrong)]; sampled once, cached."""
    cache = Path(cache) if cache else None
    if cache and cache.exists():
        return pickle.loads(cache.read_bytes())
    csv.field_size_limit(sys.maxsize)
    rnd = random.Random(seed)
    targets = {n: parse_grid(p.target) for n, p in puzzles.items()}
    out: dict[str, list] = {}
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        name = r["taskId"] + ".json"
        if name not in targets:
            continue
        try:
            acts = json.loads(r["actionSequence"])
        except json.JSONDecodeError:
            continue
        end = next((i for i, a in enumerate(acts) if a.get("operation") == "Submit"), None)
        if end is None:
            continue
        pick = {end} | set(rnd.sample(range(end), min(per_session - 1, end)))
        seen = set()
        for i in sorted(pick):
            g = np.array(acts[i]["grid"], dtype=np.int8)
            s = "|" + "|".join("".join(str(int(c)) for c in row) for row in g) + "|"
            if s in seen or g.ndim != 2 or g.size == 0:
                continue
            seen.add(s)
            out.setdefault(name, []).append((s, frac_wrong(g, targets[name])))
    if cache:
        cache.write_bytes(pickle.dumps(out))
    return out


def attach_progress(puzzles: dict[str, Puzzle], grids: dict[str, list], cache: str | Path | None = None, log=print) -> None:
    """Build graphs for the progress grids (cached: it takes a few minutes)."""
    cache = Path(cache) if cache else None
    if cache and cache.exists():
        stored = pickle.loads(cache.read_bytes())
    else:
        stored = {}
        for k, (name, items) in enumerate(sorted(grids.items())):
            stored[name] = ([grid_graph(s) for s, _ in items], np.array([y for _, y in items], dtype=np.float32))
            if (k + 1) % 100 == 0:
                log(f"progress graphs built for {k + 1}/{len(grids)} puzzles")
        if cache:
            cache.write_bytes(pickle.dumps(stored))
    for name, (gs, ys) in stored.items():
        if name in puzzles:
            puzzles[name].aux_graphs, puzzles[name].aux_y = gs, ys

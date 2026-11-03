"""H-ARC as training data for the STML learner.

Setting (the same protocol as the first experiment, so results are comparable):
for each puzzle the learner chooses among the answers people actually gave,
with up to 3 attempts and right/wrong feedback. The difference from the first
experiment is that the learner now reads the real grids through its own graph
encoder instead of a dozen hand-made features.

A training step = (puzzle, what was tried so far with its feedback, the choice
made next, whether that choice was right).
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .graphs import grid_graph, parse_grid


@dataclass
class Puzzle:
    name: str
    train: list[tuple[str, str]]
    test_input: str
    target: str
    candidates: list[str] = field(default_factory=list)
    graph_train: list = field(default_factory=list)
    graph_test_in: object = None
    graph_cands: list = field(default_factory=list)
    hand: object = None                              # optional hand-made candidate features [n_cands, k]
    aux_graphs: list = field(default_factory=list)   # ARCTraj intermediate grids (training puzzles only)
    aux_y: object = None                             # share of cells wrong for each
    aux_active: list = field(default_factory=list)   # indices sampled for the current batch

    @property
    def target_index(self) -> int:
        return self.candidates.index(self.target)


@dataclass
class Step:
    puzzle: str
    tried: list[tuple[int, bool]]     # (candidate index, was it right) for earlier attempts
    choice: int
    correct: bool


def _grid_str(s) -> str | None:
    return s if parse_grid(s) is not None else None


def load_puzzles(arc_dir: str | Path) -> dict[str, Puzzle]:
    t = pd.concat(pd.read_csv(f) for f in sorted(Path(arc_dir).glob("ARC_*_tasks.csv")))
    out = {}
    for name, g in t.groupby("task_name"):
        tr = g[g.example_type == "train"].sort_values("example_number")
        te = g[g.example_type == "test"].sort_values("example_number").iloc[0]
        out[name] = Puzzle(name, list(zip(tr.input_grid, tr.output_grid)), te.input_grid, te.output_grid)
    return out


def read_records(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def attach_pools(puzzles: dict[str, Puzzle], records: list[dict]) -> dict[str, Puzzle]:
    """Candidates = every distinct answer people submitted; keep puzzles whose correct answer is among them."""
    pool: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        for a in r["attempts"]:
            g = _grid_str(a["commit"]["answer"])
            if g:
                pool[r["task"]["id"]][g] += 1
    keep = {}
    for name, p in puzzles.items():
        if name in pool and p.target in pool[name]:
            p.candidates = sorted(pool[name])
            keep[name] = p
    return keep


def build_graphs(puzzles: dict[str, Puzzle], cache: str | Path | None = None, log=print) -> None:
    """Object graphs for every grid; cached because building them takes a minute or two."""
    key = hashlib.sha1("".join(sorted(f"{n}:{len(p.candidates)}" for n, p in puzzles.items())).encode()).hexdigest()[:12]
    cache = Path(cache) if cache else None
    if cache and cache.exists():
        stored = pickle.loads(cache.read_bytes())
        if stored.get("key") == key:
            for n, p in puzzles.items():
                p.graph_train, p.graph_test_in, p.graph_cands = stored["graphs"][n]
            log(f"graphs loaded from cache {cache}")
            return
    graphs = {}
    for k, (n, p) in enumerate(sorted(puzzles.items())):
        p.graph_train = [(grid_graph(a), grid_graph(b)) for a, b in p.train]
        p.graph_test_in = grid_graph(p.test_input)
        p.graph_cands = [grid_graph(c) for c in p.candidates]
        graphs[n] = (p.graph_train, p.graph_test_in, p.graph_cands)
        if (k + 1) % 200 == 0:
            log(f"graphs built for {k + 1}/{len(puzzles)} puzzles")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps({"key": key, "graphs": graphs}))
        log(f"graphs cached to {cache}")


def human_steps(records: list[dict], puzzles: dict[str, Puzzle], tasks: set[str]) -> list[Step]:
    """People's attempt sequences, mistakes and revisions included."""
    steps = []
    for r in records:
        name = r["task"]["id"]
        if name not in tasks or name not in puzzles:
            continue
        index = {c: i for i, c in enumerate(puzzles[name].candidates)}
        tried: list[tuple[int, bool]] = []
        for a in r["attempts"]:
            i = index.get(a["commit"]["answer"])
            if i is None or any(i == j for j, _ in tried):
                break
            ok = bool(a["feedback"]["correct"])
            steps.append(Step(name, list(tried), i, ok))
            if ok:
                break
            tried.append((i, False))
    return steps


def answer_steps(puzzles: dict[str, Puzzle], tasks: set[str], budget: int) -> list[Step]:
    """Imitate answers: the correct answer from a fresh start, spread over `budget` steps."""
    names = sorted(tasks)
    return [Step(n, [], puzzles[n].target_index, True) for k in range(budget) for n in [names[k % len(names)]]]

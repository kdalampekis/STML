"""A first, rough test of H1 on H-ARC.

Question: for a learner that must solve new puzzles with up to 3 attempts and
only right/wrong feedback, is it better to imitate
  ANSWERS  the correct answers (what people answer),
  HUMAN    people's attempt sequences, mistakes and revisions included (how people learn), or
  RL       the learning history of a reinforcement-learning agent (Algorithm Distillation's source)?

The setting is narrowed so that the comparison is clean:
  - For each puzzle the learner chooses among the answers people actually gave
    (median 8 per puzzle; the correct one is present in 787 of 800 puzzles).
  - It sees each candidate only through hand-made features computed from the
    puzzle's examples (shape rule, colours, symmetry, similarity to the input),
    plus features of what it already tried and got wrong.
  - Every condition uses the same model (a conditional logit over candidates)
    and the same amount of training data; only the source of the data differs.
So this tests the choice-and-revision strategy, not visual perception.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stml.datasets.harc import parse_grid


# ----------------------------------------------------------------------------- puzzles

@dataclass
class Puzzle:
    name: str
    train: list[tuple[list[str], list[str]]]
    test_input: list[str]
    target: str
    candidates: list[str] = field(default_factory=list)
    feats: np.ndarray | None = None          # static features, one row per candidate
    grids: list[np.ndarray] = field(default_factory=list)


def _arr(g) -> np.ndarray:
    rows = g if isinstance(g, list) else parse_grid(g)
    return np.array([[int(c) for c in r] for r in rows], dtype=np.int8)


def load_puzzles(arc_dir: str | Path) -> dict[str, Puzzle]:
    t = pd.concat(pd.read_csv(f) for f in sorted(Path(arc_dir).glob("ARC_*_tasks.csv")))
    out = {}
    for name, g in t.groupby("task_name"):
        tr = g[g.example_type == "train"].sort_values("example_number")
        te = g[g.example_type == "test"].sort_values("example_number").iloc[0]
        out[name] = Puzzle(name, [(parse_grid(a), parse_grid(b)) for a, b in zip(tr.input_grid, tr.output_grid)],
                           parse_grid(te.input_grid), te.output_grid)
    return out


# ----------------------------------------------------------------------------- features

STATIC = ["shape_ok", "shape_unknown", "same_shape_input", "copy_input", "sim_input", "palette_ok",
          "n_new_colors", "ncolor_gap", "bg_gap", "density_gap", "symh_ok", "symv_ok"]
HISTORY = ["sim_failed", "same_shape_failed", "n_failed"]
FEATURES = STATIC + HISTORY


def _shape_rule(p: Puzzle) -> tuple[int, int] | None:
    ins = [_arr(a).shape for a, _ in p.train]
    outs = [_arr(b).shape for _, b in p.train]
    ti = _arr(p.test_input).shape
    if all(i == o for i, o in zip(ins, outs)):
        return ti
    if len(set(outs)) == 1:
        return outs[0]
    ratios = {(o[0] / i[0], o[1] / i[1]) for i, o in zip(ins, outs)}
    if len(ratios) == 1:
        ry, rx = ratios.pop()
        y, x = ti[0] * ry, ti[1] * rx
        if float(y).is_integer() and float(x).is_integer():
            return int(y), int(x)
    return None


def _sym(a: np.ndarray) -> tuple[bool, bool]:
    return bool((a == a[:, ::-1]).all()), bool((a == a[::-1, :]).all())


def static_features(p: Puzzle) -> np.ndarray:
    outs = [_arr(b) for _, b in p.train]
    ti = _arr(p.test_input)
    rule = _shape_rule(p)
    allowed = set(np.unique(ti)) | {c for o in outs for c in np.unique(o)}
    ncol = np.mean([len(np.unique(o)) for o in outs])
    bg = np.mean([np.bincount(o.ravel()).max() / o.size for o in outs])
    dens = np.mean([(o != 0).mean() for o in outs])
    symh_all = all(_sym(o)[0] for o in outs)
    symv_all = all(_sym(o)[1] for o in outs)
    rows = []
    for g in p.grids:
        same_in = g.shape == ti.shape
        cols = set(np.unique(g))
        sh, sv = _sym(g)
        rows.append([
            float(rule is not None and g.shape == rule),
            float(rule is None),
            float(same_in),
            float(same_in and (g == ti).all()),
            float((g == ti).mean()) if same_in else 0.0,
            float(cols <= allowed),
            float(len(cols - allowed)),
            abs(len(cols) - ncol),
            abs(np.bincount(g.ravel()).max() / g.size - bg),
            abs((g != 0).mean() - dens),
            float(sh) if symh_all else 0.5,
            float(sv) if symv_all else 0.5,
        ])
    return np.array(rows, dtype=np.float32)


def features(p: Puzzle, failed: list[int]) -> np.ndarray:
    """All features for every candidate, given which candidates already failed."""
    hist = np.zeros((len(p.candidates), len(HISTORY)), dtype=np.float32)
    if failed:
        for i, g in enumerate(p.grids):
            sims = [float((g == p.grids[j]).mean()) if g.shape == p.grids[j].shape else 0.0 for j in failed]
            hist[i] = [max(sims), float(g.shape == p.grids[failed[-1]].shape), float(len(failed))]
    return np.concatenate([p.feats, hist], axis=1)


def attach_candidates(puzzles: dict[str, Puzzle], records: list[dict]) -> dict[str, Puzzle]:
    pool: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        for a in r["attempts"]:
            if parse_grid(a["commit"]["answer"]) is not None:
                pool[r["task"]["id"]][a["commit"]["answer"]] += 1
    keep = {}
    for name, p in puzzles.items():
        if name not in pool or p.target not in pool[name]:
            continue  # the correct answer must be choosable (787 of 800 puzzles)
        p.candidates = sorted(pool[name])
        p.grids = [_arr(c) for c in p.candidates]
        p.feats = static_features(p)
        keep[name] = p
    return keep


# ----------------------------------------------------------------------------- the model

def _scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and spread for standardizing. A feature that never varies in the data
    (e.g. history features measured only at fresh starts) keeps unit scale instead of
    being divided by ~0, which would blow its weight up (a bug found in the first run)."""
    mu, sd = X.mean(0), X.std(0)
    return mu, np.where(sd < 1e-3, 1.0, sd)


class ChoiceModel:
    """P(choose candidate i) proportional to exp(w . x_i), over candidates not yet tried."""

    def __init__(self, n_features: int, l2: float = 1e-2):
        self.w = np.zeros(n_features, dtype=np.float64)
        self.mu = np.zeros(n_features)
        self.sd = np.ones(n_features)
        self.l2 = l2

    def standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def probs(self, X: np.ndarray, allowed: np.ndarray) -> np.ndarray:
        z = self.standardize(X) @ self.w
        z = np.where(allowed, z, -np.inf)
        z = z - z[allowed].max()
        e = np.exp(z) * allowed
        return e / e.sum()

    def fit(self, steps: list[tuple[np.ndarray, np.ndarray, int]], epochs: int = 300, lr: float = 0.5,
            freeze: list[int] | None = None) -> "ChoiceModel":
        """steps: (features, allowed mask, chosen index). Maximum likelihood, full batch."""
        allX = np.concatenate([X[m] for X, m, _ in steps])
        self.mu, self.sd = _scale(allX)
        Z = [(self.standardize(X), m, c) for X, m, c in steps]
        for _ in range(epochs):
            g = np.zeros_like(self.w)
            for Xs, m, c in Z:
                z = np.where(m, Xs @ self.w, -np.inf)
                z -= z[m].max()
                p = np.exp(z) * m
                p /= p.sum()
                g += Xs[c] - p @ Xs
            g = g / len(Z) - self.l2 * self.w
            if freeze:
                g[freeze] = 0.0
            self.w += lr * g
        return self


def play(model: ChoiceModel, p: Puzzle, rng: np.random.Generator | None = None, attempts: int = 3) -> tuple[bool, int]:
    """Greedy (or sampled) play with right/wrong feedback. Returns (solved, attempts used)."""
    failed: list[int] = []
    allowed = np.ones(len(p.candidates), dtype=bool)
    for k in range(min(attempts, len(p.candidates))):
        pr = model.probs(features(p, failed), allowed)
        i = int(rng.choice(len(pr), p=pr)) if rng is not None else int(np.argmax(pr))
        if p.candidates[i] == p.target:
            return True, k + 1
        failed.append(i)
        allowed[i] = False
    return False, attempts


# ----------------------------------------------------------------------------- training data

def human_steps(records: list[dict], puzzles: dict[str, Puzzle], tasks: set[str]) -> list[tuple]:
    steps = []
    for r in records:
        name = r["task"]["id"]
        if name not in tasks or name not in puzzles:
            continue
        p = puzzles[name]
        index = {c: i for i, c in enumerate(p.candidates)}
        failed: list[int] = []
        allowed = np.ones(len(p.candidates), dtype=bool)
        for a in r["attempts"]:
            i = index.get(a["commit"]["answer"])
            if i is None or not allowed[i]:
                break
            steps.append((features(p, failed), allowed.copy(), i))
            if a["feedback"]["correct"]:
                break
            failed.append(i)
            allowed[i] = False
    return steps


def answer_steps(puzzles: dict[str, Puzzle], tasks: set[str], copies: int) -> list[tuple]:
    """Imitate answers: the correct candidate, from a fresh start, `copies` times per puzzle."""
    steps = []
    for name in tasks:
        p = puzzles[name]
        i = p.candidates.index(p.target)
        X, m = features(p, []), np.ones(len(p.candidates), dtype=bool)
        steps += [(X, m, i)] * copies
    return steps


def rl_history(puzzles: dict[str, Puzzle], tasks: list[str], episodes: int, seed: int,
               lr: float = 0.05, return_agent: bool = False):
    """REINFORCE on the same model class, learning from scratch with right/wrong feedback.
    Returns every choice it made while learning: an Algorithm Distillation source."""
    rng = np.random.default_rng(seed)
    allX = np.concatenate([features(puzzles[t], []) for t in tasks])
    agent = ChoiceModel(len(FEATURES))
    agent.mu, agent.sd = _scale(allX)
    history, baseline = [], 0.0
    for ep in range(episodes):
        p = puzzles[tasks[rng.integers(len(tasks))]]
        failed, allowed, traj = [], np.ones(len(p.candidates), dtype=bool), []
        reward = 0.0
        for k in range(min(3, len(p.candidates))):
            X = features(p, failed)
            pr = agent.probs(X, allowed)
            i = int(rng.choice(len(pr), p=pr))
            traj.append((X, allowed.copy(), i, pr))
            history.append((X, allowed.copy(), i))
            if p.candidates[i] == p.target:
                reward = 1.0 - 0.25 * k
                break
            failed.append(i)
            allowed[i] = False
        adv = reward - baseline
        baseline += 0.02 * (reward - baseline)
        for X, m, i, pr in traj:
            Xs = agent.standardize(X)
            agent.w += lr * adv * (Xs[i] - pr @ Xs)
    return (agent, history) if return_agent else history


# ----------------------------------------------------------------------------- evaluation

def evaluate(model: ChoiceModel, puzzles: dict[str, Puzzle], tasks: list[str]) -> dict:
    res = [play(model, puzzles[t]) for t in tasks]
    solved = np.array([s for s, _ in res])
    first = np.array([s and k == 1 for s, k in res])
    failed_first = ~first
    recover = solved[failed_first].mean() if failed_first.any() else float("nan")
    return {"solved": float(solved.mean()), "first": float(first.mean()), "recover": float(recover),
            "_solved": solved, "_first": first}


def human_fit(model: ChoiceModel | None, steps: list[tuple]) -> dict:
    """Mean log-likelihood of the choices real people actually made (higher = more human-like),
    on all steps and on revision steps only (after at least one wrong answer). Log-likelihood
    is a proper score: unlike mean probability, it punishes a confident model when it is wrong."""
    allp, rev = [], []
    for X, m, c in steps:
        pr = model.probs(X, m)[c] if model is not None else 1.0 / m.sum()
        lp = float(np.log(max(pr, 1e-12)))
        allp.append(lp)
        if X[c, FEATURES.index("n_failed")] > 0:
            rev.append(lp)
    return {"fit_all": float(np.mean(allp)), "fit_revision": float(np.mean(rev)) if rev else float("nan")}


def random_baseline(puzzles: dict[str, Puzzle], tasks: list[str]) -> dict:
    s, f = [], []
    for t in tasks:
        n = len(puzzles[t].candidates)
        s.append(min(3, n) / n)
        f.append(1 / n)
    s, f = np.array(s), np.array(f)
    return {"solved": float(s.mean()), "first": float(f.mean()), "recover": float(((s - f) / (1 - f + 1e-9)).mean())}


def human_reference(records: list[dict], tasks: set[str]) -> dict:
    rs = [r for r in records if r["task"]["id"] in tasks]
    by = defaultdict(list)
    for r in rs:
        by[r["task"]["id"]].append((r["outcome"]["solved"], r["attempts"][0]["feedback"]["correct"]))
    solved = np.mean([np.mean([s for s, _ in v]) for v in by.values()])
    first = np.mean([np.mean([f for _, f in v]) for v in by.values()])
    rec = [s for v in by.values() for s, f in v if not f]
    return {"solved": float(solved), "first": float(first), "recover": float(np.mean(rec))}


def run(records: list[dict], puzzles: dict[str, Puzzle], seeds: int = 5, out: Path | None = None,
        rl_restarts: int = 5) -> dict:
    names = sorted(puzzles)
    results: dict[str, list[dict]] = defaultdict(list)
    weights: dict[str, list[np.ndarray]] = defaultdict(list)
    for seed in range(seeds):
        rnd = random.Random(seed)
        shuffled = names[:]
        rnd.shuffle(shuffled)
        train, test = set(shuffled[: len(names) // 2]), shuffled[len(names) // 2:]
        h = human_steps(records, puzzles, train)
        h_test = human_steps(records, puzzles, set(test))
        budget = len(h)
        hist_idx = [FEATURES.index(f) for f in HISTORY]
        models = {
            "Imitate answers": ChoiceModel(len(FEATURES)).fit(
                answer_steps(puzzles, train, copies=max(1, budget // len(train))), freeze=hist_idx),
            "Imitate human learning": ChoiceModel(len(FEATURES)).fit(h),
        }
        # REINFORCE sometimes settles on a poor strategy; as is standard, train it several
        # times and keep the run that does best on the TRAINING puzzles (never the test ones).
        best = None
        for k in range(rl_restarts):
            agent, hist = rl_history(puzzles, sorted(train), episodes=6000, seed=100 * seed + k, return_agent=True)
            score = evaluate(agent, puzzles, sorted(train))["solved"]
            if best is None or score > best[0]:
                best = (score, agent, hist)
        rl_train_score, rl_agent, rl = best
        r_agent = evaluate(rl_agent, puzzles, test)
        r_agent.update(human_fit(rl_agent, h_test))
        results["RL agent itself (best of restarts)"].append(r_agent)
        rng = np.random.default_rng(seed)
        take = rng.choice(len(rl), size=min(budget, len(rl)), replace=False)
        models["Imitate RL learning history"] = ChoiceModel(len(FEATURES)).fit([rl[i] for i in take])
        late = rl[-budget:]
        models["Imitate converged RL agent"] = ChoiceModel(len(FEATURES)).fit(late)
        for name, m in models.items():
            r = evaluate(m, puzzles, test)
            r.update(human_fit(m, h_test))
            results[name].append(r)
            weights[name].append(m.w.copy())
        rb = random_baseline(puzzles, test)
        rb.update(human_fit(None, h_test))
        results["Random choice"].append(rb)
        results["People (reference)"].append(human_reference(records, set(test)))
        results["_n"].append({"train": len(train), "test": len(test), "human_steps": budget})

    summary = {}
    for name, rs in results.items():
        if name == "_n":
            continue
        summary[name] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
                         for k in ("solved", "first", "recover", "fit_all", "fit_revision") if k in rs[0]}
    # paired comparison on the same test puzzles: human-learning vs each machine source
    paired = {}
    for other in ("Imitate answers", "Imitate RL learning history", "Imitate converged RL agent"):
        diffs = [a["solved"] - b["solved"] for a, b in zip(results["Imitate human learning"], results[other])]
        boot = []
        rng = np.random.default_rng(0)
        for _ in range(2000):
            tot = []
            for a, b in zip(results["Imitate human learning"], results[other]):
                idx = rng.integers(len(a["_solved"]), size=len(a["_solved"]))
                tot.append(a["_solved"][idx].mean() - b["_solved"][idx].mean())
            boot.append(np.mean(tot))
        paired[other] = {"diff": float(np.mean(diffs)), "ci95": [float(np.percentile(boot, 2.5)),
                                                                 float(np.percentile(boot, 97.5))]}
    w_h = np.mean(weights["Imitate human learning"], axis=0)
    w_a = np.mean(weights["Imitate answers"], axis=0)
    per_seed = {name: [round(r["solved"], 4) for r in rs] for name, rs in results.items() if name != "_n"}
    out_obj = {"summary": summary, "paired_vs_human_learning": paired, "splits": results["_n"],
               "solved_per_split": per_seed,
               "features": FEATURES,
               "weights": {"Imitate human learning": dict(zip(FEATURES, map(float, w_h))),
                           "Imitate answers": dict(zip(FEATURES, map(float, w_a)))}}
    if out:
        out.write_text(json.dumps(out_obj, indent=1))
    return out_obj

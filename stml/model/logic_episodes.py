"""Logic worlds with HIDDEN premises and feedback, for testing in-context learning from feedback (exact labels).

An episode is one world (levels 2-5 of the game). The learner sees only a random subset of the premises (`vis_frac`), then judges K
claims one after another. After each judgement the TRUE verdict (follows / contradicted / open, computed from ALL premises) is revealed.
The learner is a graph model: every revealed claim is added to the world graph as an extra edge, so the next judgement can chain through it.

Edge-type vocabularies (premises always use the 8 types of logic_data.EDGE_TYPES):
  static  nothing revealed (the learner never sees feedback)
  asked   the claim was asked, verdict not shown                  form x direction                       (8)
  fb      verdict shown                                             form x verdict x direction            (24)
  bin     BINARY feedback: only whether the history-free judge's guess for the claim was right (form x right/wrong x direction, 16)
  fb_n    like fb (24 edge types), plus the surprise flavours as NODE features: for each kind, the number of incident revealed claims of each
          flavour (3 values). Same information as fb_s without splitting the edge types.
  fb_s    verdict shown + SIGNED SURPRISE flavour of the history-free judge:
            0 expected      the static judge already gave the true verdict probability >= 0.5
            1 underestimated it thought "open" but the claim follows / is contradicted (hidden structure revealed)
            2 overestimated  it thought the claim was derivable but the verdict differs
          form x verdict x flavour x direction                  (72)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..logic import FORM_ORDER, SYMMETRIC_FORMS, Claim, all_claims
from ..world import generate_world, level_config
from . import logic_data as ld
from .graphs import Graph
from .logic_data import EDGE_TYPES, NODE_DIM, Item

_VERDICT = {"entailed": 0, "contradicted": 1, "undetermined": 2}
BASE = len(EDGE_TYPES)                        # 8
N_TYPES = {"static": BASE, "asked": BASE + 8, "fb": BASE + 24, "fb_s": BASE + 72, "fb_n": BASE + 24, "bin": BASE + 16, "fb_why": BASE + 32}
NODE_DIM_N = NODE_DIM + 3                       # fb_n: + per-kind counts of incident revealed claims of each surprise flavour


@dataclass
class Episode:
    level: int
    seed: int
    n: int
    visible: list[Claim]
    queries: list[Claim]
    labels: list[int]
    full: list[Claim] | None = None            # all premises (for the oracle condition only)
    shown: list[int] | None = None             # verdicts REVEALED to the learner if they differ from the truth (noisy feedback); labels stay the truth


def make_episodes(levels, seeds, k: int = 10, vis_frac: float = 0.5, rng_seed: int = 0) -> list[Episode]:
    rng = random.Random(rng_seed)
    out = []
    for level in levels:
        cfg = level_config(level)
        for seed in seeds:
            w = generate_world(cfg, seed)
            prem = list(w.premises)
            vis = [p for p in prem if rng.random() < vis_frac]
            if not vis:
                vis = [rng.choice(prem)]
            if len(vis) == len(prem) and len(prem) > 1:
                vis.remove(rng.choice(vis))
            seen, pool = set(), []
            for c in all_claims(w.n, w.claim_forms):
                key = c.canonical()
                if c.a == c.b or key in w.premise_set or key in seen:
                    continue
                seen.add(key)
                pool.append(c)
            qs = rng.sample(pool, min(k, len(pool)))
            out.append(Episode(level, seed, w.n, vis, qs, [_VERDICT[w.semantics.evaluate(prem, c).value] for c in qs], prem))
    return out


def _edges(claim: Claim, base: int, direction_cnt: int = 2):
    f = FORM_ORDER.index(claim.form)
    return f, (claim.a, claim.b), claim.form in SYMMETRIC_FORMS


def episode_graph(ep: Episode, t: int, mode: str, flavours: list[int] | None = None, marks: list | None = None, keep: list[bool] | None = None) -> Graph:
    """Graph for query t: visible premises + the t revealed earlier claims (per `mode`; only those with keep[s] if `keep` is given: a limited memory) + the query flags."""
    n = ep.n
    edges, deg_in, deg_out = [], np.zeros(n), np.zeros(n)
    used = marks[t] if (mode == "fb_why" and marks is not None) else ()
    for pi, p in enumerate(ep.visible):
        f = FORM_ORDER.index(p.form)
        off = BASE + 24 if pi in used else 0                    # fb_why: this visible premise was used in a proof revealed so far
        edges.append((p.a, p.b, off + 2 * f))
        edges.append((p.b, p.a, off + (2 * f if p.form in SYMMETRIC_FORMS else 2 * f + 1)))
        deg_out[p.a] += 1
        deg_in[p.b] += 1
    if mode != "static":
        for s in range(t):
            if keep is not None and not keep[s]:
                continue
            c, y = ep.queries[s], (ep.shown if ep.shown is not None else ep.labels)[s]
            f = FORM_ORDER.index(c.form)
            if mode == "asked":
                idx = f * 2
            elif mode in ("fb", "fb_n", "fb_why"):
                idx = (f * 3 + y) * 2
            elif mode == "bin":                                 # binary feedback: was the history-free judge's guess right? (flavours[s] in {0, 1})
                idx = (f * 2 + flavours[s]) * 2
            else:
                idx = ((f * 3 + y) * 3 + flavours[s]) * 2
            sym = c.form in SYMMETRIC_FORMS
            edges.append((c.a, c.b, BASE + idx))
            edges.append((c.b, c.a, BASE + idx + (0 if sym else 1)))
    q = ep.queries[t]
    x = np.zeros((n, NODE_DIM), dtype=np.float32)
    if ld.SIZE_FREE:
        x[:, 0], x[:, 1], x[:, 2], x[:, 3] = 1.0, np.minimum(deg_in, 4) / 4, np.minimum(deg_out, 4) / 4, 0.0
    else:
        x[:, 0], x[:, 1], x[:, 2], x[:, 3] = 1.0, deg_in / max(1, n), deg_out / max(1, n), n / 12
    x[q.a, 4] = 1.0
    x[q.b, 5] = 1.0
    x[:, 6 + FORM_ORDER.index(q.form)] = 1.0
    if mode == "fb_n":
        cnt = np.zeros((n, 3), dtype=np.float32)
        for s_ in range(t):
            c = ep.queries[s_]
            cnt[c.a, flavours[s_]] += 1
            cnt[c.b, flavours[s_]] += 1
        x = np.concatenate([x, np.minimum(cnt, 4) / 4], 1)
    return Graph(x, np.array(edges, dtype=np.int64).reshape(-1, 3), N_TYPES[mode])


def episode_items(eps: list[Episode], mode: str, flavours: list[list[int]] | None = None, marks: list | None = None, keeps: list[list[bool]] | None = None) -> list[Item]:
    items = []
    for i, ep in enumerate(eps):
        for t in range(len(ep.queries)):
            it = Item(ep.level, ep.seed, ep.n, ep.queries[t], ep.labels[t], 0,
                      episode_graph(ep, t, mode, None if flavours is None else flavours[i], None if marks is None else marks[i], None if keeps is None else keeps[i]), len(ep.visible), (), ())
            it.t = t
            items.append(it)
    return items


def noisy_feedback(eps: list[Episode], p: float, seed: int = 0) -> list[Episode]:
    """Copies of the episodes in which each revealed verdict is replaced, with probability p, by one of the two other verdicts (chosen uniformly)."""
    from dataclasses import replace
    rnd = random.Random(seed)
    out = []
    for ep in eps:
        shown = [y if rnd.random() >= p else rnd.choice([v for v in (0, 1, 2) if v != y]) for y in ep.labels]
        out.append(replace(ep, shown=shown))
    return out


def oracle_items(eps: list[Episode]) -> list[Item]:
    """The same queries judged with ALL premises visible (an upper bound; no feedback needed)."""
    out = []
    for ep in eps:
        full = Episode(ep.level, ep.seed, ep.n, ep.full, ep.queries, ep.labels)
        for t in range(len(ep.queries)):
            it = Item(ep.level, ep.seed, ep.n, ep.queries[t], ep.labels[t], 0, episode_graph(full, t, "static"), len(ep.full), (), ())
            it.t = t
            out.append(it)
    return out


def flavour_of(p0: np.ndarray, y: int) -> int:
    """Signed-surprise flavour of a static prediction p0 (3 probabilities) for true verdict y."""
    if p0[y] >= 0.5:
        return 0
    return 1 if int(p0.argmax()) == 2 else 2


_SEM: dict = {}


def knowable(ep: Episode, t: int) -> bool:
    """Is the verdict of query t derivable (a proof exists) from what the learner has seen: the visible premises plus every earlier
    revealed claim that turned out true (or the negation of one that turned out contradicted)? This is the exact version of
    'the learner could know this now' (H3: knowing = having a certificate). Open verdicts are not certifiable from partial knowledge."""
    from ..logic import Semantics
    sem = _SEM.setdefault(ep.n, Semantics(ep.n))
    known = list(ep.visible)
    for s in range(t):
        c, y = ep.queries[s], ep.labels[s]
        if y == 0:
            known.append(c)
        elif y == 1:
            known.append(c.negation())
    return sem.evaluate(known, ep.queries[t]).value in ("entailed", "contradicted")


def query_after_asks(ep: Episode, asked: list[int], j: int, mode: str = "fb") -> Item:
    """Claim j of the episode judged after the claims with indices `asked` (in that order) were asked and their verdicts revealed."""
    e2 = Episode(ep.level, ep.seed, ep.n, ep.visible, [ep.queries[i] for i in asked] + [ep.queries[j]],
                 [ep.labels[i] for i in asked] + [ep.labels[j]], ep.full,
                 None if ep.shown is None else [ep.shown[i] for i in asked] + [ep.shown[j]])
    it = Item(ep.level, ep.seed, ep.n, ep.queries[j], ep.labels[j], 0, episode_graph(e2, len(asked), mode), len(ep.visible), (), ())
    it.t = len(asked)
    return it


def newly_knowable(ep: Episode, asked: list[int], candidate: int, targets: list[int]) -> int:
    """How many of the target claims become knowable if `candidate` is asked next (its true verdict revealed)."""
    base = [i for i in asked]

    def count(order):
        total = 0
        for j in targets:                                  # score each target separately, seeing only `order`
            e3 = Episode(ep.level, ep.seed, ep.n, ep.visible, [ep.queries[i] for i in order] + [ep.queries[j]],
                         [ep.labels[i] for i in order] + [ep.labels[j]], ep.full)
            total += int(knowable(e3, len(order)))
        return total
    return count(base + [candidate]) - count(base)


def make_big_episodes(n: int, seeds, k: int = 10, vis_frac: float = 0.5, chain: float = 0.0, rng_seed: int = 0) -> list[Episode]:
    """Hidden-premise episodes on worlds of n kinds (n <= 10 so that the exact semantics can label and check provability). `chain` > 0 makes long 'all' chains."""
    from ..logic import Semantics
    rng = random.Random(rng_seed)
    sem = Semantics(n)
    out = []
    for seed in seeds:
        prem = ld.big_world(n, seed, max(1, n // 4), 1, 1, max(1, n // 4), root_prob=0.05 if chain else 0.1, chain=chain)
        pset = {p.canonical() for p in prem}
        vis = [p for p in prem if rng.random() < vis_frac] or [rng.choice(prem)]
        if len(vis) == len(prem) and len(prem) > 1:
            vis.remove(rng.choice(vis))
        seen, pool = set(), []
        for c in all_claims(n):
            key = c.canonical()
            if c.a == c.b or key in pset or key in seen:
                continue
            seen.add(key)
            pool.append(c)
        qs = rng.sample(pool, min(k, len(pool)))
        out.append(Episode(n, seed, n, vis, qs, [_VERDICT[sem.evaluate(prem, c).value] for c in qs], prem))
    return out


def support_marks(ep: Episode, randomise: bool = False, seed: int = 0) -> list[set[int]]:
    """marks[t] = positions (in ep.visible) of the visible premises used in the minimal proofs of the claims revealed before step t.
    The proof of a follows-claim is its minimal support; of a contradicted claim, the minimal support of its negation. Hidden premises used
    in a proof are not shown (the learner cannot see them). randomise=True keeps the number of marked premises but picks them at random (control)."""
    from ..logic import Semantics, minimal_support
    sem = _SEM.setdefault(ep.n, Semantics(ep.n))
    rng = random.Random(seed)
    used: set[int] = set()
    out = []
    full_idx = {id(p): i for i, p in enumerate(ep.full)}
    vis_pos = {full_idx.get(id(p)): pi for pi, p in enumerate(ep.visible)}
    for t in range(len(ep.queries)):
        out.append(set(used))
        if t == len(ep.queries) - 1:
            break
        c, y = ep.queries[t], ep.labels[t]
        if y == 2:
            continue
        sup = minimal_support(sem, ep.full, c if y == 0 else c.negation())
        vis = {vis_pos[i] for i in sup if i in vis_pos}
        if randomise and vis:
            vis = set(rng.sample(range(len(ep.visible)), min(len(vis), len(ep.visible))))
        used |= vis
    return out

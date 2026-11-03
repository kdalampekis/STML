"""Lemmas as a scratchpad for the graph judge ('chain of thought' for graphs).

A world's premises are the graph; a LEMMA is a claim already established (entailed) and added to the graph as an extra edge of one of 8 new edge types
(form x direction, ids 8..15). Training shows the judge true lemmas (plus a few false ones); at test time the judge builds its own lemmas: it judges every
claim of the world, keeps the ones it is confident follow, adds them as edges and judges again, so a deep proof can be assembled from short steps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..logic import FORM_ORDER, SYMMETRIC_FORMS, Claim, all_claims
from ..referee import referee
from . import logic_data as ld
from .graphs import Graph
from .logic_data import TRUTHS, Item, claim_graph

LEMMA_BASE = len(ld.EDGE_TYPES)            # 8
N_TYPES_LEMMA = 2 * LEMMA_BASE


@dataclass
class LemmaWorld:
    n: int
    seed: int
    premises: list[Claim]
    pool: list[Claim]                      # non-premise claims, mirror duplicates removed
    labels: list[int]                      # index into TRUTHS, from the SAT referee
    cores: list[tuple]


def build_world(n: int, seed: int, chain: float) -> LemmaWorld:
    prem = ld.big_world(n, seed, max(1, n // 4), 1, 1, max(1, n // 4), root_prob=0.05 if chain else 0.1, chain=chain)
    pset = {p.canonical() for p in prem}
    seen, pool = set(), []
    for c in all_claims(n):
        k = c.canonical()
        if c.a == c.b or k in pset or k in seen:
            continue
        seen.add(k)
        pool.append(c)
    certs = [referee(n, prem, c) for c in pool]
    return LemmaWorld(n, seed, prem, pool, [TRUTHS.index(c.status) for c in certs], [tuple(c.core) if c.status != "open" else () for c in certs])


def with_lemmas(g: Graph, lemmas: list[Claim]) -> Graph:
    if not lemmas:
        return Graph(g.x, g.edges, N_TYPES_LEMMA)
    extra = []
    for c in lemmas:
        f = FORM_ORDER.index(c.form)
        extra.append((c.a, c.b, LEMMA_BASE + 2 * f))
        extra.append((c.b, c.a, LEMMA_BASE + (2 * f if c.form in SYMMETRIC_FORMS else 2 * f + 1)))
    return Graph(g.x, np.concatenate([g.edges, np.array(extra, dtype=np.int64)]), N_TYPES_LEMMA)


def item_for(w: LemmaWorld, j: int, lemmas: list[Claim]) -> Item:
    g = with_lemmas(claim_graph(w.n, w.premises, w.pool[j]), lemmas)
    return Item(w.n, w.seed, w.n, w.pool[j], w.labels[j], 0, g, len(w.premises), (), tuple(w.premises))


def picks(w: LemmaWorld, rng: random.Random, per_world: int = 21) -> list[int]:
    out = []
    for t in range(3):
        idx = [i for i, l in enumerate(w.labels) if l == t]
        out += rng.sample(idx, min(per_world // 3, len(idx)))
    return out


def train_items(worlds: list[LemmaWorld], rng: random.Random, use_lemmas: bool, qs=(0.0, 0.15, 0.3, 0.6), false_rate: float = 0.1) -> list[Item]:
    """Target items (balanced classes); with `use_lemmas` each carries a random subset of the world's TRUE lemmas (rate drawn from `qs`) plus a few false ones."""
    out = []
    for w in worlds:
        ent = [i for i, l in enumerate(w.labels) if l == 0]
        non = [i for i, l in enumerate(w.labels) if l != 0]
        for j in picks(w, rng):
            lem = []
            if use_lemmas:
                q = rng.choice(qs)
                lem = [w.pool[i] for i in ent if i != j and rng.random() < q]
                if lem and rng.random() < 0.5:
                    lem += [w.pool[i] for i in rng.sample(non, min(len(non), max(1, round(false_rate * len(lem)))))]
            out.append(item_for(w, j, lem))
    return out


def self_lemma_rounds(model, w: LemmaWorld, targets: list[int], rounds: int, tau: float, device, mode: str = "self", rng: random.Random | None = None):
    """Iterate: judge every pool claim (with the lemmas gathered so far, minus the claim itself), keep claims judged 'follows' with prob > tau as lemmas.
    mode 'self' = the model's own lemmas, 'random' = same number of random NON-entailed claims (control), 'oracle' = a random 30% of the true entailed claims.
    Returns per-round (accuracy on the targets, lemma precision, lemma count); round 0 has no lemmas (oracle: the true subset)."""
    import torch
    from .logic_judge import logits_of
    ent = [i for i, l in enumerate(w.labels) if l == 0]
    non = [i for i, l in enumerate(w.labels) if l != 0]
    lem: set[int] = set(i for i in ent if rng.random() < 0.3) if mode == "oracle" else set()
    stats = []
    for r in range(rounds + 1):
        items = [item_for(w, j, [w.pool[i] for i in lem if i != j]) for j in range(len(w.pool))]
        p = torch.softmax(logits_of(model, items, device), 1).cpu().numpy()
        pred = p.argmax(1)
        acc = float(np.mean([pred[j] == w.labels[j] for j in targets]))
        prec = float(np.mean([w.labels[i] == 0 for i in lem])) if lem else float("nan")
        stats.append((acc, prec, len(lem)))
        if mode == "oracle":
            continue
        new = {i for i in range(len(w.pool)) if p[i, 0] > tau}
        lem = new if mode == "self" else set(rng.sample(non, min(len(non), len(new))))
    return stats

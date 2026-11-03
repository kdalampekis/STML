"""Logic worlds as judging problems for the STML judge (H3).

An item = (world premises, one claim, exact truth). The judge sees the premises as a graph over
nameless kinds and must answer follows / contradicted / open; it may also abstain ("I don't know yet").
Truth comes from the exact semantics (agrees with the SAT referee on 13,252/13,252 checks); the judge never
calls either.

Graph: node per kind; one directed edge per premise, typed by form and direction (8 types: 4 forms x
forward/backward, so that information can flow both ways along a premise while keeping its direction;
the symmetric forms No/Some get the same type both ways). Node features: constant, in/out degree,
world size, is-subject-of-the-claim, is-object-of-the-claim, and the claim's form (one-hot).
Kind names and indices carry no information.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..logic import FORM_ORDER, SYMMETRIC_FORMS, Claim, all_claims
from ..world import generate_world, level_config
from .graphs import Graph

TRUTHS = ("follows", "contradicted", "open")
_VERDICT = {"entailed": 0, "contradicted": 1, "undetermined": 2}
EDGE_TYPES = tuple(f"{f.value}:{d}" for f in FORM_ORDER for d in ("fwd", "bwd"))
NODE_DIM = 6 + len(FORM_ORDER)
SIZE_FREE = False    # True: no world-size feature and raw (not size-normalised) degrees, so big worlds stay in range


@dataclass
class Item:
    level: int
    seed: int
    n: int
    claim: Claim
    label: int                  # index into TRUTHS
    support: int                # premises needed (entailed claims), else 0
    graph: Graph | None = None
    n_prem: int = 0
    core: tuple = ()            # premises (indices) the referee's proof of follows / contradicted depends on
    premises: tuple = ()        # the world's premises (kept so that sub-worlds can be rebuilt)


def claim_graph(n: int, premises: list[Claim], claim: Claim) -> Graph:
    edges, deg_in, deg_out = [], np.zeros(n), np.zeros(n)
    for p in premises:
        f = FORM_ORDER.index(p.form)
        edges.append((p.a, p.b, 2 * f))
        edges.append((p.b, p.a, 2 * f if p.form in SYMMETRIC_FORMS else 2 * f + 1))
        deg_out[p.a] += 1
        deg_in[p.b] += 1
    x = np.zeros((n, NODE_DIM), dtype=np.float32)
    if SIZE_FREE:
        x[:, 0], x[:, 1], x[:, 2], x[:, 3] = 1.0, np.minimum(deg_in, 4) / 4, np.minimum(deg_out, 4) / 4, 0.0
    else:
        x[:, 0], x[:, 1], x[:, 2], x[:, 3] = 1.0, deg_in / max(1, n), deg_out / max(1, n), n / 12
    x[claim.a, 4] = 1.0
    x[claim.b, 5] = 1.0
    x[:, 6 + FORM_ORDER.index(claim.form)] = 1.0
    return Graph(x, np.array(edges, dtype=np.int64).reshape(-1, 3), len(EDGE_TYPES))


def make_items(levels, seeds, per_world: int = 20, rng_seed: int = 0) -> list[Item]:
    """For each (level, world seed): up to `per_world` random non-premise claims (mirror duplicates removed)."""
    from ..logic import minimal_support
    from ..referee import referee
    rng = random.Random(rng_seed)
    out = []
    for level in levels:
        cfg = level_config(level)
        for seed in seeds:
            w = generate_world(cfg, seed)
            seen, pool = set(), []
            for c in all_claims(w.n, w.claim_forms):
                k = c.canonical()
                if c.a == c.b or k in w.premise_set or k in seen:
                    continue
                seen.add(k)
                pool.append(c)
            for c in rng.sample(pool, min(per_world, len(pool))):
                label = _VERDICT[w.semantics.evaluate(w.premises, c).value]
                support = len(minimal_support(w.semantics, w.premises, c)) if label == 0 else 0
                core = tuple(referee(w.n, w.premises, c).core) if label < 2 else ()
                out.append(Item(level, seed, w.n, c, label, support, claim_graph(w.n, w.premises, c), len(w.premises), core,
                                tuple(w.premises)))
    return out


# ----------------------------------------------------------------------------- larger worlds (referee-labelled)

def _holds(c: Claim, model: list[int]) -> bool:
    ia, ib = 1 << c.a, 1 << c.b
    if c.form.value == "all":
        return all(not (m & ia) or (m & ib) for m in model)
    if c.form.value == "no":
        return all(not ((m & ia) and (m & ib)) for m in model)
    if c.form.value == "some":
        return any((m & ia) and (m & ib) for m in model)
    return any((m & ia) and not (m & ib) for m in model)


def big_world(n: int, seed: int, n_no: int, n_some: int, n_some_not: int, n_overlaps: int,
              root_prob: float = 0.1, chain: float = 0.0) -> list[Claim]:
    """Premises of a random hidden world with n kinds (any n; no 2**n semantics needed).

    Same recipe as stml.world.generate_world: a forest of 'all' links, one individual per kind and ancestors,
    a few individuals crossing the hierarchy, and extra No / Some / Some-not premises that are true in the
    hidden world and not already entailed (checked with the SAT referee).
    """
    from ..referee import referee
    rng = random.Random(seed)
    # chain = probability that a kind's parent is the previous kind (long 'all' chains, i.e. deep proofs)
    parents = [None if i == 0 or rng.random() < root_prob else (i - 1 if rng.random() < chain else rng.randrange(i))
               for i in range(n)]
    anc = []
    for i, p in enumerate(parents):
        anc.append((1 << i) | (anc[p] if p is not None else 0))
    related = lambda x, y: bool(anc[x] & (1 << y)) or bool(anc[y] & (1 << x))
    model = set(anc)
    unrelated = [(x, y) for x in range(n) for y in range(x + 1, n) if not related(x, y)]
    for x, y in rng.sample(unrelated, min(n_overlaps, len(unrelated))):
        model.add(anc[x] | anc[y])
    model = sorted(model)
    premises = [Claim(FORM_ORDER[0], i, p) for i, p in enumerate(parents) if p is not None]
    for form, k in ((FORM_ORDER[1], n_no), (FORM_ORDER[2], n_some), (FORM_ORDER[3], n_some_not)):
        cands = [c for c in all_claims(n, (form,)) if c.a != c.b and _holds(c, model)
                 and (form not in SYMMETRIC_FORMS or c.a < c.b)]
        rng.shuffle(cands)
        for c in cands:
            if k == 0:
                break
            if referee(n, premises, c).status == "open":
                premises.append(c)
                k -= 1
    premises = [Claim(c.form, c.b, c.a) if c.form in SYMMETRIC_FORMS and rng.random() < 0.5 else c for c in premises]
    rng.shuffle(premises)
    return premises


def make_big_items(n: int, seeds, per_world: int = 21, rng_seed: int = 0, chain: float = 0.0) -> list[Item]:
    """Items from n-kind worlds, labelled by the SAT referee. `support` = size of the referee's unsat core."""
    from ..referee import referee
    rng = random.Random(rng_seed)
    out = []
    for seed in seeds:
        prem = big_world(n, seed, max(1, n // 4), 1, 1, max(1, n // 4), root_prob=0.05 if chain else 0.1, chain=chain)
        pset = {p.canonical() for p in prem}
        seen, pool = set(), []
        for c in all_claims(n):
            k = c.canonical()
            if c.a == c.b or k in pset or k in seen:
                continue
            seen.add(k)
            pool.append(c)
        certs = [(c, referee(n, prem, c)) for c in pool]
        by = {t: [x for x in certs if x[1].status == t] for t in TRUTHS}
        picks = []
        for t in TRUTHS:    # balanced classes: a third of the claims from each truth value when available
            picks += rng.sample(by[t], min(per_world // 3, len(by[t])))
        for c, cert in picks:
            out.append(Item(n, seed, n, c, TRUTHS.index(cert.status),
                            len(cert.core) if cert.status == "follows" else 0, claim_graph(n, prem, c), len(prem),
                            tuple(cert.core) if cert.status != "open" else (), tuple(prem)))
    return out


def sub_item(it: Item, keep: list[int]) -> Item:
    """The same claim judged in the world that keeps only the premises with these indices."""
    prem = [it.premises[j] for j in keep]
    return Item(it.level, it.seed, it.n, it.claim, it.label, it.support, claim_graph(it.n, prem, it.claim), len(prem), (), tuple(prem))

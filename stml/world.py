"""Hidden worlds and the curriculum of levels.

A world is a set of invented kinds ("wugs", "daxes", ...) with a hidden
ground-truth population. The learner sees only a few true premises and must
discover what follows from them. Nonsense names keep prior knowledge out:
the only way to succeed is to reason.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import cached_property

from .logic import FORM_ORDER, Claim, Form, Semantics, all_claims

KIND_NAMES = [
    "wugs", "daxes", "blickets", "feps", "tomas", "zorbs", "kikis", "boubas",
    "glims", "snerks", "plonks", "yarps", "vexes", "mibs", "tulvers", "quazes",
    "dorns", "fribbles", "lorps", "nerfs", "grobs", "pimbles", "skaps", "virns",
]

INDIVIDUAL_NAMES = [
    "Ika", "Romo", "Tesh", "Vala", "Nim", "Oru", "Pell", "Sef",
    "Kiro", "Mada", "Yul", "Zen", "Bori", "Lune", "Fasi", "Gred",
]


@dataclass(frozen=True)
class LevelConfig:
    level: int
    title: str
    lesson: str                 # what this level is designed to teach
    n_kinds: int
    claim_forms: tuple[Form, ...]
    n_no: int = 0               # 'No' premises
    n_some: int = 0             # 'Some' premises
    n_some_not: int = 0         # 'Some ... not' premises
    n_overlaps: int = 0         # individuals that cross the hierarchy
    root_prob: float = 0.2      # chance a kind has no parent
    min_targets: int = 2
    max_targets: int = 10


CURRICULUM: tuple[LevelConfig, ...] = (
    LevelConfig(1, "Chains", "chains of 'all' (transitivity) and their direction",
                n_kinds=4, claim_forms=(Form.ALL,), root_prob=0.0, min_targets=2, max_targets=4),
    LevelConfig(2, "Exclusion", "how 'no' spreads down a chain of 'all'",
                n_kinds=5, claim_forms=(Form.ALL, Form.NO), n_no=1, root_prob=0.3,
                min_targets=3, max_targets=7),
    LevelConfig(3, "Some", "what 'some' does and does not license",
                n_kinds=5, claim_forms=(Form.ALL, Form.NO, Form.SOME), n_no=1, n_some=1,
                n_overlaps=1, root_prob=0.3, min_targets=3, max_targets=8),
    LevelConfig(4, "Exceptions", "'some ... not' and reasoning from exceptions",
                n_kinds=6, claim_forms=FORM_ORDER, n_no=1, n_some=1, n_some_not=1,
                n_overlaps=1, root_prob=0.3, min_targets=4, max_targets=10),
    LevelConfig(5, "Transfer", "all forms together in a larger world",
                n_kinds=7, claim_forms=FORM_ORDER, n_no=2, n_some=1, n_some_not=1,
                n_overlaps=2, root_prob=0.25, min_targets=5, max_targets=12),
)


def level_config(level: int) -> LevelConfig:
    for cfg in CURRICULUM:
        if cfg.level == level:
            return cfg
    raise KeyError(f"no level {level}")


@dataclass
class World:
    level: int
    seed: int
    names: list[str]
    parents: list[int | None]
    model: list[int]                 # hidden ground-truth inhabited regions
    premises: list[Claim]
    claim_forms: tuple[Form, ...]
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.names)

    @cached_property
    def semantics(self) -> Semantics:
        return Semantics(self.n)

    @cached_property
    def entailed(self) -> list[Claim]:
        return [c for c in all_claims(self.n) if self.semantics.entails(self.premises, c)]

    def is_weak(self, claim: Claim) -> bool:
        """True but implied by a stronger entailed universal (e.g. Some A are B
        when All A are B already follows)."""
        sem, P = self.semantics, self.premises
        if claim.form == Form.SOME:
            return sem.entails(P, Claim(Form.ALL, claim.a, claim.b)) or \
                sem.entails(P, Claim(Form.ALL, claim.b, claim.a))
        if claim.form == Form.SOME_NOT:
            return sem.entails(P, Claim(Form.NO, claim.a, claim.b))
        return False

    @cached_property
    def premise_set(self) -> set[Claim]:
        return {p.canonical() for p in self.premises}

    @cached_property
    def targets(self) -> list[Claim]:
        """Strong, non-trivial conclusions not given as premises (canonical)."""
        out: set[Claim] = set()
        for c in self.entailed:
            c = c.canonical()
            if c.form not in self.claim_forms or c.is_trivial():
                continue
            if c in self.premise_set or self.is_weak(c):
                continue
            out.add(c)
        return sorted(out, key=Claim.sort_key)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "seed": self.seed,
            "names": self.names,
            "parents": self.parents,
            "model": self.model,
            "premises": [p.key() for p in self.premises],
            "claim_forms": [f.value for f in self.claim_forms],
        }


def _ancestors(parents: list[int | None]) -> list[int]:
    anc: list[int] = []
    for i, p in enumerate(parents):
        anc.append((1 << i) | (anc[p] if p is not None else 0))
    return anc


def generate_world(cfg: LevelConfig, seed: int, max_attempts: int = 2000) -> World:
    """Sample a hidden world and premises whose consequences fit the level."""
    rng = random.Random(seed)
    n = cfg.n_kinds
    for _ in range(max_attempts):
        names = rng.sample(KIND_NAMES, n)
        parents: list[int | None] = []
        for i in range(n):
            if i == 0 or rng.random() < cfg.root_prob:
                parents.append(None)
            else:
                parents.append(rng.randrange(i))
        anc = _ancestors(parents)
        model = set(anc)  # one individual of each kind (and its ancestors)

        related = lambda x, y: bool(anc[x] & (1 << y)) or bool(anc[y] & (1 << x))
        unrelated = [(x, y) for x in range(n) for y in range(x + 1, n) if not related(x, y)]
        if cfg.n_overlaps > len(unrelated):
            continue
        for x, y in rng.sample(unrelated, cfg.n_overlaps):
            model.add(anc[x] | anc[y])
        model_list = sorted(model)
        sem = Semantics(n)

        premises = [Claim(Form.ALL, i, p) for i, p in enumerate(parents) if p is not None]

        def pick(form: Form, k: int, extra=lambda c: True) -> bool:
            if k == 0:
                return True
            cands = [c for c in all_claims(n, (form,))
                     if c.a != c.b and sem.holds(c, model_list) and extra(c)
                     and not sem.entails(premises, c)]
            if form in (Form.NO, Form.SOME):
                cands = [c for c in cands if c.a < c.b]
            if len(cands) < k:
                return False
            premises.extend(rng.sample(cands, k))
            return True

        def maximal_no(c: Claim) -> bool:
            # the highest-level exclusion: parents are not also excluded
            pa, pb = parents[c.a], parents[c.b]
            if pa is not None and sem.holds(Claim(Form.NO, pa, c.b), model_list):
                return False
            if pb is not None and sem.holds(Claim(Form.NO, c.a, pb), model_list):
                return False
            return True

        ok = (pick(Form.NO, cfg.n_no, maximal_no)
              and pick(Form.SOME, cfg.n_some, lambda c: not related(c.a, c.b))
              and pick(Form.SOME_NOT, cfg.n_some_not))
        if not ok:
            continue

        # Vary surface form: shuffle order, flip symmetric statements at random.
        premises = [Claim(c.form, c.b, c.a) if c.form in (Form.NO, Form.SOME) and rng.random() < 0.5
                    else c for c in premises]
        rng.shuffle(premises)

        world = World(cfg.level, seed, names, parents, model_list, premises, cfg.claim_forms)
        assert sem.is_model(premises, model_list), "premises must be true in the hidden world"
        if cfg.min_targets <= len(world.targets) <= cfg.max_targets:
            return world
    raise RuntimeError(f"could not generate a world for level {cfg.level} with seed {seed}")

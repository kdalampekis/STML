"""Categorical (syllogistic) logic over named kinds, with exact semantics.

A world contains individuals; each individual belongs to some set of kinds.
Because only membership matters, a world is fully described by which
"regions" (combinations of kinds) are inhabited. A region is a bitmask over
kinds: region 0b101 = "is kind 0 and kind 2, and nothing else".

Four claim forms (the classical A/E/I/O statements):
    ALL       All A are B          -> region (A and not B) is empty
    NO        No A are B           -> region (A and B) is empty
    SOME      Some A are B         -> region (A and B) is inhabited
    SOME_NOT  Some A are not B     -> region (A and not B) is inhabited

With existential import (the default), every kind has at least one member.

Universal claims force regions to be empty; existential claims need some
region to be inhabited. A set of claims is satisfiable iff every existential
still has an allowed (not forced-empty) region. This makes entailment exact
and fast: premises |= c  iff  premises + not(c) is unsatisfiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence


class Form(str, Enum):
    ALL = "all"
    NO = "no"
    SOME = "some"
    SOME_NOT = "some_not"


FORM_ORDER = (Form.ALL, Form.NO, Form.SOME, Form.SOME_NOT)
UNIVERSAL_FORMS = frozenset({Form.ALL, Form.NO})
SYMMETRIC_FORMS = frozenset({Form.NO, Form.SOME})
_NEGATION = {
    Form.ALL: Form.SOME_NOT,
    Form.SOME_NOT: Form.ALL,
    Form.NO: Form.SOME,
    Form.SOME: Form.NO,
}


@dataclass(frozen=True)
class Claim:
    form: Form
    a: int
    b: int

    @property
    def is_universal(self) -> bool:
        return self.form in UNIVERSAL_FORMS

    def negation(self) -> "Claim":
        return Claim(_NEGATION[self.form], self.a, self.b)

    def canonical(self) -> "Claim":
        """'No A are B' = 'No B are A' and 'Some A are B' = 'Some B are A'."""
        if self.form in SYMMETRIC_FORMS and self.a > self.b:
            return Claim(self.form, self.b, self.a)
        return self

    def is_trivial(self) -> bool:
        """All A are A / Some A are A: true but uninformative."""
        return self.a == self.b and self.form in (Form.ALL, Form.SOME)

    def text(self, names: Sequence[str]) -> str:
        a, b = names[self.a], names[self.b]
        return {
            Form.ALL: f"All {a} are {b}",
            Form.NO: f"No {a} are {b}",
            Form.SOME: f"Some {a} are {b}",
            Form.SOME_NOT: f"Some {a} are not {b}",
        }[self.form]

    def key(self) -> str:
        return f"{self.form.value}:{self.a}:{self.b}"

    @classmethod
    def from_key(cls, key: str) -> "Claim":
        form, a, b = key.split(":")
        return cls(Form(form), int(a), int(b))

    def sort_key(self) -> tuple[int, int, int]:
        return (FORM_ORDER.index(self.form), self.a, self.b)

    def __repr__(self) -> str:  # compact, readable in test output
        return f"Claim({self.key()})"


def all_claims(n: int, forms: Iterable[Form] = FORM_ORDER) -> list[Claim]:
    forms = tuple(forms)
    return [Claim(f, a, b) for f in forms for a in range(n) for b in range(n)]


class Verdict(str, Enum):
    ENTAILED = "entailed"          # must be true given the premises
    CONTRADICTED = "contradicted"  # must be false given the premises
    UNDETERMINED = "undetermined"  # could go either way


class Semantics:
    """Exact region semantics for n kinds."""

    def __init__(self, n: int, existential_import: bool = True):
        if not 1 <= n <= 10:
            raise ValueError("n must be between 1 and 10")
        self.n = n
        self.existential_import = existential_import
        self.regions = tuple(range(1, 1 << n))
        self._full = 0
        for r in self.regions:
            self._full |= 1 << r
        self._cache: dict[Claim, int] = {}

    # -- region sets -------------------------------------------------------
    def region_set(self, claim: Claim) -> int:
        """Bitset of regions a claim talks about.

        For ALL/NO these are the regions the claim forces empty; for
        SOME/SOME_NOT the regions at least one of which must be inhabited.
        """
        cached = self._cache.get(claim)
        if cached is not None:
            return cached
        ma, mb = 1 << claim.a, 1 << claim.b
        want_b = claim.form in (Form.NO, Form.SOME)
        bits = 0
        for r in self.regions:
            if (r & ma) and bool(r & mb) == want_b:
                bits |= 1 << r
        self._cache[claim] = bits
        return bits

    def _existence_claims(self) -> list[Claim]:
        if not self.existential_import:
            return []
        return [Claim(Form.SOME, i, i) for i in range(self.n)]

    def allowed_regions(self, claims: Iterable[Claim]) -> int:
        forced = 0
        for c in claims:
            if c.is_universal:
                forced |= self.region_set(c)
        return self._full & ~forced

    # -- satisfiability and entailment --------------------------------------
    def satisfiable(self, claims: Iterable[Claim]) -> bool:
        claims = list(claims)
        allowed = self.allowed_regions(claims)
        existentials = [c for c in claims if not c.is_universal]
        existentials += self._existence_claims()
        return all(self.region_set(c) & allowed for c in existentials)

    def entails(self, premises: Sequence[Claim], claim: Claim) -> bool:
        return not self.satisfiable([*premises, claim.negation()])

    def evaluate(self, premises: Sequence[Claim], claim: Claim) -> Verdict:
        if not self.satisfiable(premises):
            raise ValueError("premises are inconsistent")
        if self.entails(premises, claim):
            return Verdict.ENTAILED
        if self.entails(premises, claim.negation()):
            return Verdict.CONTRADICTED
        return Verdict.UNDETERMINED

    # -- models ------------------------------------------------------------
    def holds(self, claim: Claim, model: Iterable[int]) -> bool:
        """Does a claim hold in a model (a collection of inhabited regions)?"""
        rs = self.region_set(claim)
        hit = any((rs >> r) & 1 for r in model)
        return not hit if claim.is_universal else hit

    def is_model(self, claims: Iterable[Claim], model: Sequence[int]) -> bool:
        claims = [*claims, *self._existence_claims()]
        return all(self.holds(c, model) for c in claims)

    def find_model(self, claims: Sequence[Claim]) -> list[int] | None:
        """A small model of the claims (fewest individuals, greedy), or None."""
        claims = list(claims)
        if not self.satisfiable(claims):
            return None
        allowed = self.allowed_regions(claims)
        needs = [c for c in claims if not c.is_universal] + self._existence_claims()
        need_sets = [self.region_set(c) & allowed for c in needs]
        chosen: list[int] = []
        remaining = list(range(len(need_sets)))
        candidates = [r for r in self.regions if (allowed >> r) & 1]
        while remaining:
            def score(r: int) -> tuple[int, int, int]:
                covered = sum(1 for i in remaining if (need_sets[i] >> r) & 1)
                # prefer covering more, then simpler individuals, then stable order
                return (covered, -bin(r).count("1"), -r)

            best = max(candidates, key=score)
            chosen.append(best)
            remaining = [i for i in remaining if not (need_sets[i] >> best) & 1]
        return sorted(set(chosen))

    def true_claims(self, model: Sequence[int], forms: Iterable[Form] = FORM_ORDER) -> list[Claim]:
        return [c for c in all_claims(self.n, forms) if self.holds(c, model)]


def minimal_support(sem: Semantics, premises: Sequence[Claim], claim: Claim) -> list[int]:
    """Indices of a minimal subset of premises that still entails the claim.

    Deletion-based: minimal (no premise can be dropped), not necessarily the
    smallest possible. Returns [] if the claim follows from existence alone.
    """
    if not sem.entails(premises, claim):
        raise ValueError(f"{claim} is not entailed by the premises")
    keep = list(range(len(premises)))
    for i in reversed(range(len(premises))):
        trial = [j for j in keep if j != i]
        if sem.entails([premises[j] for j in trial], claim):
            keep = trial
    return keep

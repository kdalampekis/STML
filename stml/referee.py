"""The SAT referee: an independent check of every verdict.

Encoding (see the proposal, Mathematics section 3):
  variable x[j][i]  <=>  individual j is a member of kind i
  All A are B        for every individual j:  -x[j][A] or  x[j][B]
  No A are B         for every individual j:  -x[j][A] or -x[j][B]
  Some A are B       a dedicated witness w:    x[w][A] and  x[w][B]
  Some A are not B   a dedicated witness w:    x[w][A] and -x[w][B]
  K_i is non-empty   a dedicated witness w_i:  x[w_i][i]

By the small-model property, one witness per existential statement suffices,
so the encoding is exact and polynomial in size. Every premise is guarded by a
selector literal, so an UNSAT answer comes with a core naming the premises
the verdict depends on, and a SAT answer comes with a countermodel.

The referee shares no code with logic.py, proof.py or critic.py, and the
learner never calls it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from pysat.solvers import Solver

from .logic import Claim, Form

SOLVER = "cadical153"


@dataclass
class Certificate:
    status: str                     # "follows" | "contradicted" | "open"
    core: list[int] = field(default_factory=list)       # premise indices (UNSAT side)
    countermodel: list[list[int]] = field(default_factory=list)  # kinds per individual (open)
    conflicts: int = 0              # solver effort, a difficulty measure
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {"status": self.status, "core": self.core, "countermodel": self.countermodel,
                "conflicts": self.conflicts}


class _Encoding:
    def __init__(self, n: int, claims: Sequence[Claim], existential_import: bool):
        self.n = n
        n_exist = sum(1 for c in claims if c.form in (Form.SOME, Form.SOME_NOT))
        self.m = n_exist + (n if existential_import else 0)
        self.m = max(self.m, 1)
        self.top = n * self.m
        self.next_witness = 0

    def var(self, j: int, i: int) -> int:
        return 1 + j * self.n + i

    def selector(self) -> int:
        self.top += 1
        return self.top

    def witness(self) -> int:
        w = self.next_witness
        self.next_witness += 1
        return w

    def clauses(self, claim: Claim, sel: int | None) -> list[list[int]]:
        guard = [-sel] if sel is not None else []
        a, b = claim.a, claim.b
        out: list[list[int]] = []
        if claim.form == Form.ALL:
            for j in range(self.m):
                out.append(guard + [-self.var(j, a), self.var(j, b)])
        elif claim.form == Form.NO:
            for j in range(self.m):
                out.append(guard + [-self.var(j, a), -self.var(j, b)])
        else:
            w = self.witness()
            out.append(guard + [self.var(w, a)])
            if claim.form == Form.SOME:
                out.append(guard + [self.var(w, b)])
            else:
                out.append(guard + [-self.var(w, b)])
        return out


def _negate(c: Claim) -> Claim:
    return Claim({Form.ALL: Form.SOME_NOT, Form.SOME_NOT: Form.ALL,
                  Form.NO: Form.SOME, Form.SOME: Form.NO}[c.form], c.a, c.b)


def _check(n: int, premises: Sequence[Claim], extra: Claim,
           existential_import: bool) -> tuple[bool, list[int], list[list[int]], int]:
    """Is premises + extra satisfiable? Returns (sat, core, model, conflicts)."""
    enc = _Encoding(n, [*premises, extra], existential_import)
    clauses: list[list[int]] = []
    selectors: list[int] = []
    for p in premises:
        s = enc.selector()
        selectors.append(s)
        clauses += enc.clauses(p, s)
    clauses += enc.clauses(extra, None)
    if existential_import:
        for i in range(n):
            clauses.append([enc.var(enc.witness(), i)])
    with Solver(name=SOLVER, bootstrap_with=clauses) as s:
        sat = s.solve(assumptions=selectors)
        stats = s.accum_stats() or {}
        conflicts = int(stats.get("conflicts", 0))
        if sat:
            model = set(v for v in s.get_model() if v > 0)
            people = []
            for j in range(enc.m):
                kinds = [i for i in range(n) if enc.var(j, i) in model]
                if kinds and kinds not in people:
                    people.append(kinds)
            return True, [], people, conflicts
        core = sorted(selectors.index(v) for v in (s.get_core() or []) if v in selectors)
        return False, core, [], conflicts


def referee(n: int, premises: Sequence[Claim], claim: Claim,
            existential_import: bool = True) -> Certificate:
    """Certified status of a claim given premises."""
    t0 = time.perf_counter()
    sat_neg, core_f, model, c1 = _check(n, premises, _negate(claim), existential_import)
    if not sat_neg:
        return Certificate("follows", core=core_f, conflicts=c1, seconds=time.perf_counter() - t0)
    sat_pos, core_c, _, c2 = _check(n, premises, claim, existential_import)
    if not sat_pos:
        return Certificate("contradicted", core=core_c, conflicts=c1 + c2,
                           seconds=time.perf_counter() - t0)
    return Certificate("open", countermodel=model, conflicts=c1 + c2,
                       seconds=time.perf_counter() - t0)


def consistent(n: int, premises: Sequence[Claim], existential_import: bool = True) -> bool:
    if not premises:
        return True
    sat, *_ = _check(n, premises[:-1], premises[-1], existential_import)
    return sat

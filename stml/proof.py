"""Forward-chaining proofs with named syllogistic rules.

The semantics in logic.py decides what is true; this module explains *why*,
by deriving conclusions with the classical rules people learn in a logic
class. Every derived step records its rule and parents, so a proof can be
shown to a human or fed to a learner as a structured critique.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator, Sequence

from .logic import Claim, Form

# Rule ids -> pattern shown to people. Letters are placeholders for kinds.
RULES: dict[str, str] = {
    "premise": "given",
    "existence": "every kind has at least one member",
    "identity": "every A is an A",
    "conversion": "'no' and 'some' statements can be read in either order",
    "barbara": "All A are B, All B are C  =>  All A are C",
    "celarent": "All A are B, No B are C  =>  No A are C",
    "darii": "Some A are B, All B are C  =>  Some A are C",
    "ferio": "Some A are B, No B are C  =>  Some A are not C",
    "baroco": "Some A are not B, All C are B  =>  Some A are not C",
    "bocardo": "Some A are not B, All A are C  =>  Some C are not B",
}


@dataclass(frozen=True)
class Step:
    claim: Claim
    rule: str
    parents: tuple[Claim, ...] = ()


def _combine(x: Claim, y: Claim) -> Iterator[tuple[Claim, str]]:
    """Two-premise rules with x as the first premise."""
    A, B = x.a, x.b
    if x.form == Form.ALL:
        if y.form == Form.ALL and y.a == B:
            yield Claim(Form.ALL, A, y.b), "barbara"
        if y.form == Form.NO and y.a == B:
            yield Claim(Form.NO, A, y.b), "celarent"
    elif x.form == Form.SOME:
        if y.form == Form.ALL and y.a == B:
            yield Claim(Form.SOME, A, y.b), "darii"
        if y.form == Form.NO and y.a == B:
            yield Claim(Form.SOME_NOT, A, y.b), "ferio"
    elif x.form == Form.SOME_NOT:
        if y.form == Form.ALL and y.b == B:
            yield Claim(Form.SOME_NOT, A, y.a), "baroco"
        if y.form == Form.ALL and y.a == A:
            yield Claim(Form.SOME_NOT, y.b, B), "bocardo"


def derive(premises: Sequence[Claim], n: int, existential_import: bool = True) -> dict[Claim, Step]:
    """Deductive closure of the premises, each fact with its first derivation.

    Facts are found in breadth-first order, so recorded derivations are short.
    """
    known: dict[Claim, Step] = {}
    queue: deque[Claim] = deque()

    def add(c: Claim, rule: str, parents: tuple[Claim, ...] = ()) -> None:
        if c not in known:
            known[c] = Step(c, rule, parents)
            queue.append(c)

    for p in premises:
        add(p, "premise")
    for i in range(n):
        add(Claim(Form.ALL, i, i), "identity")
    if existential_import:
        for i in range(n):
            add(Claim(Form.SOME, i, i), "existence")

    while queue:
        c = queue.popleft()
        if c.form in (Form.NO, Form.SOME) and c.a != c.b:
            add(Claim(c.form, c.b, c.a), "conversion", (c,))
        for d in list(known):
            for new, rule in _combine(c, d):
                add(new, rule, (c, d))
            for new, rule in _combine(d, c):
                add(new, rule, (d, c))
    return known


def proof_steps(claim: Claim, closure: dict[Claim, Step]) -> list[Step]:
    """Steps proving a claim, parents before children, without repeats."""
    if claim not in closure:
        return []
    ordered: list[Step] = []
    seen: set[Claim] = set()

    def visit(c: Claim) -> None:
        if c in seen:
            return
        seen.add(c)
        step = closure[c]
        for p in step.parents:
            visit(p)
        ordered.append(step)

    visit(claim)
    return ordered


def rules_used(steps: Sequence[Step]) -> list[str]:
    return sorted({s.rule for s in steps if s.rule not in ("premise", "existence", "identity")})

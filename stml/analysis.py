"""Classify wrong claims into classical reasoning errors.

These labels are the raw material for the cognitive learner (Pillar 1): they
say *which* wrong rule a learner seems to be using, so its learning
trajectory can be modelled as rules gaining and losing trust over time.
"""

from __future__ import annotations

from .logic import Claim, Form
from .world import World

ERROR_TYPES = {
    "illicit_conversion": "Reversed an 'all' or 'some ... not' statement (All A are B does not give All B are A).",
    "undistributed_middle": "Linked two kinds only because both sit inside a third (All A are C, All B are C).",
    "some_chaining": "Chained two 'some' statements (Some A are B, Some B are C).",
    "two_negatives": "Drew a conclusion from two exclusions (No A are B, No B are C).",
    "illicit_process": "Spread an exclusion upward instead of downward (All B are A, No B are C does not give No A are C).",
    "overgeneralization": "Turned 'some' into 'all' or 'some ... not' into 'no'.",
    "contradiction": "Claimed something the premises rule out.",
    "other": "No classical pattern matched.",
}


def classify_error(world: World, claim: Claim) -> list[str]:
    """Labels for a wrong claim; [] if the claim is actually entailed."""
    sem, P, n = world.semantics, world.premises, world.n
    if sem.entails(P, claim):
        return []
    e = lambda f, a, b: sem.entails(P, Claim(f, a, b))
    a, b = claim.a, claim.b
    others = [m for m in range(n) if m not in (a, b)]
    labels: list[str] = []

    if claim.form == Form.ALL:
        if e(Form.ALL, b, a):
            labels.append("illicit_conversion")
        if any(e(Form.ALL, a, m) and e(Form.ALL, b, m) for m in others):
            labels.append("undistributed_middle")
        if e(Form.SOME, a, b):
            labels.append("overgeneralization")
    elif claim.form == Form.NO:
        if any(e(Form.NO, a, m) and e(Form.NO, m, b) for m in others):
            labels.append("two_negatives")
        if any((e(Form.ALL, m, a) and e(Form.NO, m, b)) or (e(Form.ALL, m, b) and e(Form.NO, m, a))
               for m in others):
            labels.append("illicit_process")
        if e(Form.SOME_NOT, a, b) or e(Form.SOME_NOT, b, a):
            labels.append("overgeneralization")
    elif claim.form == Form.SOME:
        if any(e(Form.SOME, a, m) and e(Form.SOME, m, b) for m in others):
            labels.append("some_chaining")
        if any(e(Form.NO, a, m) and e(Form.NO, m, b) for m in others):
            labels.append("two_negatives")
    elif claim.form == Form.SOME_NOT:
        if e(Form.SOME_NOT, b, a):
            labels.append("illicit_conversion")
        if any(e(Form.SOME_NOT, a, m) and e(Form.SOME_NOT, m, b) for m in others):
            labels.append("some_chaining")

    if sem.entails(P, claim.negation()):
        labels.append("contradiction")
    return labels or ["other"]

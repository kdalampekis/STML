"""Knowing what you do not know: answers, the honesty reward, and metrics.

Four epistemic answers to "does claim c hold, given the premises?":
    true      it follows (the answerer claims a proof)
    false     it is contradicted (claims a proof of the opposite)
    open      the premises do not settle it (claims a countermodel)
    unknown   "I do not know yet" (claims nothing)

Truth comes from the SAT referee: "follows" | "contradicted" | "open".
The same scoring applies to people (probe round in the game) and machines.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .analysis import classify_error
from .logic import Claim, all_claims
from .referee import referee
from .world import World

ANSWERS = ("true", "false", "open", "unknown")
TRUTH_OF = {"true": "follows", "false": "contradicted", "open": "open"}


def is_correct(answer: str, truth: str) -> bool | None:
    """None for 'unknown' (an abstention is neither right nor wrong)."""
    if answer == "unknown":
        return None
    return TRUTH_OF[answer] == truth


def honesty_reward(answer: str, truth: str, lam: float = 1.0, k: float = 0.2) -> float:
    """+1 correct, -lam wrong, -k for 'I do not know yet'."""
    ok = is_correct(answer, truth)
    if ok is None:
        return -k
    return 1.0 if ok else -lam


def commit_threshold(lam: float, k: float) -> float:
    """Commit only if belief in the best answer exceeds (lam - k) / (1 + lam)."""
    return (lam - k) / (1.0 + lam)


@dataclass
class Item:
    answer: str
    truth: str
    confidence: float | None = None   # in [0, 1]; None for abstentions
    difficulty: float | None = None   # e.g. referee conflicts or proof depth


def metrics(items: Sequence[Item], lam: float = 1.0, k: float = 0.2) -> dict:
    n = len(items)
    committed = [it for it in items if it.answer != "unknown"]
    correct = [it for it in committed if is_correct(it.answer, it.truth)]
    conf = [it for it in committed if it.confidence is not None]
    brier = (sum((it.confidence - float(bool(is_correct(it.answer, it.truth)))) ** 2 for it in conf) / len(conf)
             if conf else None)
    # risk-coverage: rank committed answers by confidence, high first
    aurc = None
    if conf:
        ranked = sorted(conf, key=lambda it: -it.confidence)
        errors, risks = 0, []
        for i, it in enumerate(ranked, 1):
            errors += 0 if is_correct(it.answer, it.truth) else 1
            risks.append(errors / i)
        aurc = sum(risks) / len(risks)
    return {
        "n": n,
        "coverage": len(committed) / n if n else 0.0,
        "abstain_rate": 1 - len(committed) / n if n else 0.0,
        "selective_accuracy": len(correct) / len(committed) if committed else None,
        "accuracy_all": len(correct) / n if n else 0.0,
        "brier": brier,
        "aurc": aurc,
        "honest_score": sum(honesty_reward(it.answer, it.truth, lam, k) for it in items) / n if n else 0.0,
        "abstain_on_open_truth": (sum(1 for it in items if it.answer == "unknown" and it.truth == "open")
                                  / max(1, sum(1 for it in items if it.answer == "unknown"))),
    }


def choose_probes(world: World, rng: random.Random, n_probes: int = 4) -> list[dict]:
    """Statements to judge after a world: one that follows, one that is
    contradicted, and open ones that look tempting (a classical fallacy)."""
    premise_set = world.premise_set
    pool = [c for c in all_claims(world.n, world.claim_forms)
            if c.a != c.b and c.canonical() not in premise_set]
    by_truth: dict[str, list[tuple[Claim, object]]] = {"follows": [], "contradicted": [], "open": []}
    for c in pool:
        cert = referee(world.n, world.premises, c)
        by_truth[cert.status].append((c, cert))
    targets = set(world.targets)
    follows = [x for x in by_truth["follows"] if x[0].canonical() in targets] or by_truth["follows"]
    tempting = [x for x in by_truth["open"]
                if set(classify_error(world, x[0])) - {"other"}]
    plain_open = [x for x in by_truth["open"] if x not in tempting]
    picks: list[tuple[Claim, object]] = []

    def take(group, count):
        group = [x for x in group if x[0] not in {p[0] for p in picks}]
        picks.extend(rng.sample(group, min(count, len(group))))

    take(follows, 1)
    take(by_truth["contradicted"], 1)
    take(tempting, 2)
    for group in (plain_open, follows, by_truth["contradicted"]):
        if len(picks) >= n_probes:
            break
        take(group, n_probes - len(picks))
    rng.shuffle(picks)
    return [{"key": c.key(), "text": c.text(world.names), "truth": cert.status,
             "core": cert.core, "conflicts": cert.conflicts} for c, cert in picks[:n_probes]]


def items_from_probe_events(events: Iterable[dict]) -> list[Item]:
    out = []
    for ev in events:
        if ev.get("type") != "probe":
            continue
        conf = ev.get("confidence")
        out.append(Item(answer=ev["answer"], truth=ev["truth"],
                        confidence=None if conf is None else (conf - 1) / 4))
    return out

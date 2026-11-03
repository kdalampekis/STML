"""A synthetic human-like learner for the Wugs and Daxes game (Pillar 1, minimal version).

Purpose: size the human study before running it. This is a *generative model of behaviour*, not a claim about how
people reason. Its structure follows the literature (see literature/SUMMARY.md); its numbers are ASSUMPTIONS, chosen so
that summary statistics land in plausible ranges (see `summary_stats` and the calibration targets in the power
experiment) and varied in a sensitivity analysis. Replace them with estimates from the real game logs when these exist.

It plays the real game: the 15 episodes, the real critic statuses, the real probe statements of every episode, and the
real error taxonomy (stml.analysis.ERROR_TYPES) as the fallacies a person may fall for.

Cognitive ingredients (the paper that motivates each in brackets):
  - limited reasoning depth: a proof of s premises is found with probability sigmoid(k (depth - s + 0.25))   [Dasgupta 2017: few
    sampled hypotheses; ACT-R style capacity limits]
  - fallacy proneness per error type, a logit per type                                                    [classical reasoning errors]
  - an internal evidence strength m for every answer; abstain ("don't know yet") when m < tau; confidence is a noisy
    read-out of m (confidence = a second-order computation)                                              [Pouget 2016, Fleming & Dolan 2012]
  - learning from feedback after each attempt: errors lower the fallacy logits, correct proofs raise depth, and the
    size of the correction grows with the confidence of the error (hypercorrection, strength `hc`)           [Metcalfe 2017]
  - two feedback arms: "scalar" (right/wrong only: only a non-specific correction) and "diagnostic" (the critic names
    the error and shows the proof: a type-specific correction and a depth gain, scaled by `diag_gain`)       [process vs outcome feedback: Lightman 2023]
  - dropout: a share of participants leaves before finishing (default 19.4% = share of H-ARC participants who did not
    finish, 1393 of 1729 finished)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .analysis import ERROR_TYPES, classify_error
from .critic import Status
from .episodes import build_world, episode_seed
from .logic import Claim, all_claims, minimal_support
from .metacog import choose_probes
from .world import CURRICULUM

ERR = [e for e in ERROR_TYPES if e != "contradiction"]          # fallacies a person can fall for ("other" = unstructured slips)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ----------------------------------------------------------------------------- the game's items

@dataclass
class ProbeInfo:
    truth: str                 # follows | contradicted | open
    support: int               # premises in the proof (referee core); 0 for open claims
    tempting: str | None       # fallacy type that makes an open claim look true, else None
    key: str = ""              # the claim as the game writes it (Claim.key()), so logs can be emitted


@dataclass
class WorldInfo:
    eid: str
    level: int
    n: int
    targets: list[int]                                 # support size of each target
    errors: dict[str, list[Status]]                    # fallacy type -> statuses of the wrong claims of that type
    probes: list[ProbeInfo] = field(default_factory=list)
    target_keys: list[str] = field(default_factory=list)                  # aligned with `targets`
    error_claims: dict[str, list[tuple[str, str]]] = field(default_factory=dict)   # fallacy type -> (claim key, critic status)


@lru_cache(maxsize=1)
def game_worlds() -> tuple[WorldInfo, ...]:
    out = []
    for cfg in CURRICULUM:
        for k in range(3):
            w = build_world(cfg.level, k)
            sem, P = w.semantics, w.premises
            pairs = sorted((len(minimal_support(sem, P, t)), t.key()) for t in w.targets)
            targets = [x[0] for x in pairs]
            errors: dict[str, list[Status]] = {}
            error_claims: dict[str, list[tuple[str, str]]] = {}
            seen = set()
            for c in all_claims(w.n, w.claim_forms):
                key = c.canonical()
                if c.a == c.b or key in w.premise_set or key in seen or sem.entails(P, c):
                    continue
                seen.add(key)
                status = Status.CONTRADICTED if sem.entails(P, c.negation()) else Status.UNDETERMINED
                et = classify_error(w, c)[0]
                errors.setdefault(et, []).append(status)
                error_claims.setdefault(et, []).append((c.key(), status.value))
            probes = []
            for pr in choose_probes(w, random.Random(episode_seed(cfg.level, k) + 7)):
                claim = Claim.from_key(pr["key"])
                tempting = None
                if pr["truth"] == "open":
                    lab = [x for x in classify_error(w, claim) if x != "other"]
                    tempting = lab[0] if lab else None
                probes.append(ProbeInfo(pr["truth"], len(pr["core"] or []), tempting, pr["key"]))
            out.append(WorldInfo(f"L{cfg.level}-{k + 1}", cfg.level, w.n, targets, errors, probes, [x[1] for x in pairs], error_claims))
    return tuple(out)


# ----------------------------------------------------------------------------- the population (ASSUMPTIONS)

@dataclass
class Population:
    depth_mean: float = 1.4          # reasoning depth at the start (premises per proof)
    depth_sd: float = 0.6
    depth_k: float = 1.4             # steepness of "can I find a proof of this size"
    fallacy_mean: float = -0.6       # mean logit of falling for a tempting fallacy
    fallacy_sd_person: float = 0.8
    fallacy_sd_type: float = 0.6
    tau_mean: float = 0.15           # abstention threshold on evidence strength (log-normal)
    sigma_meta: float = 0.6          # noise of the confidence read-out (log-normal mean)
    lr_sd: float = 0.3               # log-sd of the individual learning-rate multiplier
    n_proposals: int = 8             # attempts per world before the test round
    drop_prob: float = 0.194         # 1 - 1393/1729 (H-ARC completion)
    # learning
    eta_depth: float = 0.025          # depth gain per correct new proof
    rho_generic: float = 0.07        # non-specific lowering of all fallacy logits after any error (both arms)
    rho_specific: float = 0.30       # type-specific lowering of the fallacy logit after an error
    scalar_share: float = 0.35       # share of that correction the scalar arm gets (it sees which claim was wrong, not why)
    eta_diag_depth: float = 0.10     # depth gain from seeing the proof after an error, per unit of (diag_gain - scalar_share)
    probe_learning: float = 0.2      # share of the learning that probe feedback gives
    hc: float = 1.0                  # hypercorrection: correction multiplier is 1 + hc (conf - 3) / 2
    naf_prob: float = 0.0            # negation as failure: probability that a claim the person cannot derive (truth 'open') is judged 'must be false' instead of 'can't tell' (pilot-01: 9 of 18)


@dataclass
class Participant:
    arm: str
    completed: bool
    proposals: list = field(default_factory=list)   # (world, kind, error_type|None, confidence)  kind: "correct" | "error"
    probes: list = field(default_factory=list)      # (world, truth, answer, confidence|None, correct)
    retests: list = field(default_factory=list)     # design option: earlier errors re-presented at the end of the next episode: (confidence at the error, fell for it again)


def _confidence(m: float, rng: np.random.Generator, sigma: float) -> int:
    return int(min(5, max(1, 1 + round(4 * sigmoid(2.0 * (m - 0.8) + rng.normal(0, sigma))))))


def simulate_participant(rng: np.random.Generator, arm: str, pop: Population = Population(), diag_gain: float = 1.0) -> Participant:
    worlds = game_worlds()
    depth = max(0.5, rng.normal(pop.depth_mean, pop.depth_sd))
    f_person = rng.normal(pop.fallacy_mean, pop.fallacy_sd_person)
    f = {e: f_person + rng.normal(0, pop.fallacy_sd_type) for e in ERR}
    f["other"] = -1.5                                           # unstructured slips: not learnable, rare
    tau = pop.tau_mean * math.exp(rng.normal(0, 0.3))
    sigma = pop.sigma_meta * math.exp(rng.normal(0, 0.25))
    lr = math.exp(rng.normal(0, pop.lr_sd))
    drop_at = int(rng.integers(2, len(worlds))) if rng.random() < pop.drop_prob else None
    diag = arm == "diagnostic"
    out = Participant(arm, drop_at is None)

    def learn(correct: bool, etype: str | None, conf: int | None, scale: float) -> None:
        nonlocal depth
        if conf is None:
            return
        h = max(0.2, 1 + pop.hc * (conf - 3) / 2)
        if correct:
            return
        for e in f:
            f[e] -= scale * lr * pop.rho_generic * h
        spec = diag_gain if diag else pop.scalar_share           # diag_gain == scalar_share means 'no benefit of diagnostic feedback'
        if etype in f:
            f[etype] -= scale * lr * pop.rho_specific * spec * h
        if diag:
            depth += scale * lr * pop.eta_diag_depth * max(0.0, diag_gain - pop.scalar_share) * h

    for wi, w in enumerate(worlds):
        if drop_at is not None and wi >= drop_at:
            break
        found: set[int] = set()
        made_errors: list[tuple[str, int]] = []
        remaining = list(range(len(w.targets)))
        for _ in range(pop.n_proposals):
            if not remaining:
                break                                          # every target found: the episode is complete
            made = None
            for ti in remaining:                                 # easiest unfound target first
                s = w.targets[ti]
                if rng.random() < sigmoid(pop.depth_k * (depth - s + 0.25)):
                    made = ti
                    break
            if made is not None:
                s = w.targets[made]
                remaining.remove(made)
                conf = _confidence(1 + 0.5 * (depth - s) + rng.normal(0, 0.4), rng, sigma)
                out.proposals.append((wi, "correct", None, conf, w.target_keys[made], "target"))
                depth += lr * pop.eta_depth * (0.5 if s < depth - 1 else 1.0)
                continue
            types = [e for e in f if w.errors.get(e)]
            if not types:
                continue
            wts = np.array([math.exp(f[e]) for e in types])
            etype = types[int(rng.choice(len(types), p=wts / wts.sum()))]
            conf = _confidence(0.8 + rng.normal(0, 0.4), rng, sigma)
            pool = w.error_claims[etype]
            key, status = pool[(31 * wi + 7 * len(out.proposals)) % len(pool)]            # deterministic pick: no extra random draws
            out.proposals.append((wi, "error", etype, conf, key, status))
            made_errors.append((etype, conf))
            learn(False, etype, conf, 1.0)
        # test round (4 fixed statements), answered before any feedback, then the verdicts are shown
        round_results = []
        for pr in w.probes:
            if pr.truth in ("follows", "contradicted"):
                if rng.random() < sigmoid(pop.depth_k * (depth - pr.support + 0.25)):
                    perceived, m = ("true" if pr.truth == "follows" else "false"), 1 + 0.5 * (depth - pr.support) + rng.normal(0, 0.5)
                else:
                    perceived, m = "open", rng.normal(0.3, 0.4)
            else:
                lure = pr.tempting is not None and rng.random() < sigmoid(f.get(pr.tempting, f["other"]))
                if lure or rng.random() < 0.04:
                    perceived, m = "true", rng.normal(0.9, 0.4)
                else:
                    perceived, m = "open", rng.normal(0.6 + 0.25 * (depth - 2) - 0.08 * (w.n - 4), 0.5)
                    if pop.naf_prob > 0 and rng.random() < pop.naf_prob:
                        perceived = "false"
            if m < tau:
                answer, conf = "unknown", None
            else:
                answer, conf = perceived, _confidence(m, rng, sigma)
            truth_ans = {"follows": "true", "contradicted": "false", "open": "open"}[pr.truth]
            correct = answer == truth_ans
            out.probes.append((wi, pr.truth, answer, conf, correct, pr.key))
            round_results.append((pr, answer, conf, correct))
        for pr, answer, conf, correct in round_results:
            if answer != "unknown" and not correct:
                learn(False, pr.tempting, conf, pop.probe_learning)
        for etype, conf in made_errors:                          # a re-presentation of each error type, judged after this world's learning
            out.retests.append((conf, bool(rng.random() < sigmoid(f[etype]))))
    return out


# ----------------------------------------------------------------------------- endpoints

def auroc(score: np.ndarray, right: np.ndarray) -> float:
    pos, neg = score[right == 1], score[right == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # mid-ranks (ties averaged)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    r = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def endpoints(p: Participant) -> dict:
    """Per-participant endpoints (the quantities a real log gives)."""
    pr = p.probes
    last = [x for x in pr if x[0] >= 12]                         # the Transfer level (3 worlds x 4 probes)
    answered = [x for x in pr if x[3] is not None]
    conf = np.array([x[3] for x in answered], dtype=float)
    right = np.array([x[4] for x in answered], dtype=float)
    # repeats: consecutive errors inside a world; did the next error have the same type as this one?
    rep = {"low": [], "high": []}
    errs_by_world: dict[int, list] = {}
    for (wi, kind, et, c, *_) in p.proposals:
        if kind == "error":
            errs_by_world.setdefault(wi, []).append((et, c))
    for lst in errs_by_world.values():
        for (e1, c1), (e2, _) in zip(lst, lst[1:]):
            if c1 <= 2:
                rep["low"].append(e1 == e2)
            elif c1 >= 4:
                rep["high"].append(e1 == e2)
    rt = {"low": [x[1] for x in p.retests if x[0] <= 2], "high": [x[1] for x in p.retests if x[0] >= 4]}
    return {
        "retest_low": float(np.mean(rt["low"])) if rt["low"] else float("nan"),
        "retest_high": float(np.mean(rt["high"])) if rt["high"] else float("nan"),
        "n_rt_low": len(rt["low"]), "n_rt_high": len(rt["high"]),
        "completed": p.completed,
        "acc_final": float(np.mean([x[4] for x in last])) if len(last) == 12 else float("nan"),
        "acc_all": float(np.mean([x[4] for x in pr])) if pr else float("nan"),
        "abstain": float(np.mean([x[2] == "unknown" for x in pr])) if pr else float("nan"),
        "auroc": auroc(conf, right) if len(answered) >= 8 else float("nan"),
        "n_errors": sum(1 for x in p.proposals if x[1] == "error"),
        "repeat_low": float(np.mean(rep["low"])) if rep["low"] else float("nan"),
        "repeat_high": float(np.mean(rep["high"])) if rep["high"] else float("nan"),
        "n_low": len(rep["low"]), "n_high": len(rep["high"]),
    }

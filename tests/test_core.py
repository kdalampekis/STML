import random

import pytest

from stml.agents import OracleAgent, RandomAgent, run_episode
from stml.analysis import classify_error
from stml.critic import SocraticCritic, Status
from stml.env import STMLEnv
from stml.episodes import build_world, episode_payload, world_for_episode
from stml.logic import Claim, Form, Semantics, Verdict, all_claims, minimal_support
from stml.proof import derive, proof_steps
from stml.trajectories import summary, to_trajectories
from stml.world import CURRICULUM, generate_world

A, B, C, D = 0, 1, 2, 3
ALL, NO, SOME, SNOT = Form.ALL, Form.NO, Form.SOME, Form.SOME_NOT


# --- semantics ---------------------------------------------------------------

def test_classic_syllogisms():
    sem = Semantics(3)
    # Barbara
    assert sem.entails([Claim(ALL, A, B), Claim(ALL, B, C)], Claim(ALL, A, C))
    # Celarent
    assert sem.entails([Claim(ALL, A, B), Claim(NO, B, C)], Claim(NO, A, C))
    # Darii
    assert sem.entails([Claim(SOME, A, B), Claim(ALL, B, C)], Claim(SOME, A, C))
    # Ferio
    assert sem.entails([Claim(SOME, A, B), Claim(NO, B, C)], Claim(SOME_NOT := SNOT, A, C))


def test_classic_fallacies_do_not_follow():
    sem = Semantics(3)
    # conversion of 'all'
    assert sem.evaluate([Claim(ALL, A, B)], Claim(ALL, B, A)) == Verdict.UNDETERMINED
    # undistributed middle
    assert not sem.entails([Claim(ALL, A, C), Claim(ALL, B, C)], Claim(ALL, A, B))
    # two 'some'
    assert not sem.entails([Claim(SOME, A, B), Claim(SOME, B, C)], Claim(SOME, A, C))
    # two negatives
    assert not sem.entails([Claim(NO, A, B), Claim(NO, B, C)], Claim(NO, A, C))


def test_existential_import():
    sem = Semantics(2)
    assert sem.entails([Claim(ALL, A, B)], Claim(SOME, A, B))  # subalternation
    assert not sem.satisfiable([Claim(NO, A, A)])
    assert Semantics(2, existential_import=False).satisfiable([Claim(NO, A, A)])


def test_negation_and_contradiction():
    sem = Semantics(2)
    for c in all_claims(2):
        if sem.satisfiable([c]):
            assert sem.evaluate([c], c.negation()) == Verdict.CONTRADICTED


def test_models_and_minimal_support():
    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(2, 5)
        sem = Semantics(n)
        prem = [rng.choice(all_claims(n)) for _ in range(rng.randint(1, 5))]
        if not sem.satisfiable(prem):
            assert sem.find_model(prem) is None
            continue
        model = sem.find_model(prem)
        assert sem.is_model(prem, model)
        for c in all_claims(n):
            if sem.entails(prem, c):
                sup = minimal_support(sem, prem, c)
                assert sem.entails([prem[i] for i in sup], c)
                for i in sup:  # minimality
                    assert not sem.entails([prem[j] for j in sup if j != i], c)


# --- proofs -------------------------------------------------------------------

def test_rules_are_sound_and_complete():
    rng = random.Random(0)
    for _ in range(1500):
        n = rng.randint(2, 5)
        sem = Semantics(n)
        prem = [rng.choice(all_claims(n)) for _ in range(rng.randint(1, 6))]
        if not sem.satisfiable(prem):
            continue
        clo = derive(prem, n)
        for c in all_claims(n):
            entailed = sem.entails(prem, c)
            if c in clo:
                assert entailed, f"unsound: {prem} => {c}"
            else:
                assert not entailed, f"incomplete: {prem} => {c}"


def test_proof_steps_are_ordered():
    prem = [Claim(ALL, A, B), Claim(ALL, B, C), Claim(NO, C, D)]
    clo = derive(prem, 4)
    steps = proof_steps(Claim(NO, A, D), clo)
    seen = set()
    for s in steps:
        assert all(p in seen for p in s.parents)
        seen.add(s.claim)
    assert steps[-1].claim == Claim(NO, A, D)


# --- worlds and critic --------------------------------------------------------------

@pytest.mark.parametrize("cfg", CURRICULUM, ids=lambda c: f"L{c.level}")
def test_worlds_fit_their_level(cfg):
    for seed in range(5):
        w = generate_world(cfg, seed)
        assert w.semantics.is_model(w.premises, w.model)
        assert cfg.min_targets <= len(w.targets) <= cfg.max_targets
        assert all(t.form in cfg.claim_forms for t in w.targets)
        assert generate_world(cfg, seed).premises == w.premises  # deterministic


def test_critic_statuses_are_consistent():
    for cfg in CURRICULUM:
        w = generate_world(cfg, 7)
        critic = SocraticCritic(w)
        for key, c in critic.critique_all().items():
            claim = Claim.from_key(key)
            if c.status == Status.TARGET:
                assert claim.canonical() in w.targets
                assert c.proof and c.proof[-1]["claim"] == claim.key()
            if c.status == Status.UNDETERMINED:
                people = c.countermodel
                regions = [sum(1 << k for k in p["kinds"]) for p in people]
                sem = w.semantics
                assert sem.is_model(w.premises, regions)
                assert not sem.holds(claim, regions)
                if claim.is_universal:
                    assert c.witness is not None
            if c.status == Status.CONTRADICTED:
                assert c.proof[-1]["claim"] == claim.negation().key()
            assert c.question


def test_premise_reversal_is_recognised():
    w = build_world(3, 0)
    critic = SocraticCritic(w)
    for p in w.premises:
        if p.form in (NO, SOME):
            rev = Claim(p.form, p.b, p.a)
            assert critic.critique(rev).status == Status.PREMISE


def test_error_classification():
    from stml.world import World
    w = World(1, 0, ["wugs", "daxes", "feps"], [None, 0, 0], [1, 3, 5],
              [Claim(ALL, 1, 0), Claim(ALL, 2, 0)], (ALL,))
    assert "illicit_conversion" in classify_error(w, Claim(ALL, 0, 1))
    assert "undistributed_middle" in classify_error(w, Claim(ALL, 1, 2))
    assert classify_error(w, Claim(ALL, 1, 0)) == []


# --- environment and logs -------------------------------------------------------------

def test_env_oracle_beats_random():
    for level in (1, 3, 5):
        oracle = run_episode(STMLEnv(level), OracleAgent(), seed=3)
        rand = run_episode(STMLEnv(level, max_steps=30), RandomAgent(0), seed=3)
        assert oracle["found"] == oracle["targets"]
        assert oracle["return"] > rand["return"]


def test_scalar_feedback_hides_critique():
    env = STMLEnv(2, feedback="scalar")
    env.reset(0)
    _, _, _, info = env.step(env.actions[1])
    assert "critique" not in info


def test_episode_payload_and_log_roundtrip():
    payload = episode_payload(2, 0)
    assert len(payload["critiques"]) == len(payload["names"]) ** 2 * len(payload["claim_forms"])
    w = world_for_episode(payload["id"])
    wrong = next(k for k, c in payload["critiques"].items() if c["status"] == "undetermined")
    right = payload["targets"][0]
    events = [
        {"type": "episode_start", "t": 0, "episode": payload["id"]},
        {"type": "claim", "t": 1000, "episode": payload["id"], "claim": wrong, "confidence": 4,
         "used": [0], "why": "guess", "status": "undetermined", "rt_ms": 1000},
        {"type": "reveal", "t": 2000, "episode": payload["id"], "claim": wrong},
        {"type": "claim", "t": 5000, "episode": payload["id"], "claim": right, "confidence": 5,
         "used": [], "why": "", "status": "target", "rt_ms": 3000},
        {"type": "episode_end", "t": 6000, "episode": payload["id"], "reason": "skip",
         "found": 1, "total": len(w.targets), "reflection": "learned direction matters"},
    ]
    trajs = to_trajectories(events)
    assert len(trajs) == 1 and len(trajs[0].steps) == 2
    assert trajs[0].steps[0].revealed and trajs[0].steps[0].errors
    row = summary(trajs)[0]
    assert row["precision"] == 0.5 and row["reveals"] == 1


# --- SAT referee and metacognition ---------------------------------------------

def test_referee_agrees_with_semantics():
    from stml.referee import referee
    status = {Verdict.ENTAILED: "follows", Verdict.CONTRADICTED: "contradicted", Verdict.UNDETERMINED: "open"}
    rng = random.Random(11)
    for _ in range(150):
        n = rng.randint(2, 5)
        sem = Semantics(n)
        prem = [rng.choice(all_claims(n)) for _ in range(rng.randint(1, 5))]
        if not sem.satisfiable(prem):
            continue
        for c in all_claims(n):
            cert = referee(n, prem, c)
            assert cert.status == status[sem.evaluate(prem, c)]
            if cert.status == "open":
                regions = [sum(1 << k for k in p) for p in cert.countermodel]
                assert sem.is_model(prem, regions) and not sem.holds(c, regions)
            else:
                target = c if cert.status == "follows" else c.negation()
                assert sem.entails([prem[i] for i in cert.core], target)


def test_referee_scales_past_region_method():
    from stml.referee import referee
    n = 40
    prem = [Claim(ALL, i, i + 1) for i in range(n - 1)]
    assert referee(n, prem, Claim(ALL, 0, n - 1)).status == "follows"
    assert referee(n, prem, Claim(ALL, n - 1, 0)).status == "open"


def test_honesty_rule_and_metrics():
    from stml.metacog import Item, commit_threshold, honesty_reward, metrics
    assert honesty_reward("unknown", "open", lam=4, k=0.2) == -0.2
    assert honesty_reward("open", "open") == 1.0 and honesty_reward("true", "open", lam=4) == -4
    assert abs(commit_threshold(4, 0.2) - 0.76) < 1e-9
    m = metrics([Item("true", "follows", 1.0), Item("false", "open", 0.5), Item("unknown", "open")])
    assert m["coverage"] == pytest.approx(2 / 3) and m["selective_accuracy"] == 0.5 and m["abstain_rate"] == pytest.approx(1 / 3)


def test_probes_are_certified_and_varied():
    for level in range(1, 6):
        p = episode_payload(level, 0)
        assert len(p["probes"]) == 4
        truths = {x["truth"] for x in p["probes"]}
        assert "follows" in truths and "open" in truths
        assert all(x["key"] in p["critiques"] for x in p["probes"])


def test_env_judge():
    env = STMLEnv(3)
    env.reset(0)
    t = env.world.targets[0]
    r, info = env.judge(t, "true", lam=2)
    assert info["truth"] == "follows" and r == 1.0
    r, _ = env.judge(t, "unknown", k=0.3)
    assert r == -0.3


def test_probe_events_in_trajectories():
    payload = episode_payload(2, 0)
    eid = payload["id"]
    pr = payload["probes"][0]
    events = [{"type": "episode_start", "t": 0, "episode": eid},
              {"type": "probe", "t": 10, "episode": eid, "claim": pr["key"], "answer": "unknown",
               "confidence": None, "truth": pr["truth"], "rt_ms": 900},
              {"type": "episode_end", "t": 20, "episode": eid, "reason": "skip", "found": 0, "total": 3}]
    row = summary(to_trajectories(events))[0]
    assert row["probe"]["abstain_rate"] == 1.0

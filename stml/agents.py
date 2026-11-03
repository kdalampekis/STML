"""Reference agents: floor and ceiling for the environment.

RandomAgent   proposes untried claims at random (the floor).
OracleAgent   proposes exactly the targets (the ceiling; uses hidden info).

Learners (the cognitive learner, the imitation model) sit between these.
"""

from __future__ import annotations

import random

from .env import Observation, STMLEnv
from .logic import Claim


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, env: STMLEnv, obs: Observation) -> Claim:
        tried = {c for c, _ in obs.history}
        options = [c for c in env.actions if c not in tried and not c.is_trivial()]
        return self.rng.choice(options or env.actions)


class OracleAgent:
    def act(self, env: STMLEnv, obs: Observation) -> Claim:
        for t in env.world.targets:
            if t not in obs.found:
                return t
        return env.actions[0]


def run_episode(env: STMLEnv, agent, seed: int = 0) -> dict:
    obs = env.reset(seed)
    total, done = 0.0, False
    while not done:
        obs, r, done, _ = env.step(agent.act(env, obs))
        total += r
    return {"return": round(total, 3), "steps": env.steps,
            "found": len(obs.found), "targets": obs.n_targets}

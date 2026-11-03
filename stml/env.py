"""A Gym-style environment over the same worlds people play.

An agent proposes claims; the environment answers with a reward and the
Socratic critique. Two feedback modes support the H2 comparison:

  scalar      reward only; the critique in `info` is reduced to correct/wrong
  diagnostic  reward plus the full structured critique (support premises,
              proof rules, countermodel)

No gym dependency: reset() -> obs, step(claim) -> (obs, reward, done, info).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .critic import Critique, SocraticCritic, Status
from .logic import Claim, all_claims
from .metacog import ANSWERS, honesty_reward
from .referee import referee
from .world import World, generate_world, level_config

REWARDS = {
    Status.TARGET: 1.0,
    Status.WEAK: 0.2,
    Status.PREMISE: 0.0,
    Status.TRIVIAL: 0.0,
    Status.CONTRADICTED: -1.0,
    Status.UNDETERMINED: -1.0,
}
STEP_COST = 0.05
FEEDBACK_MODES = ("scalar", "diagnostic")


@dataclass
class Observation:
    names: list[str]
    premises: list[Claim]
    claim_forms: tuple
    found: set[Claim] = field(default_factory=set)
    history: list[tuple[Claim, Status]] = field(default_factory=list)
    n_targets: int = 0


class STMLEnv:
    def __init__(self, level: int = 1, feedback: str = "diagnostic", max_steps: int = 60):
        if feedback not in FEEDBACK_MODES:
            raise ValueError(f"feedback must be one of {FEEDBACK_MODES}")
        self.cfg = level_config(level)
        self.feedback = feedback
        self.max_steps = max_steps
        self.world: World | None = None
        self.critic: SocraticCritic | None = None

    # action space helpers -------------------------------------------------
    @property
    def actions(self) -> list[Claim]:
        assert self.world is not None, "call reset() first"
        return all_claims(self.world.n, self.world.claim_forms)

    def reset(self, seed: int = 0, world: World | None = None) -> Observation:
        self.world = world or generate_world(self.cfg, seed)
        self.critic = SocraticCritic(self.world)
        self.targets = set(self.world.targets)
        self.steps = 0
        self.obs = Observation(self.world.names, list(self.world.premises), self.world.claim_forms,
                               n_targets=len(self.targets))
        return self.obs

    def step(self, claim: Claim) -> tuple[Observation, float, bool, dict]:
        assert self.critic is not None, "call reset() first"
        crit: Critique = self.critic.critique(claim)
        status = crit.status
        canon = claim.canonical()
        if status == Status.TARGET and canon in self.obs.found:
            reward = 0.0  # repeats earn nothing
        else:
            reward = REWARDS[status]
        if status == Status.TARGET:
            self.obs.found.add(canon)
        self.obs.history.append((claim, status))
        self.steps += 1
        reward -= STEP_COST
        done = self.obs.found >= self.targets or self.steps >= self.max_steps
        if self.feedback == "scalar":
            info = {"correct": status.correct}
        else:
            info = {"correct": status.correct, "critique": crit}
        return self.obs, reward, done, info

    # -- H3: judging a claim with one of four epistemic answers -------------
    def judge(self, claim: Claim, answer: str, lam: float = 1.0, k: float = 0.2) -> tuple[float, dict]:
        """Score an epistemic answer ('true' | 'false' | 'open' | 'unknown').

        Truth comes from the SAT referee. The learner only gets the reward and,
        in diagnostic mode, the critique; it never sees the referee itself.
        """
        assert self.world is not None, "call reset() first"
        if answer not in ANSWERS:
            raise ValueError(f"answer must be one of {ANSWERS}")
        cert = referee(self.world.n, self.world.premises, claim)
        reward = honesty_reward(answer, cert.status, lam, k)
        info = {"truth": cert.status, "difficulty": cert.conflicts}
        if self.feedback == "diagnostic":
            info["critique"] = self.critic.critique(claim)
        return reward, info

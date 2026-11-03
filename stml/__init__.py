"""STML: Socratic Topological Meta-Learning, Phase 1 core.

Modules
  logic        claims, exact region semantics, entailment, models
  proof        named-rule derivations (Barbara, Celarent, ...)
  world        hidden worlds and the level curriculum
  critic       the Socratic critic: verdicts, proofs, countermodels, questions
  env          Gym-style environment (scalar vs diagnostic feedback)
  analysis     classify wrong claims into classical reasoning errors
  episodes     fixed episode sets shared by the game and agents
  trajectories load human game logs into learning trajectories
  agents       random and oracle reference agents
"""

from .critic import Critique, SocraticCritic, Status
from .logic import Claim, Form, Semantics, Verdict
from .world import CURRICULUM, World, generate_world, level_config

__version__ = "0.1.0"
__all__ = ["Claim", "Form", "Semantics", "Verdict", "SocraticCritic", "Critique", "Status",
           "World", "generate_world", "level_config", "CURRICULUM"]

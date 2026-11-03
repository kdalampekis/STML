"""Fixed episode sets shared by the human game and the agents.

Each episode is a (level, seed) pair, so a world can always be rebuilt
exactly from a log. The game embeds the precomputed critique for every
possible claim, so Python stays the single source of truth for what is
correct and why.
"""

from __future__ import annotations

import random

from .critic import SocraticCritic
from .metacog import ANSWERS, choose_probes
from .logic import Claim
from .proof import RULES
from .world import CURRICULUM, World, generate_world, level_config

GAME_VERSION = "pilot-0.2"
EPISODES_PER_LEVEL = 3


def episode_id(level: int, k: int) -> str:
    return f"L{level}-{k + 1}"


def episode_seed(level: int, k: int) -> int:
    return 1000 * level + k


def build_world(level: int, k: int) -> World:
    return generate_world(level_config(level), episode_seed(level, k))


def world_for_episode(eid: str) -> World:
    level, k = eid[1:].split("-")
    return build_world(int(level), int(k) - 1)


def episode_payload(level: int, k: int) -> dict:
    world = build_world(level, k)
    critic = SocraticCritic(world)
    critiques = {key: c.to_dict() for key, c in critic.critique_all().items()}
    cfg = level_config(level)
    return {
        "id": episode_id(level, k),
        "level": level,
        "seed": episode_seed(level, k),
        "title": cfg.title,
        "lesson": cfg.lesson,
        "names": world.names,
        "premises": [{"key": p.key(), "text": p.text(world.names)} for p in world.premises],
        "claim_forms": [f.value for f in world.claim_forms],
        "targets": [t.key() for t in world.targets],
        "critiques": critiques,
        "probes": choose_probes(world, random.Random(episode_seed(level, k) + 7)),
    }


def game_payload(episodes_per_level: int = EPISODES_PER_LEVEL) -> dict:
    return {
        "version": GAME_VERSION,
        "answers": list(ANSWERS),
        "rules": RULES,
        "levels": [
            {"level": c.level, "title": c.title, "lesson": c.lesson,
             "claim_forms": [f.value for f in c.claim_forms]}
            for c in CURRICULUM
        ],
        "episodes": [episode_payload(c.level, k)
                     for c in CURRICULUM for k in range(episodes_per_level)],
    }


__all__ = ["Claim", "GAME_VERSION", "build_world", "episode_payload", "game_payload",
           "world_for_episode", "episode_id", "episode_seed"]

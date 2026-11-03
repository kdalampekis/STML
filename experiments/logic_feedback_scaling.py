"""What limits a learner that learns from feedback? One-factor-at-a-time sweep around the default hidden-premise setting.

Default: 300 training worlds per level (levels 2-5), 4 GNN layers, 10 claims per episode, 50% of the premises visible.
Factors: training worlds {100, 300, 1000}; GNN layers {2, 4, 6}; claims per episode {5, 10, 20}; visible fraction {0.3, 0.5, 0.7}.
For each setting: a static learner (no feedback) and a feedback learner (verdicts revealed and added to the graph). The feedback gain is
fb - static; 'gap closed' is the gain divided by the gap between static and a perfect learner (100%).
PREDICTIONS, written before running: the feedback learner is data-limited (more worlds raise its gain), not depth-limited; more claims per
episode raise the gain at late positions; fewer visible premises raise the gain but lower the absolute accuracy.

    python experiments/logic_feedback_scaling.py --quick
    python experiments/logic_feedback_scaling.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stml.model.logic_data as ld  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, make_episodes  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

DEFAULT = {"worlds": 300, "layers": 4, "k": 10, "vis": 0.5}
SWEEP = [("default", {})] + [(f"{f}={v}", {f: v}) for f, vs in (("worlds", (100, 1000)), ("layers", (2, 6)), ("k", (5, 20)), ("vis", (0.3, 0.7))) for v in vs]


def run_setting(cfg: dict, seed: int, device, epochs: int, quick: bool) -> dict:
    nt, nv = (cfg["worlds"], 50) if not quick else (24, 8)
    levels = [2, 3, 4, 5]
    tr = make_episodes(levels, range(nt), cfg["k"], cfg["vis"], rng_seed=1)
    va = make_episodes(levels, range(nt, nt + nv), cfg["k"], cfg["vis"], rng_seed=2)
    te = make_episodes(levels, range(1000, 1000 + (100 if not quick else 10)), cfg["k"], cfg["vis"], rng_seed=3)
    out = {}
    for mode in ("static", "fb"):
        torch.manual_seed(seed)
        m = LogicJudge(64, cfg["layers"], n_types=N_TYPES[mode]).to(device)
        fit(m, episode_items(tr, mode), episode_items(va, mode), device, epochs=epochs, seed=seed, log=lambda s: None)
        items = episode_items(te, mode)
        y = np.array([it.label for it in items])
        tt = np.array([it.t for it in items])
        right = (logits_of(m, items, device).argmax(1).cpu().numpy() == y).astype(float)
        out[mode] = {"all": float(right.mean()), "early": float(right[tt < cfg["k"] / 3].mean()), "late": float(right[tt >= 2 * cfg["k"] / 3].mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    epochs = 2 if a.quick else 10
    if a.quick:
        a.seeds = 1
    device = pick_device(None)
    with Run("logic_feedback_scaling" + ("_quick" if a.quick else ""), {**vars(a), "sweep": [s for s, _ in SWEEP]}) as run:
        res = {}
        for name, change in SWEEP:
            cfg = {**DEFAULT, **change}
            res[name] = [run_setting(cfg, s, device, epochs, a.quick) for s in range(a.seeds)]
            r = res[name]
            run.log(f"{name}: static {np.mean([x['static']['all'] for x in r]):.1%}, fb {np.mean([x['fb']['all'] for x in r]):.1%}")
        L = ["# What limits learning from feedback? (hidden-premise logic episodes)", "", f"{a.seeds} seeds; accuracy on unseen worlds; gain = feedback minus static; "
             "'late' = last third of the claims in an episode, 'early' = first third; gap closed = gain / (100% - static).", "",
             "| setting | static | feedback | gain (all) | gain late | gain early | gap closed (late) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, _ in SWEEP:
            r = res[name]
            g = lambda mode, k: np.array([x[mode][k] for x in r])
            gl, ge, ga = g("fb", "late") - g("static", "late"), g("fb", "early") - g("static", "early"), g("fb", "all") - g("static", "all")
            closed = gl / (1 - g("static", "late"))
            L.append(f"| {name} | {g('static', 'all').mean():.1%} | {g('fb', 'all').mean():.1%} | {100 * ga.mean():+.1f} ({100 * ga.std():.1f}) | "
                     f"{100 * gl.mean():+.1f} ({100 * gl.std():.1f}) | {100 * ge.mean():+.1f} ({100 * ge.std():.1f}) | {closed.mean():.0%} |")
        run.finish(res, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

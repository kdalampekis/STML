"""Active querying, deeper (H2): how robust is the advantage of uncertainty-based asking?

Baseline (logic_active_query): asking by predictive entropy beat asking at random by +5.3 points (budget 6, pool 12). Here, one factor at a time, same 3 seeds:
  budget {3, 6, 10}; pool size {8, 12, 20}; model configuration {default, wide-long (the H1 validation winner), shallow (2 layers)};
  on-policy training: the feedback model retrained on episodes whose claim order follows the entropy policy (a DAgger-style round), against a CONTROL trained on the same
  amount of data in random order (so any gain is not just more data).
Metric: accuracy on 8 held-out claims after the asks; gain = entropy minus random with the same model, same seed.

PREDICTIONS, written before running:
  D1  the entropy gain is positive at every budget, >= 2 points already at budget 3, and does not shrink from 6 to 10 (more asks, same pool: it may saturate).
  D2  the gain grows with the pool size (more to choose from): gain(pool 20) >= gain(pool 12) >= gain(pool 8) >= 1 point.
  D3  the gain is >= 3 points with the wide-long and with the shallow model (it is not an artefact of one configuration).
  D4  on-policy training does not hurt: entropy-policy accuracy with the on-policy model is >= that of the equal-data random-order control; a drop would mean distribution shift.
  Falsifier of D3: the gain is within the seed sd of zero for either alternative model.

    python experiments/logic_active_query_deeper.py --quick
    python experiments/logic_active_query_deeper.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import stml.model.logic_data as ld  # noqa: E402
from logic_active_query import softmax  # noqa: E402
from logic_active_query_mechanism import run as run_policy  # noqa: E402
from stml.model.logic_episodes import Episode, N_TYPES, episode_items, make_episodes, query_after_asks  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

DEFAULT = {"lr": 1e-3, "d": 64, "layers": 4, "epochs": 10, "wd": 1e-4, "batch": 128}
MODELS = {"default": DEFAULT, "wide-long": {"lr": 1.5e-3, "d": 128, "layers": 4, "epochs": 16, "wd": 1.5e-4, "batch": 256},
          "shallow": {**DEFAULT, "layers": 2}}
HELD = 8


def train(cfg, tr, va, device, seed):
    torch.manual_seed(seed)
    m = LogicJudge(cfg["d"], cfg["layers"], n_types=N_TYPES["fb"]).to(device)
    fit(m, episode_items(tr, "fb"), episode_items(va, "fb"), device, epochs=cfg["epochs"], lr=cfg["lr"], batch=cfg["batch"], seed=seed,
        weight_decay=cfg["wd"], log=lambda s: None)
    return m


def reorder(ep: Episode, order: list[int]) -> Episode:
    rest = [i for i in range(len(ep.queries)) if i not in order]
    idx = order + rest
    return Episode(ep.level, ep.seed, ep.n, ep.visible, [ep.queries[i] for i in idx], [ep.labels[i] for i in idx], ep.full)


def entropy_orders(model, eps, pool_n, budget, device) -> list[list[int]]:
    asked = [[] for _ in eps]
    for _ in range(budget):
        cand = [[c for c in range(min(pool_n, len(ep.queries))) if c not in asked[i]] for i, ep in enumerate(eps)]
        cit = [query_after_asks(ep, asked[i], c) for i, ep in enumerate(eps) for c in cand[i]]
        p = softmax(logits_of(model, cit, device).cpu().numpy())
        ent = -(p * np.log(p + 1e-12)).sum(1)
        pos = 0
        for i in range(len(eps)):
            sc = ent[pos:pos + len(cand[i])]
            asked[i].append(cand[i][int(sc.argmax())])
            pos += len(cand[i])
    return asked


def gain(model, te_all, pool, budget, device, seed):
    eps = [e for e in te_all if len(e.queries) >= pool + HELD]
    held = [list(range(pool, pool + HELD)) for _ in eps]
    r = run_policy(model, eps, pool, held, budget, "random", device, random.Random(seed))
    e = run_policy(model, eps, pool, held, budget, "entropy", device, random.Random(seed))
    return {"random": r, "entropy": e, "gain": e - r, "n": len(eps)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw = (1000, 50, 100)
    if a.quick:
        a.seeds, nw = 1, (24, 8, 10)
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_active_query_deeper" + ("_quick" if a.quick else ""), {**vars(a), "models": MODELS}) as run:
        K = 28
        tr = make_episodes(levels, range(nw[0]), K, 0.5, rng_seed=1)
        va = make_episodes(levels, range(nw[0], nw[0] + nw[1]), K, 0.5, rng_seed=2)
        te = make_episodes(levels, range(1000, 1000 + nw[2]), K, 0.5, rng_seed=3)
        R = {}
        for seed in range(a.seeds):
            base = train(DEFAULT, tr, va, device, seed)
            for b in (3, 6, 10):
                R.setdefault(f"budget={b}", []).append(gain(base, te, 12, b, device, seed))
            for pool in (8, 12, 20):
                R.setdefault(f"pool={pool}", []).append(gain(base, te, pool, 6, device, seed))
            for name in ("wide-long", "shallow"):
                R.setdefault(f"model={name}", []).append(gain(train(MODELS[name], tr, va, device, seed), te, 12, 6, device, seed))
            # on-policy round (DAgger style) vs an equal-data random-order control
            sub = tr[::2]
            pol_orders = entropy_orders(base, sub, 12, 6, device)
            rnd = random.Random(seed)
            rand_orders = [rnd.sample(range(min(12, len(e.queries))), 6) for e in sub]
            on = train(DEFAULT, tr + [reorder(e, o) for e, o in zip(sub, pol_orders)], va, device, seed)
            ctl = train(DEFAULT, tr + [reorder(e, o) for e, o in zip(sub, rand_orders)], va, device, seed)
            R.setdefault("on-policy training", []).append(gain(on, te, 12, 6, device, seed))
            R.setdefault("equal-data control", []).append(gain(ctl, te, 12, 6, device, seed))
            run.log(f"seed {seed}: " + ", ".join(f"{k} {100 * R[k][-1]['gain']:+.1f}" for k in R))
        L = ["# Active querying, deeper: how robust is the entropy advantage?", "", f"{a.seeds} seeds; gain = accuracy after the asks with entropy asking minus random asking, same model and seed "
             "(points, mean and sd over seeds); n = test episodes with enough claims. Predictions D1-D4 are in the script.", "",
             "| setting | random | entropy | gain | n episodes |", "| --- | ---: | ---: | ---: | ---: |"]
        for k, v in R.items():
            g = 100 * np.array([x["gain"] for x in v])
            L.append(f"| {k} | {100 * np.mean([x['random'] for x in v]):.1f}% | {100 * np.mean([x['entropy'] for x in v]):.1f}% | {g.mean():+.1f} ({g.std():.1f}) | {v[0]['n']} |")
        d = 100 * (np.array([x["entropy"] for x in R["on-policy training"]]) - np.array([x["entropy"] for x in R["equal-data control"]]))
        L += ["", f"On-policy model minus equal-data control, entropy policy, accuracy: {d.mean():+.1f} points (sd {d.std():.1f})."]
        run.finish(R, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

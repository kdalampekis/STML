"""Why does uncertainty-based asking beat random asking? (follow-up Q1b to logic_active_query)

In logic_active_query, entropy sampling beat random by 5.3 points, and also beat the greedy 'make the most held-out claims provable' oracle.
Hypothesis: the gain comes from asking UNPROVABLE claims (their verdicts are new information), not from asking where the model is unsure per se; a claim that is
already provable from what the learner has seen teaches nothing it could not derive. Same setting, same trained model, same budget (6 asks).
Policies: random; entropy; entropy_provable (max entropy among candidates that are provable now); entropy_unprovable (max entropy among the unprovable ones);
random_provable; random_unprovable. (If a class is empty the policy falls back to all candidates.) 'Provable' uses exact reasoning, so these two are diagnostics, not learners.

PREDICTIONS, written before running:
  B1  entropy_unprovable reaches at least 90% of entropy's gain over random; entropy_provable is not better than random by more than 1 point.
  B2  random_unprovable beats random and random_provable by >= 2 points (asking unprovable claims is what teaches).
  Falsifier of B1: entropy_provable gains >= 2 points over random, or entropy_unprovable gains less than 70% of entropy's gain.

    python experiments/logic_active_query_mechanism.py --quick
    python experiments/logic_active_query_mechanism.py
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
from logic_active_query import _after, softmax  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, knowable, make_episodes, query_after_asks  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

POLICIES = ["random", "entropy", "entropy_provable", "entropy_unprovable", "random_provable", "random_unprovable"]


def run(model, eps, pool_n, heldout, budget, policy, device, rnd) -> float:
    asked = [[] for _ in eps]
    for _ in range(budget):
        cand = [[c for c in range(pool_n) if c not in asked[i]] for i in range(len(eps))]
        prov = [[knowable(_after(ep, asked[i], c), len(asked[i])) for c in cand[i]] for i, ep in enumerate(eps)]
        ent = None
        if "entropy" in policy:
            cit = [query_after_asks(ep, asked[i], c) for i, ep in enumerate(eps) for c in cand[i]]
            p = softmax(logits_of(model, cit, device).cpu().numpy())
            ent = -(p * np.log(p + 1e-12)).sum(1)
        pos = 0
        for i in range(len(eps)):
            idx = list(range(len(cand[i])))
            if policy.endswith("_provable") and not policy.endswith("unprovable"):
                keep = [j for j in idx if prov[i][j]] or idx
            elif policy.endswith("unprovable"):
                keep = [j for j in idx if not prov[i][j]] or idx
            else:
                keep = idx
            if ent is not None:
                sc = ent[pos:pos + len(cand[i])]
                j = max(keep, key=lambda j: sc[j])
            else:
                j = rnd.choice(keep)
            asked[i].append(cand[i][j])
            pos += len(cand[i])
    items = [query_after_asks(ep, asked[i], j) for i, ep in enumerate(eps) for j in heldout[i]]
    pred = logits_of(model, items, device).argmax(1).cpu().numpy()
    return float(np.mean([p == it.label for p, it in zip(pred, items)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    pool_n, n_held, budget, k = 12, 8, 6, 20
    nw, epochs = (1000, 50, 100), 10
    if a.quick:
        a.seeds, nw, epochs = 1, (24, 8, 10), 2
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_active_query_mechanism" + ("_quick" if a.quick else ""), {**vars(a)}) as r:
        tr = make_episodes(levels, range(nw[0]), k, 0.5, rng_seed=1)
        va = make_episodes(levels, range(nw[0], nw[0] + nw[1]), k, 0.5, rng_seed=2)
        te = [e for e in make_episodes(levels, range(1000, 1000 + nw[2]), k, 0.5, rng_seed=3) if len(e.queries) >= pool_n + n_held]
        heldout = [list(range(pool_n, pool_n + n_held)) for _ in te]
        res = {p: [] for p in POLICIES}
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            m = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
            fit(m, episode_items(tr, "fb"), episode_items(va, "fb"), device, epochs=epochs, seed=seed, log=lambda s: None)
            for p in POLICIES:
                res[p].append(run(m, te, pool_n, heldout, budget, p, device, random.Random(seed)))
            r.log(f"seed {seed}: " + ", ".join(f"{p} {res[p][-1]:.1%}" for p in POLICIES))
        base = np.array(res["random"])
        L = ["# Why does uncertainty-based asking win? (provable vs unprovable candidates)", "",
             f"{a.seeds} seeds; {len(te)} test episodes; accuracy on 8 held-out claims after 6 asks; paired difference to random in points (sd). Predictions B1-B2 in the script.", "",
             "| policy | accuracy | difference to random |", "| --- | ---: | ---: |"]
        for p in POLICIES:
            d = 100 * (np.array(res[p]) - base)
            L.append(f"| {p} | {100 * np.mean(res[p]):.1f}% ({100 * np.std(res[p]):.1f}) | {d.mean():+.1f} ({d.std():.1f}) |")
        r.finish(res, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

"""Active querying: does CHOOSING which claim to ask next beat asking at random? (the Socratic question as an information-gain choice; F7)

Hidden-premise episodes (stml/model/logic_episodes.py, 50% of premises visible). Per episode: a pool of 12 candidate claims, a budget of B = 6 asks (the true
verdict of an asked claim is revealed and added to the graph), then the learner judges 8 held-out claims of the same world. ONE feedback model is trained on
random-order episodes and used by every policy, so only the choice of what to ask differs.
Policies: random; entropy (largest predictive entropy of the model on the candidate); margin (smallest gap between the top two classes);
infogain_oracle (an UPPER BOUND, not a learner: greedy choice of the candidate whose true verdict makes the most held-out claims provable, using exact reasoning
and knowledge of the held-out claims).
Metric: accuracy on the held-out claims after b = 0..6 asks, and the share of held-out claims that are provable.

PREDICTIONS, written before running:
  A1  infogain_oracle beats random by >= 2 points at B = 6 (there is something to gain from choosing).
  A2  entropy and margin are NOT better than random by more than the seed sd: the most uncertain claims are mostly unknowable ones, which teach little.
  A3  infogain_oracle makes more held-out claims provable than random.
  Falsifier of A2: entropy or margin beats random by >= 2 points at B = 6.

    python experiments/logic_active_query.py --quick
    python experiments/logic_active_query.py
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

import stml.model.logic_data as ld  # noqa: E402
from stml.model.logic_episodes import (N_TYPES, episode_items, knowable, make_episodes, newly_knowable,  # noqa: E402
                                       query_after_asks)
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

POLICIES = ["random", "entropy", "margin", "infogain_oracle"]


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def provable_share(ep, asked, targets) -> float:
    return float(np.mean([knowable(_after(ep, asked, j), len(asked)) for j in targets]))


def _after(ep, asked, j):
    from stml.model.logic_episodes import Episode
    return Episode(ep.level, ep.seed, ep.n, ep.visible, [ep.queries[i] for i in asked] + [ep.queries[j]], [ep.labels[i] for i in asked] + [ep.labels[j]], ep.full)


def run_policy(model, eps, pool_n, heldout, budget, policy, device, rnd) -> tuple[list[float], list[float]]:
    asked = [[] for _ in eps]
    accs, prov = [], []
    for b in range(budget + 1):
        items = [query_after_asks(ep, asked[i], j) for i, ep in enumerate(eps) for j in heldout[i]]
        pred = logits_of(model, items, device).argmax(1).cpu().numpy()
        accs.append(float(np.mean([p == it.label for p, it in zip(pred, items)])))
        prov.append(float(np.mean([provable_share(ep, asked[i], heldout[i]) for i, ep in enumerate(eps)])))
        if b == budget:
            break
        if policy in ("entropy", "margin"):
            cand = [[c for c in range(pool_n) if c not in asked[i]] for i in range(len(eps))]
            cit = [query_after_asks(ep, asked[i], c) for i, ep in enumerate(eps) for c in cand[i]]
            p = softmax(logits_of(model, cit, device).cpu().numpy())
            score = -(p * np.log(p + 1e-12)).sum(1) if policy == "entropy" else -(np.sort(p, 1)[:, -1] - np.sort(p, 1)[:, -2])
            pos = 0
            for i in range(len(eps)):
                sc = score[pos:pos + len(cand[i])]
                asked[i].append(cand[i][int(sc.argmax())])
                pos += len(cand[i])
        else:
            for i, ep in enumerate(eps):
                cand = [c for c in range(pool_n) if c not in asked[i]]
                if policy == "random":
                    asked[i].append(rnd.choice(cand))
                else:
                    gains = [newly_knowable(ep, asked[i], c, heldout[i]) for c in cand]
                    best = max(gains)
                    asked[i].append(rnd.choice([c for c, g in zip(cand, gains) if g == best]))
    return accs, prov


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
    with Run("logic_active_query" + ("_quick" if a.quick else ""), {**vars(a), "pool": pool_n, "held": n_held, "budget": budget}) as run:
        tr = make_episodes(levels, range(nw[0]), k, 0.5, rng_seed=1)
        va = make_episodes(levels, range(nw[0], nw[0] + nw[1]), k, 0.5, rng_seed=2)
        te = [e for e in make_episodes(levels, range(1000, 1000 + nw[2]), k, 0.5, rng_seed=3) if len(e.queries) >= pool_n + n_held]
        heldout = [list(range(pool_n, pool_n + n_held)) for _ in te]
        run.log(f"test episodes with >= {pool_n + n_held} claims: {len(te)}")
        res = {p: {"acc": [], "prov": []} for p in POLICIES}
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            m = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
            fit(m, episode_items(tr, "fb"), episode_items(va, "fb"), device, epochs=epochs, seed=seed, log=lambda s: None)
            for p in POLICIES:
                acc, prov = run_policy(m, te, pool_n, heldout, budget, p, device, random.Random(seed))
                res[p]["acc"].append(acc)
                res[p]["prov"].append(prov)
            run.log(f"seed {seed}: final accuracy " + ", ".join(f"{p} {res[p]['acc'][-1][-1]:.1%}" for p in POLICIES))
        L = ["# Active querying: choosing what to ask", "", f"{a.seeds} seeds; {len(te)} test episodes; pool {pool_n} claims, {budget} asks, {n_held} held-out claims judged afterwards. "
             "Predictions A1-A3 are in the script docstring.", "",
             "| policy | " + " | ".join(f"after {b} asks" for b in range(budget + 1)) + " |", "| --- | " + " | ".join("---:" for _ in range(budget + 1)) + " |"]
        for p in POLICIES:
            arr = np.array(res[p]["acc"])
            L.append(f"| {p} | " + " | ".join(f"{100 * arr[:, b].mean():.1f}% ({100 * arr[:, b].std():.1f})" for b in range(budget + 1)) + " |")
        L += ["", "Share of the held-out claims that are provable (a proof exists from what has been asked):", "",
              "| policy | " + " | ".join(f"after {b} asks" for b in range(budget + 1)) + " |", "| --- | " + " | ".join("---:" for _ in range(budget + 1)) + " |"]
        for p in POLICIES:
            arr = np.array(res[p]["prov"])
            L.append(f"| {p} | " + " | ".join(f"{100 * arr[:, b].mean():.1f}%" for b in range(budget + 1)) + " |")
        L += ["", "Paired difference to random at the last ask (points, mean over seeds; sd of the paired difference):", ""]
        for p in POLICIES[1:]:
            d = 100 * (np.array(res[p]["acc"])[:, -1] - np.array(res["random"]["acc"])[:, -1])
            L.append(f"- {p}: {d.mean():+.1f} ({d.std():.1f})")
        run.finish(res, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

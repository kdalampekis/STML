"""Q10: can a LEARNED asking policy (REINFORCE, Stage-B style) beat the hand-made 'ask where the model is most uncertain' rule?

Setting as in logic_active_query (hidden premises, pool of 12 claims, B = 6 asks, 8 held-out claims judged afterwards); ONE frozen feedback judge (trained on random-order
episodes). The policy is a small MLP scoring each unasked candidate from features; it samples one candidate per step (softmax over scores), the reward is the judge's accuracy on
the 8 held-out claims after the 6 asks, baseline = batch mean, entropy bonus 0.01. It is trained on episodes of worlds the judge never saw and tested greedily on the test worlds.
Policies: random; entropy (hand rule); learned_full (features: judge's class probabilities, entropy, margin, form, kind-overlap with already-asked claims, step);
learned_structure (same without the judge's probabilities, entropy and margin: structure features only, the control for 'what the policy learns from').
PREDICTIONS, written before running (accuracy after 6 asks, points):
  L1  learned_full is within +-2 of entropy (entropy is already near the best use of the judge's uncertainty; the exact information-gain oracle was WORSE than entropy in Q1).
  L2  learned_structure is below entropy by >= 2 (without the judge's uncertainty there is little to choose by) and not below random by more than the seed sd.
  Falsifier of L1: learned_full beats entropy by >= 2 and by > 2 seed sd (the hand rule is suboptimal and RL finds a better one).

    python experiments/logic_learned_asking.py --quick
    python experiments/logic_learned_asking.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stml.model.logic_data as ld  # noqa: E402
from stml.logic import FORM_ORDER  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, make_episodes, query_after_asks  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

POOL, HELD, BUDGET = 12, 8, 6
NF = len(FORM_ORDER)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def features(model, eps, asked, device, use_unc: bool) -> np.ndarray:
    """[E, POOL, F] candidate features given the claims asked so far."""
    items = [query_after_asks(ep, asked[i], c) for i, ep in enumerate(eps) for c in range(POOL)]
    p = softmax(logits_of(model, items, device).cpu().numpy()).reshape(len(eps), POOL, 3)
    ent = -(p * np.log(p + 1e-12)).sum(2, keepdims=True)
    srt = np.sort(p, 2)
    mar = (srt[:, :, -1] - srt[:, :, -2])[:, :, None]
    form = np.zeros((len(eps), POOL, NF))
    ovl = np.zeros((len(eps), POOL, 1))
    for i, ep in enumerate(eps):
        seen = {k for j in asked[i] for k in (ep.queries[j].a, ep.queries[j].b)}
        for c in range(POOL):
            q = ep.queries[c]
            form[i, c, FORM_ORDER.index(q.form)] = 1
            ovl[i, c, 0] = ((q.a in seen) + (q.b in seen)) / 2
    step = np.full((len(eps), POOL, 1), len(asked[0]) / BUDGET)
    parts = [form, ovl, step] + ([p, ent, mar] if use_unc else [])
    return np.concatenate(parts, 2).astype(np.float32)


def heldout_acc(model, eps, asked, device) -> np.ndarray:
    items = [query_after_asks(ep, asked[i], j) for i, ep in enumerate(eps) for j in range(POOL, POOL + HELD)]
    pred = logits_of(model, items, device).argmax(1).cpu().numpy()
    y = np.array([it.label for it in items])
    return (pred == y).reshape(len(eps), HELD).mean(1)


def rollout(model, eps, device, policy, use_unc=True, sample=False, rnd=None, net=None):
    asked = [[] for _ in eps]
    logps = []
    ents = []
    for b in range(BUDGET):
        if policy == "random":
            for i in range(len(eps)):
                asked[i].append(rnd.choice([c for c in range(POOL) if c not in asked[i]]))
            continue
        mask = torch.zeros(len(eps), POOL, dtype=torch.bool)
        for i in range(len(eps)):
            mask[i, asked[i]] = True
        if policy == "entropy":
            p = softmax(logits_of(model, [query_after_asks(ep, asked[i], c) for i, ep in enumerate(eps) for c in range(POOL)], device).cpu().numpy()).reshape(len(eps), POOL, 3)
            sc = torch.tensor(-(p * np.log(p + 1e-12)).sum(2)).masked_fill(mask, -1e9)
            pick = sc.argmax(1)
        else:
            f = features(model, eps, asked, device, use_unc)
            sc = net(torch.tensor(f)).squeeze(-1).masked_fill(mask, -1e9)
            dist = torch.distributions.Categorical(logits=sc)
            pick = dist.sample() if sample else sc.argmax(1)
            if sample:
                logps.append(dist.log_prob(pick))
                ents.append(dist.entropy())
        for i in range(len(eps)):
            asked[i].append(int(pick[i]))
    return asked, logps, ents


def train_policy(model, eps, device, use_unc, seed, iters, bs, log):
    torch.manual_seed(seed)
    nf = NF + 2 + (5 if use_unc else 0)
    net = nn.Sequential(nn.Linear(nf, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    rnd = np.random.RandomState(seed)
    for it in range(iters):
        idx = rnd.choice(len(eps), bs, replace=False)
        batch = [eps[i] for i in idx]
        asked, logps, ents = rollout(model, batch, device, "learned", use_unc, True, net=net)
        r = torch.tensor(heldout_acc(model, batch, asked, device), dtype=torch.float32)
        adv = r - r.mean()
        loss = -(torch.stack(logps).sum(0) * adv).mean() - 0.01 * torch.stack(ents).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 10 == 0:
            log(f"  iter {it}: train reward {r.mean():.3f}")
    return net


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    k, nw, epochs, iters, bs = POOL + HELD, (1000, 50, 100, 400), 10, 80, 128
    if a.quick:
        a.seeds, nw, epochs, iters, bs = 1, (24, 8, 10, 20), 2, 3, 16
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    names = ["random", "entropy", "learned_full", "learned_structure"]
    with Run("logic_learned_asking" + ("_quick" if a.quick else ""), {**vars(a), "pool": POOL, "budget": BUDGET, "iters": iters}) as run:
        tr = make_episodes(levels, range(nw[0]), k, 0.5, rng_seed=1)
        va = make_episodes(levels, range(nw[0], nw[0] + nw[1]), k, 0.5, rng_seed=2)
        te = [e for e in make_episodes(levels, range(1000, 1000 + nw[2]), k, 0.5, rng_seed=3) if len(e.queries) >= k]
        pol = [e for e in make_episodes(levels, range(2000, 2000 + nw[3]), k, 0.5, rng_seed=4) if len(e.queries) >= k]
        run.log(f"test episodes {len(te)}, policy-training episodes {len(pol)}")
        res = {n: [] for n in names}
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            m = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
            fit(m, episode_items(tr, "fb"), episode_items(va, "fb"), device, epochs=epochs, seed=seed, log=lambda s: None)
            nets = {"learned_full": train_policy(m, pol, device, True, seed, iters, bs, run.log),
                    "learned_structure": train_policy(m, pol, device, False, seed, iters, bs, run.log)}
            for n in names:
                if n in nets:
                    asked, _, _ = rollout(m, te, device, "learned", n == "learned_full", False, net=nets[n])
                else:
                    asked, _, _ = rollout(m, te, device, n, rnd=random.Random(seed))
                res[n].append(float(heldout_acc(m, te, asked, device).mean()))
            run.log(f"seed {seed}: " + ", ".join(f"{n} {res[n][-1]:.1%}" for n in names))
        L = ["# Learned asking policy vs hand rule", "", f"{a.seeds} seeds; {len(te)} test episodes; accuracy on {HELD} held-out claims after {BUDGET} asks of a pool of {POOL}. Predictions L1-L2 in the script.", "",
             "| policy | accuracy (sd over seeds) |", "| --- | ---: |"]
        for n in names:
            L.append(f"| {n} | {100 * np.mean(res[n]):.1f}% ({100 * np.std(res[n]):.1f}) |")
        L += ["", "Paired differences (points; sd):", ""]
        for x, y in (("learned_full", "entropy"), ("learned_full", "random"), ("entropy", "random"), ("learned_structure", "random"), ("learned_structure", "entropy")):
            d = 100 * (np.array(res[x]) - np.array(res[y]))
            L.append(f"- {x} minus {y}: {d.mean():+.1f} ({d.std():.1f})")
        run.finish(res, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

"""The best pieces as one system (Q4), and the baseline numbers for the game logs.

Loop per episode (hidden-premise logic world, 50% of the premises visible): (1) ASK: choose 6 of 12 candidate claims, their true verdicts are revealed and added to the graph;
(2) JUDGE 8 held-out claims of the same world with the verdict model; (3) ABSTAIN ("I don't know yet") with a gate. Two models per seed, trained on random-order episodes of levels 2-5:
the verdict model M_fb and the proof-availability head M_cert (exact 'knowable' labels). Ask policies: random, entropy. Gates: none (answer all), confidence, head, oracle (abstain iff not knowable).
Encoders: the original per-relation graph encoder, and the hypernetwork shared core (the transfer-ready one). Rewards: luck-adjusted (a right answer without a proof earns 0) and standard,
at lam = 2 and 4, abstain = -0.2; thresholds fitted on validation episodes processed with the same ask policy. Test sets: in-distribution (levels 2-5), 8 kinds with long chains, 10 kinds (trees).

PREDICTIONS, written before running:
  P1  entropy asks raise held-out verdict accuracy over random asks by >= 3 points in-distribution and >= 2 points on the large worlds.
  P2  the head gate beats the confidence gate in luck-adjusted reward (lam = 4) by >= 0.03 in-distribution and >= 0.05 on the large worlds; the oracle gate is best.
  P3  entropy asks + head gate is the best non-oracle combination on every test set (falsifier: random asks + head gate is within one seed sd of it).
  P4  the hypernetwork encoder is within 1.5 accuracy points and 0.02 reward of the original encoder.

    python experiments/logic_pipeline.py --quick
    python experiments/logic_pipeline.py
    python experiments/logic_pipeline.py --extended --encoders original      # Q4b

RESULT of the first run: P1 true in-distribution (+5.0) and weak on large worlds (+1.7..+2.9); P2 false in-distribution (head = confidence = oracle) and true only on 8-kind chains; P3 true
in-distribution; P4 mostly true. On large worlds the verdict model collapses (57-75%), because it was trained on <= 6 kinds.
Q24 (--remask 5 --encoders original): training worlds are re-masked 5 times (new visible subset and asked claims per draw) instead of seen once; everything else as Q4.
  Reference (Q4, original encoder, 600 worlds per level, one episode each): in-dist verdict accuracy 86.5% with entropy asks; luck-adjusted reward lam = 4: confidence gate +0.206, head gate +0.206, oracle gate +0.208;
  8-kind chains 59% accuracy, 10-kind trees 75%. PREDICTIONS, written before running: (a) in-dist verdict accuracy with entropy asks rises by >= 2 points; (b) the gain on 8-kind chains and 10-kind trees
  (trained without those sizes) is smaller than the in-distribution gain; (c) the head-minus-confidence gate gap stays within +-0.02 of Q4 (the gate is a function of the graph, not of the amount of data).
Q4b (--extended, written after seeing that): the training episodes also include 8- and 10-kind worlds (trees and long chains, 400 each; different seeds from the tests).
  E1  large-world verdict accuracy rises by >= 8 points over the first run, and entropy asks then beat random asks by >= 3 points there.
  E2  the head gate then beats the confidence gate by >= 0.03 (luck-adjusted, lam = 4) on the large worlds; in-distribution results change by less than 1 point.
  Falsifier of E1: large-world accuracy rises by < 4 points (then the collapse is not a training-range problem).
"""

from __future__ import annotations

import argparse
import dataclasses
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
from logic_active_query_deeper import entropy_orders  # noqa: E402
from logic_knowable_shift import best_thr  # noqa: E402
from stml.model.logic_data import EDGE_TYPES, NODE_DIM  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, knowable, make_big_episodes, make_episodes, query_after_asks  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.model.shared_gnn import DomainEncoder, SharedCore  # noqa: E402
from stml.runlog import Run  # noqa: E402

POOL, BUDGET, HELD, ABSTAIN = 12, 6, 8, 0.2


def build(encoder: str, device):
    if encoder == "original":
        return LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
    return LogicJudge(64, 4, gnn=DomainEncoder(SharedCore(64, 4, 32, "hyper"), NODE_DIM, N_TYPES["fb"])).to(device)


def asks(policy, model, eps, device, rnd):
    if policy == "entropy":
        return entropy_orders(model, eps, POOL, BUDGET, device)
    return [rnd.sample(range(POOL), BUDGET) for _ in eps]


def arrays(mf, mc, eps, asked, device):
    held = list(range(POOL, POOL + HELD))
    items = [query_after_asks(ep, asked[i], j) for i, ep in enumerate(eps) for j in held]
    y = np.array([x.label for x in items])
    p = softmax(logits_of(mf, items, device).cpu().numpy())
    c = softmax(logits_of(mc, items, device).cpu().numpy())[:, 1]
    kn = np.array([knowable(_after(ep, asked[i], j), len(asked[i])) for i, ep in enumerate(eps) for j in held])
    return {"conf": p.max(1), "head": c, "right": (p.argmax(1) == y).astype(float), "kn": kn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--extended", action="store_true", help="add 8- and 10-kind training worlds (Q4b)")
    ap.add_argument("--encoders", default="original,hyper")
    ap.add_argument("--remask", type=int, default=1, help="Q24: number of re-masked draws per training world (levels 2-5 and the extended big worlds); 1 = as before")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, epochs, nbig, K = (600, 50, 100), 10, 100, POOL + HELD
    if a.quick:
        a.seeds, nw, epochs, nbig = 1, (24, 8, 12), 2, 12
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_pipeline" + ("_extended" if a.extended else "") + (f"_remask{a.remask}" if a.remask > 1 else "") + ("_quick" if a.quick else ""), {**vars(a)}) as run:
        tr = sum((make_episodes(levels, range(nw[0]), K, 0.5, rng_seed=1 + 100 * r) for r in range(a.remask)), [])
        if a.extended:
            nb = 400 if not a.quick else 10
            for n_, ch_, rs in ((8, 0.0, 11), (8, 0.9, 12), (10, 0.0, 13), (10, 0.9, 14)):
                tr += sum((make_big_episodes(n_, range(5000, 5000 + nb), K, 0.5, ch_, rs + 100 * r) for r in range(a.remask)), [])
        va = [e for e in make_episodes(levels, range(nw[0], nw[0] + nw[1]), K, 0.5, rng_seed=2) if len(e.queries) >= K]
        tests = {"in-dist (levels 2-5)": [e for e in make_episodes(levels, range(1000, 1000 + nw[2]), K, 0.5, rng_seed=3) if len(e.queries) >= K],
                 "8 kinds, chains": make_big_episodes(8, range(1000, 1000 + nbig), K, 0.5, 0.9, 4), "10 kinds, trees": make_big_episodes(10, range(1000, 1000 + nbig), K, 0.5, 0.0, 5)}
        kn = lambda eps: np.array([knowable(e, t) for e in eps for t in range(len(e.queries))])
        cert = lambda items, kv: [dataclasses.replace(x, label=int(f)) for x, f in zip(items, kv)]
        itr, iva = episode_items(tr, "fb"), episode_items(va, "fb")
        ktr, kva = kn(tr), kn(va)
        R = {}
        for seed in range(a.seeds):
            for enc in a.encoders.split(","):
                torch.manual_seed(seed)
                mf = build(enc, device)
                fit(mf, itr, iva, device, epochs=epochs, seed=seed, log=lambda s: None)
                torch.manual_seed(seed)
                mc = build(enc, device)
                fit(mc, cert(itr, ktr), cert(iva, kva), device, epochs=epochs, seed=seed, log=lambda s: None)
                for pol in ("random", "entropy"):
                    rnd = random.Random(seed)
                    V = arrays(mf, mc, va, asks(pol, mf, va, device, rnd), device)
                    for name, eps in tests.items():
                        T = arrays(mf, mc, eps, asks(pol, mf, eps, device, rnd), device)
                        for lam in (2.0, 4.0):
                            for rew in ("luck", "standard"):
                                def ans(D):
                                    return np.where(D["right"] == 1, np.where(D["kn"], 1.0, 0.0) if rew == "luck" else 1.0, -lam)
                                tc, th = best_thr(V["conf"], ans(V)), best_thr(V["head"], ans(V))
                                gates = {"none": np.ones(len(T["right"]), bool), "confidence": T["conf"] > tc, "head": T["head"] > th, "oracle": T["kn"]}
                                for g, mask in gates.items():
                                    r = float(np.where(mask, ans(T), -ABSTAIN).mean())
                                    R.setdefault((enc, pol, g, name, lam, rew), []).append(r)
                        R.setdefault((enc, pol, "acc", name), []).append(float(T["right"].mean()))
                        R.setdefault((enc, pol, "coverage head", name), []).append(float((T["head"] > 0.5).mean()))
            run.log(f"seed {seed} done")
        m = lambda *k: float(np.mean(R[k]))
        s_ = lambda *k: float(np.std(R[k]))
        L = ["# The pipeline: ask by uncertainty, learn from feedback, abstain when no proof is available", "",
             f"{a.seeds} seeds; 6 asks from a pool of 12, 8 held-out claims judged per episode; abstention costs {ABSTAIN}. Reward = mean per judged claim. Predictions P1-P4 are in the script.", ""]
        for name in tests:
            L += [f"## {name}", "", "| encoder | asks | verdict accuracy | luck-adjusted reward lam=4: answer all | confidence gate | head gate | oracle gate | standard reward lam=4: answer all | confidence gate | head gate |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for enc in a.encoders.split(","):
                for pol in ("random", "entropy"):
                    row = [f"{100 * m(enc, pol, 'acc', name):.1f}%"]
                    for rew, gs in (("luck", ("none", "confidence", "head", "oracle")), ("standard", ("none", "confidence", "head"))):
                        row += [f"{m(enc, pol, g, name, 4.0, rew):+.3f} ({s_(enc, pol, g, name, 4.0, rew):.3f})" for g in gs]
                    L.append(f"| {enc} | {pol} | " + " | ".join(row) + " |")
            L.append("")
        run.finish({str(k): v for k, v in R.items()}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

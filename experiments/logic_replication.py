"""Replication of the headline effects on FRESH worlds (Q9), to measure the variance that comes from WHICH worlds are drawn.

All earlier scripts evaluated on one fixed set of test worlds and varied only the model seed, so the 'seed sd' did not include world-sampling noise. Here, per model seed, the trained
models are evaluated on 5 independent sets of unseen worlds (100 per level, levels 2-5, 20 claims each) and, for the gate comparison, 5 independent sets of 8-kind chain worlds.
Models (trained on levels 2-5, 600 worlds per level, K = 20): static, fb (feedback graph), verdict model for asking, proof-availability head.
Quantities: (a) feedback gain = fb - static accuracy at late claims (t >= 14 of 20); (b) entropy asking minus random asking, accuracy on 8 held-out claims after 6 asks from a pool of 12;
(c) luck-adjusted reward (lam = 4, abstain -0.2) of the head gate minus the confidence gate after entropy asks, in-distribution and on 8-kind chains.

PREDICTIONS, written before running:
  Ra  the feedback gain is positive in all 15 (seed x test set) cells and its mean is within 2 points of the earlier estimates (+8 to +14).
  Rb  the entropy-minus-random gain is positive in all cells, mean between +3 and +7 points.
  Rc  the head-minus-confidence gap is within +-0.02 in-distribution and >= +0.03 on 8-kind chains in at least 4 of 5 test sets (trained on levels 2-5 only, i.e. out of range for 8 kinds).
  Rd  the sd across test sets is <= 1.5 points for (a) and (b): world-sampling noise is no larger than model-seed noise.
  Falsifier: a sign flip in (a) or (b) in any cell, or a test-set sd above 3 points.

    python experiments/logic_replication.py --quick
    python experiments/logic_replication.py
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
from logic_active_query_mechanism import run as run_policy  # noqa: E402
from logic_knowable_shift import best_thr  # noqa: E402
from logic_pipeline import ABSTAIN, BUDGET, HELD, POOL, arrays, asks  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, knowable, make_big_episodes, make_episodes  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sets", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only-c", action="store_true", help="skip (a) and (b); only the gate comparison (Q9b)")
    ap.add_argument("--nper-chain", type=int, default=0, help="episodes per 8-kind chain test set (default: same as the other sets)")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    K, nw, nper, epochs = POOL + HELD, 600, 100, 10
    if a.quick:
        a.seeds, a.sets, nw, nper, epochs = 1, 2, 24, 12, 2
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_replication" + ("_chain400" if a.only_c else "") + ("_quick" if a.quick else ""), {**vars(a)}) as run:
        tr = make_episodes(levels, range(nw), K, 0.5, rng_seed=1)
        va = [e for e in make_episodes(levels, range(nw, nw + 50), K, 0.5, rng_seed=2) if len(e.queries) >= K]
        sets = [[e for e in make_episodes(levels, range(10000 + 1000 * j, 10000 + 1000 * j + nper), K, 0.5, rng_seed=100 + j) if len(e.queries) >= K] for j in range(a.sets)]
        nchain = a.nper_chain or nper
        chains = [make_big_episodes(8, range(20000 + 1000 * j, 20000 + 1000 * j + nchain), K, 0.5, 0.9, 200 + j) for j in range(a.sets)]
        kn = lambda eps: np.array([knowable(e, t) for e in eps for t in range(len(e.queries))])
        cert = lambda items, kv: [dataclasses.replace(x, label=int(f)) for x, f in zip(items, kv)]
        ktr, kva = kn(tr), kn(va)
        R = {q: [] for q in ("a", "b", "c_in", "c_chain")}                     # lists of (seed, set, value)
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            ms = LogicJudge(64, 4, n_types=N_TYPES["static"]).to(device)
            fit(ms, episode_items(tr, "static"), episode_items(va, "static"), device, epochs=epochs, seed=seed, log=lambda s: None)
            torch.manual_seed(seed)
            mf = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
            fit(mf, episode_items(tr, "fb"), episode_items(va, "fb"), device, epochs=epochs, seed=seed, log=lambda s: None)
            torch.manual_seed(seed)
            mc = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
            fit(mc, cert(episode_items(tr, "fb"), ktr), cert(episode_items(va, "fb"), kva), device, epochs=epochs, seed=seed, log=lambda s: None)
            V = arrays(mf, mc, va, asks("entropy", mf, va, device, random.Random(seed)), device)
            rv = np.where(V["right"] == 1, np.where(V["kn"], 1.0, 0.0), -4.0)
            tc, th = best_thr(V["conf"], rv), best_thr(V["head"], rv)
            for j in range(a.sets):
                eps = sets[j]
                if not a.only_c:
                    # (a) feedback gain at late claims
                    it_f, it_s = episode_items(eps, "fb"), episode_items(eps, "static")
                    y = np.array([x.label for x in it_f])
                    late = np.array([x.t >= 14 for x in it_f])
                    rf = (logits_of(mf, it_f, device).argmax(1).cpu().numpy() == y)
                    rs = (logits_of(ms, it_s, device).argmax(1).cpu().numpy() == y)
                    R["a"].append((seed, j, 100 * float(rf[late].mean() - rs[late].mean())))
                    # (b) entropy minus random asking
                    held = [list(range(POOL, POOL + HELD)) for _ in eps]
                    r_ = run_policy(mf, eps, POOL, held, BUDGET, "random", device, random.Random(seed))
                    e_ = run_policy(mf, eps, POOL, held, BUDGET, "entropy", device, random.Random(seed))
                    R["b"].append((seed, j, 100 * (e_ - r_)))
                # (c) head gate minus confidence gate
                for name, ee in (("c_in", eps), ("c_chain", chains[j])):
                    T = arrays(mf, mc, ee, asks("entropy", mf, ee, device, random.Random(seed)), device)
                    rt = np.where(T["right"] == 1, np.where(T["kn"], 1.0, 0.0), -4.0)
                    g = lambda mask: float(np.where(mask, rt, -ABSTAIN).mean())
                    R[name].append((seed, j, g(T["head"] > th) - g(T["conf"] > tc)))
            run.log(f"seed {seed} done")
        R = {q: v for q, v in R.items() if v}

        def stats(q):
            v = np.array([x[2] for x in R[q]]).reshape(a.seeds, a.sets)
            return v.mean(), v.mean(0).std(), v.mean(1).std(), v.min(), v.max()
        L = ["# Replication on fresh worlds", "", f"{a.seeds} model seeds x {a.sets} independent test sets of unseen worlds. 'sd over sets' = sd of the per-set means (world-sampling noise); 'sd over seeds' = sd of the per-seed means. Predictions Ra-Rd in the script.", "",
             "| quantity | mean | sd over test sets | sd over model seeds | min cell | max cell |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for q, label, f in [t for t in (("a", "(a) feedback gain, late claims (points)", "{:+.1f}"), ("b", "(b) entropy minus random asking (points)", "{:+.1f}"),
                                       ("c_in", "(c) head gate minus confidence gate, in-distribution (luck-adjusted reward, lam=4)", "{:+.3f}"),
                                       ("c_chain", "(c) head gate minus confidence gate, 8-kind chains", "{:+.3f}")) if t[0] in R]:
            m, ss, sd_, lo, hi = stats(q)
            L.append(f"| {label} | {f.format(m)} | {f.format(ss).lstrip('+')} | {f.format(sd_).lstrip('+')} | {f.format(lo)} | {f.format(hi)} |")
        pos_chain = int(sum(np.array([x[2] for x in R['c_chain']]).reshape(a.seeds, a.sets).mean(0) >= 0.03))
        L += ["", f"8-kind chains: the head gate beats the confidence gate by >= 0.03 in {pos_chain} of {a.sets} test sets (mean over seeds)."]
        run.finish({q: R[q] for q in R}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

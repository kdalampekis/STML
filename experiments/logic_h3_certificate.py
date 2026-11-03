"""H3, the main test: can a learner that checks its own proof know when not to answer?

Judge = relational GNN over the premise graph, answering follows / contradicted / open, plus a 'why' head that
names the premises of its proof. Two training regimes:
  small     levels 1-4 (up to 6 kinds), as in logic_h3_scaling
  extended  levels 1-4 + tree and chain worlds of 8-12 kinds (referee-labelled training data)
Tested on unseen worlds of 12, 16, 24 and 32 kinds, tree and long-chain families (classes balanced).

Gates, i.e. ways to decide "I don't know yet" (each gives a score; higher = commit):
  confidence      calibrated max probability
  ensemble        mean probability of the 5 seed models (5x the compute)
  depth           minus the length of the proof the judge names (a competence-horizon gate)
  suff            re-judge the claim keeping ONLY the premises the judge named; P(same verdict)
  nec             remove each named premise in turn; 1 - max P(same verdict)  (a proof breaks without its steps)
  cert            suff * nec;     conf x cert   confidence * suff * nec
The learner only uses its own network (never the referee). Thresholds are chosen on in-distribution validation
worlds, then applied unchanged to the larger test worlds. Reward: +1 right, -lam wrong, -0.2 abstain.

    python experiments/logic_h3_certificate.py --quick
    python experiments/logic_h3_certificate.py --seeds 5
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
from stml.model.logic_data import make_big_items, make_items  # noqa: E402
from stml.model.logic_judge import LogicJudge, certificate_scores, fit, fit_temperature, logits_of  # noqa: E402
from stml.runlog import Run  # noqa: E402

K = 0.2
LAMS = (1.0, 2.0, 4.0)
SINGLE = ["confidence", "depth", "suff", "nec", "cert", "conf x cert"]


def aurc(score: np.ndarray, right: np.ndarray) -> float:
    o = np.argsort(-score, kind="stable")
    err = np.cumsum(1.0 - right[o]) / np.arange(1, len(o) + 1)
    return float(err.mean())


def auroc(score: np.ndarray, right: np.ndarray) -> float:
    pos, neg = score[right == 1], score[right == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sel_acc(score: np.ndarray, right: np.ndarray, cov: float) -> float:
    k = max(1, int(round(cov * len(score))))
    return float(right[np.argsort(-score, kind="stable")[:k]].mean())


def honest(score, right, thr, lam) -> float:
    commit = score > thr
    return float(np.where(commit, np.where(right == 1, 1.0, -lam), -K).mean())


def best_threshold(score, right, lam) -> float:
    cands = np.unique(np.quantile(score, np.linspace(0, 1, 101)))
    return float(max(cands, key=lambda t: honest(score, right, t, lam)))


def scores_for(model, items, device, temp) -> dict:
    c = certificate_scores(model, items, device, temp)
    conf = c["probs"].max(1)
    return {"probs": c["probs"], "confidence": conf, "depth": -c["named"].astype(float), "suff": c["suff"],
            "nec": c["nec"], "cert": c["cert"], "conf x cert": conf * c["cert"]}


def analyse(method_scores: dict, val_scores: dict, y: np.ndarray, yv: np.ndarray, probs, vprobs) -> dict:
    """Metrics for each gate on one test set, thresholds fixed on the validation set."""
    right = (probs.argmax(1) == y).astype(float)
    rightv = (vprobs.argmax(1) == yv).astype(float)
    out = {"accuracy": float(right.mean()), "always_commit": {str(l): float(np.where(right == 1, 1.0, -l).mean()) for l in LAMS},
           "always_abstain": -K}
    for name, s in method_scores.items():
        r = {"aurc": aurc(s, right), "auroc": auroc(s, right), "sel50": sel_acc(s, right, 0.5), "sel80": sel_acc(s, right, 0.8)}
        for lam in LAMS:
            t = best_threshold(val_scores[name], rightv, lam)
            r[f"honest{lam}"] = honest(s, right, t, lam)
            r[f"coverage{lam}"] = float((s > t).mean())
        out[name] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    sizes, ntest, nbig = (12, 16, 24, 32), 30, 100
    nw = (300, 50)
    if a.quick:
        a.seeds, a.epochs, sizes, ntest, nbig, nw = 1, 2, (12, 16), 5, 8, (30, 10)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with Run("logic_h3_certificate" + ("_quick" if a.quick else ""), {**vars(a), "sizes": sizes}) as run:
        small_tr = make_items([1, 2, 3, 4], range(nw[0]), rng_seed=1)
        small_va = make_items([1, 2, 3, 4], range(nw[0], sum(nw)), rng_seed=2)
        ext_tr, ext_va = [], []
        for n in (8, 10, 12):
            for chain in (0.0, 0.9):
                ext_tr += make_big_items(n, range(5000, 5000 + nbig), rng_seed=n, chain=chain)
                ext_va += make_big_items(n, range(6000, 6000 + max(5, nbig // 5)), rng_seed=n + 1, chain=chain)
        regimes = {"small": (small_tr, small_va), "extended": (small_tr + ext_tr, small_va + ext_va)}
        run.log(f"training items: small {len(small_tr)}, extended {len(regimes['extended'][0])}")
        tests = {}
        for fam, chain in (("tree", 0.0), ("chain", 0.9)):
            for n in sizes:
                tests[(fam, n)] = make_big_items(n, range(1000, 1000 + ntest), rng_seed=n, chain=chain)
                run.log(f"test {fam} n={n}: {len(tests[(fam, n)])} items, longest proof {max(i.support for i in tests[(fam, n)])}")
        raw = {}
        for rname, (tr, va) in regimes.items():
            yv = np.array([it.label for it in va])
            yvt = torch.as_tensor(yv, device=device)
            val_s, test_s = [], {k: [] for k in tests}
            for seed in range(a.seeds):
                torch.manual_seed(seed)
                m = LogicJudge(64, a.layers).to(device)
                fit(m, tr, va, device, epochs=a.epochs, seed=seed, log=lambda s: None, why_weight=0.5)
                temp = fit_temperature(logits_of(m, va, device), yvt)
                val_s.append(scores_for(m, va, device, temp))
                for k, items in tests.items():
                    test_s[k].append(scores_for(m, items, device, temp))
                run.log(f"{rname}, seed {seed}: temperature {temp:.2f}")
            # per-seed analysis + ensemble
            res = {}
            for k, items in tests.items():
                y = np.array([it.label for it in items])
                per = [analyse({n: ts[n] for n in SINGLE}, {n: vs[n] for n in SINGLE}, y, yv, ts["probs"], vs["probs"])
                       for ts, vs in zip(test_s[k], val_s)]
                ens_p = np.mean([ts["probs"] for ts in test_s[k]], 0)
                ens_v = np.mean([vs["probs"] for vs in val_s], 0)
                ens = analyse({"ensemble": ens_p.max(1)}, {"ensemble": ens_v.max(1)}, y, yv, ens_p, ens_v)
                res[f"{k[0]}-{k[1]}"] = {"per_seed": per, "ensemble": ens}
            raw[rname] = res
            run.log(f"{rname}: analysed")

        def avg(rname, key, method, metric):
            r = raw[rname][key]
            if method == "ensemble":
                return r["ensemble"]["ensemble"][metric]
            return float(np.mean([p[method][metric] for p in r["per_seed"]]))
        methods = SINGLE + ["ensemble"]
        L = ["# H3: can the learner check its own proof?", "",
             f"{a.seeds} seeds, {a.layers}-layer judge. Thresholds fixed on in-distribution validation worlds, applied to larger test worlds. "
             "AURC = area under the risk-coverage curve (lower is better). Sel@50 = accuracy on the 50% most confident answers. "
             "Honest score: +1 right, -lam wrong, -0.2 abstain (compare with always committing).", ""]
        for rname in raw:
            for fam in ("tree", "chain"):
                L += [f"## {rname} training, {fam} worlds", ""]
                L += ["| | " + " | ".join(f"acc {n}" for n in sizes) + " |", "| --- | " + " | ".join("---:" for _ in sizes) + " |"]
                L += ["| accuracy of answers | " + " | ".join(f"{np.mean([p['accuracy'] for p in raw[rname][f'{fam}-{n}']['per_seed']]):.1%}" for n in sizes) + " |", ""]
                L += ["AURC by world size (lower is better)", "", "| gate | " + " | ".join(f"{n} kinds" for n in sizes) + " |",
                      "| --- | " + " | ".join("---:" for _ in sizes) + " |"]
                for me in methods:
                    L.append(f"| {me} | " + " | ".join(f"{avg(rname, f'{fam}-{n}', me, 'aurc'):.3f}" for n in sizes) + " |")
                L.append("")
                for n in sizes[-2:]:
                    key = f"{fam}-{n}"
                    L += [f"{n} kinds: AUROC (can the gate tell right from wrong), Sel@50, honest score", "",
                          "| gate | AUROC | Sel@50 | Sel@80 | honest lam=1 | honest lam=2 | honest lam=4 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
                    for me in methods:
                        L.append(f"| {me} | {avg(rname, key, me, 'auroc'):.3f} | {avg(rname, key, me, 'sel50'):.1%} | {avg(rname, key, me, 'sel80'):.1%} | "
                                 + " | ".join(f"{avg(rname, key, me, f'honest{l}'):+.3f}" for l in LAMS) + " |")
                    ac_c = {str(l): float(np.mean([p['always_commit'][str(l)] for p in raw[rname][key]['per_seed']])) for l in LAMS}
                    L += [f"| *always commit* | | | | {ac_c['1.0']:+.3f} | {ac_c['2.0']:+.3f} | {ac_c['4.0']:+.3f} |",
                          f"| *always abstain* | | | | -0.200 | -0.200 | -0.200 |", ""]
        run.finish(raw, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

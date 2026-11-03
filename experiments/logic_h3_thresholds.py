"""H3, deployment: how should the learner choose its abstention threshold when the problems get harder?

logic_h3_certificate showed that the sufficiency check ranks the learner's answers by reliability better than
softmax confidence, but a threshold fixed on small in-distribution worlds almost never abstains on large ones.
Same models and test worlds, now with four ways of choosing the threshold (per gate, per lambda):
  in-dist       best threshold on in-distribution validation worlds (the failure in the previous run)
  + hard val    validation worlds plus 40 worlds of 16 kinds (both families), still smaller than the test worlds
  few-shot m    the learner answered m problems in the new setting and saw right/wrong (the environment's reward);
                m = 25, 50, 100, 200 drawn from calibration worlds, mean over 10 draws
  oracle        best threshold on the evaluation items themselves (a ceiling, not deployable)
Each test set's worlds are split in half: calibration worlds and evaluation worlds. Every strategy is scored on
the evaluation half. Gates: confidence, suff (sufficiency check), ensemble (mean probability of 5 seeds).
Honest reward: +1 right, -lam wrong, -0.2 abstain.

    python experiments/logic_h3_thresholds.py --quick
    python experiments/logic_h3_thresholds.py --seeds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import stml.model.logic_data as ld  # noqa: E402
from logic_h3_certificate import K, best_threshold, honest, scores_for  # noqa: E402
from stml.model.logic_data import make_big_items, make_items  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, fit_temperature, logits_of  # noqa: E402
from stml.runlog import Run  # noqa: E402

LAMS = (1.0, 2.0, 4.0)
MS = (25, 50, 100, 200)
GATES = ("confidence", "suff", "ensemble")


def strategies(score_cal, right_cal, score_ev, right_ev, thr_val, thr_hard, lam, rng) -> dict:
    out = {"in-dist": honest(score_ev, right_ev, thr_val, lam), "+ hard val": honest(score_ev, right_ev, thr_hard, lam)}
    for m in MS:
        vals = []
        for _ in range(10):
            idx = rng.choice(len(score_cal), size=min(m, len(score_cal)), replace=False)
            vals.append(honest(score_ev, right_ev, best_threshold(score_cal[idx], right_cal[idx], lam), lam))
        out[f"few-shot {m}"] = float(np.mean(vals))
    out["oracle"] = honest(score_ev, right_ev, best_threshold(score_ev, right_ev, lam), lam)
    out["always commit"] = float(np.where(right_ev == 1, 1.0, -lam).mean())
    out["always abstain"] = -K
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    sizes, ntest, nbig, nhard = (16, 24, 32), 30, 100, 20
    nw = (300, 50)
    if a.quick:
        a.seeds, a.epochs, sizes, ntest, nbig, nhard, nw = 1, 2, (16, 24), 8, 8, 4, (30, 10)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with Run("logic_h3_thresholds" + ("_quick" if a.quick else ""), {**vars(a), "sizes": sizes, "ms": MS}) as run:
        small_tr = make_items([1, 2, 3, 4], range(nw[0]), rng_seed=1)
        small_va = make_items([1, 2, 3, 4], range(nw[0], sum(nw)), rng_seed=2)
        ext_tr, ext_va = [], []
        for n in (8, 10, 12):
            for chain in (0.0, 0.9):
                ext_tr += make_big_items(n, range(5000, 5000 + nbig), rng_seed=n, chain=chain)
                ext_va += make_big_items(n, range(6000, 6000 + max(5, nbig // 5)), rng_seed=n + 1, chain=chain)
        hard = []
        for chain in (0.0, 0.9):
            hard += make_big_items(16, range(7000, 7000 + nhard), rng_seed=77, chain=chain)
        regimes = {"small": (small_tr, small_va), "extended": (small_tr + ext_tr, small_va + ext_va)}
        tests = {}
        for fam, chain in (("tree", 0.0), ("chain", 0.9)):
            for n in sizes:
                tests[(fam, n)] = make_big_items(n, range(1000, 1000 + ntest), rng_seed=n, chain=chain)
        run.log("data ready: " + ", ".join(f"{k[0]}-{k[1]}: {len(v)}" for k, v in tests.items()))
        rng = np.random.default_rng(0)
        report = {}
        for rname, (tr, va) in regimes.items():
            yv = np.array([it.label for it in va])
            yh = np.array([it.label for it in hard])
            vs, hs, ts = [], [], {k: [] for k in tests}
            for seed in range(a.seeds):
                torch.manual_seed(seed)
                m = LogicJudge(64, a.layers).to(device)
                fit(m, tr, va, device, epochs=a.epochs, seed=seed, log=lambda s: None, why_weight=0.5)
                temp = fit_temperature(logits_of(m, va, device), torch.as_tensor(yv, device=device))
                vs.append(scores_for(m, va, device, temp))
                hs.append(scores_for(m, hard, device, temp))
                for k, items in tests.items():
                    ts[k].append(scores_for(m, items, device, temp))
                run.log(f"{rname}, seed {seed} trained and scored")
            np.savez_compressed(run.dir / f"scores_{rname}.npz", **{
                f"{k[0]}-{k[1]}-{g}-{s}": ts[k][s][g] for k in tests for s in range(a.seeds) for g in ("probs", "confidence", "suff")})
            res = {}
            for k, items in tests.items():
                y = np.array([it.label for it in items])
                worlds = sorted({it.seed for it in items})
                cal_w = set(worlds[: len(worlds) // 2])
                cal = np.array([it.seed in cal_w for it in items])
                ens_t = np.mean([t["probs"] for t in ts[k]], 0)
                ens_v = np.mean([v["probs"] for v in vs], 0)
                ens_h = np.mean([h["probs"] for h in hs], 0)
                for lam in LAMS:
                    per = {g: [] for g in GATES}
                    for s in range(a.seeds):
                        for g in GATES:
                            if g == "ensemble":
                                if s > 0:
                                    continue
                                probs_t, probs_v, probs_h = ens_t, ens_v, ens_h
                                st, sv, sh = ens_t.max(1), ens_v.max(1), ens_h.max(1)
                            else:
                                probs_t, probs_v, probs_h = ts[k][s]["probs"], vs[s]["probs"], hs[s]["probs"]
                                st, sv, sh = ts[k][s][g], vs[s][g], hs[s][g]
                            rt = (probs_t.argmax(1) == y).astype(float)
                            rv = (probs_v.argmax(1) == yv).astype(float)
                            rh = (probs_h.argmax(1) == yh).astype(float)
                            thr_v = best_threshold(sv, rv, lam)
                            thr_h = best_threshold(np.concatenate([sv, sh]), np.concatenate([rv, rh]), lam)
                            per[g].append(strategies(st[cal], rt[cal], st[~cal], rt[~cal], thr_v, thr_h, lam, rng))
                    res[(k, lam)] = {g: {n: float(np.mean([p[n] for p in per[g]])) for n in per[g][0]} for g in GATES}
                    res[(k, lam)]["_accuracy_eval"] = float(np.mean([(ts[k][s]["probs"].argmax(1) == y)[~cal].mean() for s in range(a.seeds)]))
            report[rname] = res
            run.log(f"{rname}: analysed")
        names = ["in-dist", "+ hard val"] + [f"few-shot {m}" for m in MS] + ["oracle", "always commit", "always abstain"]
        L = ["# H3 deployment: choosing the abstention threshold under distribution shift", "",
             f"{a.seeds} seeds, {a.layers}-layer judge. Honest score on evaluation worlds (higher is better; +1 right, -lam wrong, -0.2 abstain). "
             "Calibration and evaluation use disjoint worlds. few-shot m = threshold set from m answered problems in the new setting.", ""]
        for rname in report:
            for fam in ("tree", "chain"):
                for lam in (2.0, 4.0):
                    L += [f"## {rname} training, {fam} worlds, lambda = {lam}", ""]
                    cols = [(n, g) for n in sizes[1:] for g in GATES]
                    L += ["| strategy | " + " | ".join(f"{n}k {g}" for n, g in cols) + " |", "| --- | " + " | ".join("---:" for _ in cols) + " |"]
                    for nm in names:
                        L.append(f"| {nm} | " + " | ".join(f"{report[rname][((fam, n), lam)][g][nm]:+.3f}" for n, g in cols) + " |")
                    L.append("")
        run.finish({f"{r}|{k[0]}-{k[1]}|{lam}": v for r, d in report.items() for (k, lam), v in d.items()}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

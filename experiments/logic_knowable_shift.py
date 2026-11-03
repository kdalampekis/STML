"""Abstention gates under distribution shift in the hidden-premise regime (Q3, revised).

Models trained on levels 2-4 with 50% of the premises visible: M_fb (verdict, with feedback edges) and M_cert (is a proof available now: the exact 'knowable' label).
Tested on: in-distribution; 30% visible; 70% visible; level 5 (larger worlds, never trained); level 5 with 30% visible.
Gates: confidence of M_fb; P(knowable) of M_cert; oracle (abstain iff not knowable). Rewards: standard (+1 right, -lam wrong, -0.2 abstain) and luck-adjusted (a right answer
without a proof available earns 0). Thresholds either (a) fitted on in-distribution validation episodes, or (b) re-fitted on 100 answered items from half of the shifted episodes
(few-shot recalibration), then evaluated on the other half (identical evaluation half in both cases).

PREDICTIONS, written before running:
  S1  under the visible-fraction shifts the proof-availability head keeps AUROC >= 0.98 for 'knowable' while the confidence AUROC falls by >= 0.05 from its in-distribution value.
  S2  under the level-5 shift both degrade, the head less than confidence.
  S3  with in-distribution thresholds the head's luck-adjusted reward (lam = 4) stays within 0.03 of the oracle in every shifted set, and the confidence gate falls >= 0.05 below the oracle in at least one.
  Falsifier: the confidence gate degrades no more than the head, or the head loses >= 0.05 against the oracle.

    python experiments/logic_knowable_shift.py --quick
    python experiments/logic_knowable_shift.py
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import stml.model.logic_data as ld  # noqa: E402
from logic_h3_certificate import K, auroc  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, knowable, make_episodes  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

LAMS = (2.0, 4.0)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def best_thr(score, ans_reward):
    cands = np.unique(np.quantile(score, np.linspace(0, 1, 101)))
    return float(max(cands, key=lambda t: np.where(score > t, ans_reward, -K).mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, epochs, k = (1000, 50, 100), 10, 10
    if a.quick:
        a.seeds, nw, epochs, k = 1, (24, 8, 12), 2, 6
    device = pick_device(None)
    with Run("logic_knowable_shift" + ("_quick" if a.quick else ""), {**vars(a)}) as run:
        tr = make_episodes([2, 3, 4], range(nw[0]), k, 0.5, rng_seed=1)
        va = make_episodes([2, 3, 4], range(nw[0], nw[0] + nw[1]), k, 0.5, rng_seed=2)
        tests = {"in-dist": make_episodes([2, 3, 4], range(1000, 1000 + nw[2]), k, 0.5, rng_seed=3),
                 "vis 0.3": make_episodes([2, 3, 4], range(1000, 1000 + nw[2]), k, 0.3, rng_seed=4),
                 "vis 0.7": make_episodes([2, 3, 4], range(1000, 1000 + nw[2]), k, 0.7, rng_seed=5),
                 "level 5": make_episodes([5], range(1000, 1000 + nw[2]), k, 0.5, rng_seed=6),
                 "level 5, vis 0.3": make_episodes([5], range(1000, 1000 + nw[2]), k, 0.3, rng_seed=7)}
        kn = lambda eps: np.array([knowable(e, t) for e in eps for t in range(len(e.queries))])
        mk = lambda eps: episode_items(eps, "fb")
        cert = lambda items, kv: [dataclasses.replace(x, label=int(f)) for x, f in zip(items, kv)]
        itr, iva = mk(tr), mk(va)
        ktr, kva = kn(tr), kn(va)
        R = {n: {} for n in tests}
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            mf = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
            fit(mf, itr, iva, device, epochs=epochs, seed=seed, log=lambda s: None)
            torch.manual_seed(seed)
            mc = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
            fit(mc, cert(itr, ktr), cert(iva, kva), device, epochs=epochs, seed=seed, log=lambda s: None)
            pv = softmax(logits_of(mf, iva, device).cpu().numpy())
            cv = softmax(logits_of(mc, cert(iva, kva), device).cpu().numpy())[:, 1]
            yv = np.array([x.label for x in iva])
            rightv = (pv.argmax(1) == yv).astype(float)
            thr = {}
            for lam in LAMS:
                rv = np.where(rightv == 1, np.where(kva, 1.0, 0.0), -lam)
                thr[lam] = (best_thr(pv.max(1), rv), best_thr(cv, rv))
            rng = np.random.default_rng(seed)
            for name, eps in tests.items():
                items = mk(eps)
                kt = kn(eps)
                y = np.array([x.label for x in items])
                p = softmax(logits_of(mf, items, device).cpu().numpy())
                c = softmax(logits_of(mc, cert(items, kt), device).cpu().numpy())[:, 1]
                conf = p.max(1)
                right = (p.argmax(1) == y).astype(float)
                ke = len(eps[0].queries)
                ev = np.array([(i // ke) % 2 == 1 for i in range(len(items))])             # evaluation half: odd episodes; calibration half: even
                r = R[name]
                r.setdefault("knowable share", []).append(float(kt.mean()))
                r.setdefault("acc", []).append(float(right.mean()))
                r.setdefault("AUROC conf", []).append(auroc(conf, kt.astype(float)))
                r.setdefault("AUROC head", []).append(auroc(c, kt.astype(float)))
                for lam in LAMS:
                    rt = np.where(right == 1, np.where(kt, 1.0, 0.0), -lam)
                    g = lambda gate: float(np.where(gate[ev], rt[ev], -K).mean())
                    r.setdefault(f"oracle {lam}", []).append(g(kt))
                    r.setdefault(f"conf {lam}", []).append(g(conf > thr[lam][0]))
                    r.setdefault(f"head {lam}", []).append(g(c > thr[lam][1]))
                    fs_c, fs_h = [], []
                    cal = np.nonzero(~ev)[0]
                    for _ in range(10):
                        idx = rng.choice(cal, size=min(100, len(cal)), replace=False)
                        fs_c.append(g(conf > best_thr(conf[idx], rt[idx])))
                        fs_h.append(g(c > best_thr(c[idx], rt[idx])))
                    r.setdefault(f"conf few-shot {lam}", []).append(float(np.mean(fs_c)))
                    r.setdefault(f"head few-shot {lam}", []).append(float(np.mean(fs_h)))
            run.log(f"seed {seed} done")
        m = lambda n, key: float(np.nanmean(R[n][key]))
        L = ["# Abstention gates under shift (hidden-premise regime)", "", f"{a.seeds} seeds; trained on levels 2-4 with 50% visible. Luck-adjusted reward (a right answer without a proof earns 0), "
             "evaluated on half of each shifted set's episodes; 'few-shot' = threshold re-fitted on 100 items from the other half. Predictions S1-S3 are in the script.", "",
             "| test set | knowable | accuracy | AUROC confidence | AUROC head |", "| --- | ---: | ---: | ---: | ---: |"]
        for n in tests:
            L.append(f"| {n} | {m(n, 'knowable share'):.1%} | {m(n, 'acc'):.1%} | {m(n, 'AUROC conf'):.3f} | {m(n, 'AUROC head'):.3f} |")
        for lam in LAMS:
            L += ["", f"## Luck-adjusted reward, lam = {lam}", "", "| test set | oracle | confidence (in-dist thr) | head (in-dist thr) | confidence (few-shot) | head (few-shot) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
            for n in tests:
                L.append(f"| {n} | {m(n, f'oracle {lam}'):+.3f} | {m(n, f'conf {lam}'):+.3f} | {m(n, f'head {lam}'):+.3f} | {m(n, f'conf few-shot {lam}'):+.3f} | {m(n, f'head few-shot {lam}'):+.3f} |")
        run.finish(R, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

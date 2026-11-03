"""Does the proof-availability head stay perfect on LARGER worlds and deep chains? (Q3b, follow-up of logic_knowable_shift)

Same models as logic_knowable_shift (trained on game levels 2-4, at most 6 kinds, 50% of the premises visible): M_fb (verdict) and M_cert (a proof is available now).
Tested on hidden-premise episodes of 8 and 10 kinds (tree and long-chain families; 10-12 premises, twice what was seen), exact labels from the semantics (n <= 10).
PREDICTIONS, written before running:
  R1  the head's AUROC for 'knowable' stays >= 0.95 on 8-10 kinds, including chains (it computes derivability, a local property).
  R2  the verdict model's confidence AUROC falls below 0.80 on those sets.
  R3  with in-distribution thresholds the head's luck-adjusted reward (lam = 4) stays within 0.05 of the oracle; the confidence gate does not.
  Falsifier of R1: head AUROC < 0.90 on any of the four sets (then 'shift-robust' only held for the small shifts tested so far).

    python experiments/logic_knowable_big.py --quick
    python experiments/logic_knowable_big.py
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
from logic_knowable_shift import best_thr, softmax  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, knowable, make_big_episodes, make_episodes  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, epochs, k, nbig = (1000, 50), 10, 10, 100
    if a.quick:
        a.seeds, nw, epochs, k, nbig = 1, (24, 8), 2, 6, 12
    device = pick_device(None)
    with Run("logic_knowable_big" + ("_quick" if a.quick else ""), {**vars(a)}) as run:
        tr = make_episodes([2, 3, 4], range(nw[0]), k, 0.5, rng_seed=1)
        va = make_episodes([2, 3, 4], range(nw[0], nw[0] + nw[1]), k, 0.5, rng_seed=2)
        tests = {"8 kinds, tree": make_big_episodes(8, range(1000, 1000 + nbig), k, 0.5, 0.0, 1), "10 kinds, tree": make_big_episodes(10, range(1000, 1000 + nbig), k, 0.5, 0.0, 2),
                 "8 kinds, chain": make_big_episodes(8, range(1000, 1000 + nbig), k, 0.5, 0.9, 3), "10 kinds, chain": make_big_episodes(10, range(1000, 1000 + nbig), k, 0.5, 0.9, 4)}
        kn = lambda eps: np.array([knowable(e, t) for e in eps for t in range(len(e.queries))])
        cert = lambda items, kv: [dataclasses.replace(x, label=int(f)) for x, f in zip(items, kv)]
        itr, iva, ktr, kva = episode_items(tr, "fb"), episode_items(va, "fb"), kn(tr), kn(va)
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
            rightv = (pv.argmax(1) == np.array([x.label for x in iva])).astype(float)
            thr = {lam: (best_thr(pv.max(1), np.where(rightv == 1, np.where(kva, 1.0, 0.0), -lam)), best_thr(cv, np.where(rightv == 1, np.where(kva, 1.0, 0.0), -lam))) for lam in (2.0, 4.0)}
            for name, eps in tests.items():
                items, kt = episode_items(eps, "fb"), kn(eps)
                y = np.array([x.label for x in items])
                p = softmax(logits_of(mf, items, device).cpu().numpy())
                c = softmax(logits_of(mc, cert(items, kt), device).cpu().numpy())[:, 1]
                right = (p.argmax(1) == y).astype(float)
                r = R[name]
                r.setdefault("knowable share", []).append(float(kt.mean()))
                r.setdefault("acc", []).append(float(right.mean()))
                r.setdefault("AUROC conf", []).append(auroc(p.max(1), kt.astype(float)))
                r.setdefault("AUROC head", []).append(auroc(c, kt.astype(float)))
                r.setdefault("head accuracy on knowable", []).append(float(((c > 0.5) == kt).mean()))
                for lam in (2.0, 4.0):
                    rt = np.where(right == 1, np.where(kt, 1.0, 0.0), -lam)
                    g = lambda gate: float(np.where(gate, rt, -K).mean())
                    r.setdefault(f"oracle {lam}", []).append(g(kt))
                    r.setdefault(f"conf {lam}", []).append(g(p.max(1) > thr[lam][0]))
                    r.setdefault(f"head {lam}", []).append(g(c > thr[lam][1]))
            run.log(f"seed {seed} done")
        m = lambda n, key: float(np.nanmean(R[n][key]))
        L = ["# Proof-availability head on larger worlds and deep chains", "", f"{a.seeds} seeds; trained on levels 2-4 (<= 6 kinds); {nbig} test episodes x {k} claims per set. Predictions R1-R3 in the script.", "",
             "| test set | knowable | verdict accuracy | AUROC confidence | AUROC head | head right/wrong on 'knowable' |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for n in tests:
            L.append(f"| {n} | {m(n, 'knowable share'):.1%} | {m(n, 'acc'):.1%} | {m(n, 'AUROC conf'):.3f} | {m(n, 'AUROC head'):.3f} | {m(n, 'head accuracy on knowable'):.1%} |")
        for lam in (2.0, 4.0):
            L += ["", f"## Luck-adjusted reward, lam = {lam} (in-distribution thresholds)", "", "| test set | oracle | confidence gate | head gate |", "| --- | ---: | ---: | ---: |"]
            for n in tests:
                L.append(f"| {n} | {m(n, f'oracle {lam}'):+.3f} | {m(n, f'conf {lam}'):+.3f} | {m(n, f'head {lam}'):+.3f} |")
        run.finish(R, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

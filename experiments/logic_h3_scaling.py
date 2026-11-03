"""H3 stress test: the same judge, trained on small worlds (levels 1-4, up to 6 kinds), tested on much larger
worlds (6-24 kinds) and on long-chain worlds with deep proofs. Where does accuracy fall, and does the judge's
confidence fall with it? If not, it is overconfident exactly where it should say "I don't know yet".

Labels for the large worlds come from the SAT referee (checked against the exact semantics for n <= 10).
Two families: 'tree' (random hierarchy, shallow proofs) and 'chain' (long 'all' chains, deep proofs).

    python experiments/logic_h3_scaling.py --quick
    python experiments/logic_h3_scaling.py --seeds 5
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from logic_h3_judge import K, LAMBDAS, evaluate, softmax  # noqa: E402
from stml.model.logic_data import make_big_items, make_items  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, fit_temperature, logits_of  # noqa: E402
from stml.runlog import Run, table  # noqa: E402

SIZES = (6, 8, 12, 16, 24)
BUCKETS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 99))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--worlds", type=int, default=40, help="test worlds per size and family")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--size-free", action="store_true", help="drop size-dependent node features")
    a = ap.parse_args()
    if a.size_free:
        import stml.model.logic_data as ld
        ld.SIZE_FREE = True
    sizes, nw = SIZES, (300, 50)
    if a.quick:
        a.seeds, a.epochs, a.worlds, nw, sizes = 1, 3, 6, (30, 10), (6, 12)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with Run("logic_h3_scaling" + ("_sizefree" if a.size_free else "") + ("_quick" if a.quick else ""), {**vars(a), "sizes": sizes}) as run:
        tr = make_items([1, 2, 3, 4], range(nw[0]), rng_seed=1)
        va = make_items([1, 2, 3, 4], range(nw[0], sum(nw)), rng_seed=2)
        tests = {}
        for fam, chain in (("tree", 0.0), ("chain", 0.9)):
            for n in sizes:
                tests[(fam, n)] = make_big_items(n, range(1000, 1000 + a.worlds), rng_seed=n, chain=chain)
                run.log(f"{fam} n={n}: {len(tests[(fam, n)])} items, longest proof {max(i.support for i in tests[(fam, n)])}")
        yv = torch.as_tensor([it.label for it in va], device=device)
        res = {k: [] for k in tests}
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            m = LogicJudge(64, 4).to(device)
            fit(m, tr, va, device, epochs=a.epochs, seed=seed, log=lambda s: None)
            temp = fit_temperature(logits_of(m, va, device), yv)
            for k, items in tests.items():
                lg = logits_of(m, items, device)
                r = evaluate(lg, items, temp)
                p = softmax(lg.cpu().numpy() / temp)
                y = np.array([it.label for it in items])
                r["mean_confidence"] = float(p.max(1).mean())
                r["by_support"] = {f"{lo}-{hi}" if lo != hi else str(lo): float((p.argmax(1) == y)[[it.label == 0 and lo <= it.support <= hi for it in items]].mean())
                                   for lo, hi in BUCKETS if any(it.label == 0 and lo <= it.support <= hi for it in items)}
                r["conf_by_support"] = {f"{lo}-{hi}" if lo != hi else str(lo): float(p.max(1)[[it.label == 0 and lo <= it.support <= hi for it in items]].mean())
                                        for lo, hi in BUCKETS if any(it.label == 0 and lo <= it.support <= hi for it in items)}
                res[k].append(r)
            run.log(f"seed {seed} done (temperature {temp:.2f})")
        mean = lambda k, f: float(np.mean([f(r) for r in res[k]]))
        rep = ["# H3 stress test: small-world judge on large worlds", "",
               f"{a.seeds} seeds, trained on levels 1-4 only (up to 6 kinds). Test classes are balanced (a third each). "
               "Calibration temperature fitted on small validation worlds only.", ""]
        for fam in ("tree", "chain"):
            rows = {}
            for n in sizes:
                k = (fam, n)
                rows[f"{n} kinds"] = {
                    "acc": mean(k, lambda r: r["raw"]["accuracy"]), "conf": mean(k, lambda r: r["mean_confidence"]),
                    "ece": mean(k, lambda r: r["calibrated"]["ece"]),
                    "abs": mean(k, lambda r: 1 - r["by_lambda"]["2.0"]["coverage"]),
                    "h2": mean(k, lambda r: r["by_lambda"]["2.0"]["honest_score"]),
                    "c2": mean(k, lambda r: r["by_lambda"]["2.0"]["always_commit"]),
                    "h4": mean(k, lambda r: r["by_lambda"]["4.0"]["honest_score"]),
                    "c4": mean(k, lambda r: r["by_lambda"]["4.0"]["always_commit"])}
            rep += [f"## {fam} worlds", "", table(rows, [
                ("acc", "Accuracy", "{:.1%}"), ("conf", "Mean confidence", "{:.1%}"), ("ece", "ECE (calibrated)", "{:.3f}"),
                ("abs", "Abstains (lam=2)", "{:.1%}"), ("h2", "Honest score lam=2", "{:+.3f}"), ("c2", "Always commit lam=2", "{:+.3f}"),
                ("h4", "Honest score lam=4", "{:+.3f}"), ("c4", "Always commit lam=4", "{:+.3f}")]), ""]
        big = [k for k in tests if k[1] == max(sizes)]
        for k in big:
            sup = defaultdict(list)
            conf = defaultdict(list)
            for r in res[k]:
                for b, v in r["by_support"].items():
                    sup[b].append(v)
                for b, v in r["conf_by_support"].items():
                    conf[b].append(v)
            rep += [f"Claims that follow, {k[0]} worlds with {k[1]} kinds, by premises in the proof: " +
                    "; ".join(f"{b}: accuracy {np.mean(sup[b]):.0%}, confidence {np.mean(conf[b]):.0%}" for b in sup), ""]
        run.finish({f"{k[0]}-{k[1]}": v for k, v in res.items()}, "\n".join(rep) + "\n")


if __name__ == "__main__":
    main()

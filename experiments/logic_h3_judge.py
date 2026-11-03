"""H3 in the logic domain: does a graph judge know what it does not know, including on harder worlds?

Train on levels 1-4 (worlds seeds 0-299), choose the epoch and fit the temperature on validation worlds
(300-349, levels 1-4), test on unseen worlds of levels 1-4 (350-449) and on level 5 (350-449), which is
larger and never seen in training. Answers: follows / contradicted / open, or "I don't know yet" when the
calibrated confidence is below (lam - k) / (1 + lam). Reward: +1 right, -lam wrong, -k abstain (k = 0.2).

    python experiments/logic_h3_judge.py --quick
    python experiments/logic_h3_judge.py --seeds 5
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

from stml.model.logic_data import TRUTHS, make_items  # noqa: E402
from stml.model.logic_judge import (LogicJudge, answer_with_abstention, ece, fit, fit_temperature,  # noqa: E402
                                    logits_of)
from stml.model.network import n_params, pick_device  # noqa: E402
from stml.runlog import Run, table  # noqa: E402

LAMBDAS = (1.0, 2.0, 4.0)
K = 0.2


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def evaluate(logits: torch.Tensor, items, temp: float) -> dict:
    y = np.array([it.label for it in items])
    out = {}
    for name, t in (("raw", 1.0), ("calibrated", temp)):
        p = softmax(logits.cpu().numpy() / t)
        right = (p.argmax(1) == y).astype(float)
        out[name] = {"accuracy": float(right.mean()), "ece": ece(p.max(1), right),
                     "brier": float(((p.max(1) - right) ** 2).mean())}
    p = softmax(logits.cpu().numpy() / temp)
    out["by_lambda"] = {}
    for lam in LAMBDAS:
        m = answer_with_abstention(p, items, lam, K)
        always = answer_with_abstention(p, items, lam, K) if False else None
        # references: commit to everything (threshold 0) and abstain on everything
        commit_all = float(np.mean([1.0 if p[i].argmax() == y[i] else -lam for i in range(len(y))]))
        out["by_lambda"][str(lam)] = {**{k: v for k, v in m.items() if v is not None},
                                      "always_commit": commit_all, "always_abstain": -K}
    # does abstention track real difficulty?  (lam=2)
    thr = (2.0 - K) / 3.0
    abst = p.max(1) <= thr
    out["abstain_by_level"] = {str(L): float(abst[[it.level == L for it in items]].mean())
                               for L in sorted({it.level for it in items})}
    out["abstain_by_truth"] = {TRUTHS[j]: float(abst[y == j].mean()) for j in range(3) if (y == j).any()}
    ent = [(it.support, a) for it, a in zip(items, abst) if it.label == 0]
    sup = defaultdict(list)
    for s, a in ent:
        sup[min(s, 6)].append(a)
    out["abstain_by_support"] = {str(k): float(np.mean(v)) for k, v in sorted(sup.items())}
    out["majority_accuracy"] = float(np.bincount(y).max() / len(y))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    nw = (30, 10, 20)
    if a.quick:
        a.seeds, a.epochs = 1, 3
    else:
        nw = (300, 50, 100)
    device = pick_device(a.device)
    with Run("logic_h3_judge" + ("_quick" if a.quick else ""), {**vars(a), "worlds": nw, "lambdas": LAMBDAS, "k": K,
                                                                 "device_used": str(device)}) as run:
        tr = make_items([1, 2, 3, 4], range(nw[0]), rng_seed=1)
        va = make_items([1, 2, 3, 4], range(nw[0], nw[0] + nw[1]), rng_seed=2)
        te = make_items([1, 2, 3, 4], range(nw[0] + nw[1], sum(nw)), rng_seed=3)
        ood = make_items([5], range(nw[0] + nw[1], sum(nw)), rng_seed=4)
        run.log(f"items: train {len(tr)}, val {len(va)}, test {len(te)}, level-5 {len(ood)}; "
                f"classes in train {np.bincount([i.label for i in tr]).tolist()}")
        res = {"in_distribution": [], "level5": []}
        yv = torch.as_tensor([it.label for it in va], device=device)
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            m = LogicJudge(a.d, a.layers).to(device)
            if seed == 0:
                run.log(f"model size: {n_params(m):,} parameters")
            fit(m, tr, va, device, epochs=a.epochs, seed=seed, log=lambda s: None)
            temp = fit_temperature(logits_of(m, va, device), yv)
            for name, items in (("in_distribution", te), ("level5", ood)):
                r = evaluate(logits_of(m, items, device), items, temp)
                r["temperature"] = temp
                res[name].append(r)
                run.metric({"seed": seed, "set": name, **{k: v for k, v in r.items() if k in ("raw", "calibrated", "temperature")}})
            run.log(f"seed {seed}: temperature {temp:.2f}; in-dist acc {res['in_distribution'][-1]['raw']['accuracy']:.1%}, "
                    f"level-5 acc {res['level5'][-1]['raw']['accuracy']:.1%}")

        def mean(name, path):
            vals = []
            for r in res[name]:
                x = r
                for p in path:
                    x = x[p]
                vals.append(x)
            return float(np.mean(vals)), float(np.std(vals))

        rows = {}
        for name, label in (("in_distribution", "levels 1-4 (unseen worlds)"), ("level5", "level 5 (harder, never trained)")):
            rows[label] = {"acc": mean(name, ["raw", "accuracy"])[0], "ece_raw": mean(name, ["raw", "ece"])[0],
                           "ece_cal": mean(name, ["calibrated", "ece"])[0], "brier_cal": mean(name, ["calibrated", "brier"])[0],
                           "maj": float(np.mean([r["majority_accuracy"] for r in res[name]]))}
        cols = [("acc", "Accuracy", "{:.1%}"), ("maj", "Majority class", "{:.1%}"), ("ece_raw", "ECE raw", "{:.3f}"),
                ("ece_cal", "ECE calibrated", "{:.3f}"), ("brier_cal", "Brier", "{:.3f}")]
        rep = ["# H3 in the logic domain: graph judge with abstention", "",
               f"{a.seeds} model seeds · {a.epochs} epochs · d={a.d} · {a.layers} GNN layers · k={K}. "
               "Trained on levels 1-4; temperature fitted on validation worlds only.", "", table(rows, cols), ""]
        for lam in LAMBDAS:
            h = {}
            for name, label in (("in_distribution", "levels 1-4"), ("level5", "level 5")):
                g = lambda key: float(np.mean([r["by_lambda"][str(lam)][key] for r in res[name]]))
                h[label] = {"cov": g("coverage"), "sel": g("selective_accuracy"), "hon": g("honest_score"),
                            "all": g("always_commit"), "abs": g("always_abstain")}
            rep += [f"## lambda = {lam} (commit above {(lam - K) / (1 + lam):.2f})", "",
                    table(h, [("cov", "Coverage", "{:.1%}"), ("sel", "Accuracy when committing", "{:.1%}"),
                              ("hon", "Honest score", "{:+.3f}"), ("all", "Always commit", "{:+.3f}"),
                              ("abs", "Always abstain", "{:+.3f}")]), ""]
        for name, label in (("in_distribution", "levels 1-4"), ("level5", "level 5")):
            by = defaultdict(list)
            for r in res[name]:
                for k, v in r["abstain_by_support"].items():
                    by[k].append(v)
            rep += [f"Abstention at lambda=2 on {label}: by truth " +
                    ", ".join(f"{k} {np.mean([r['abstain_by_truth'].get(k, np.nan) for r in res[name]]):.1%}" for k in TRUTHS)
                    + "; follows-claims by premises needed " + ", ".join(f"{k}: {np.mean(v):.0%}" for k, v in sorted(by.items())), ""]
        run.finish(res, "\n".join(rep) + "\n")


if __name__ == "__main__":
    main()

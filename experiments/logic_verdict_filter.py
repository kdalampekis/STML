"""Q22 (out of the box): VERIFY A VERDICT BEFORE STORING IT. Noisy feedback (20% wrong, Q11/Q21) costs 40-60% of the feedback gain and data diversity alone does not cure it.
A learner can check a revealed verdict against the rest of its graph: re-judge claim s from the visible premises and the OTHER revealed verdicts (leave-one-out), and drop the edge if the
shown verdict has very low probability. This is the 'hypercorrection / prediction error as a checking signal' idea (C1/C6) used as a filter rather than a loss weight.

Setting as Q21: 300 worlds x 10 draws, hidden-premise episodes (50% visible, 10 claims), 20% of the shown verdicts wrong. Two models: clean-trained (trusts its edges) and noise-trained
(noise redrawn per draw). Evaluation on noisy test episodes at claims 7-10: unfiltered vs filtered graphs (edge s dropped at time t if P(shown verdict of s | visible + other shown verdicts before t) < tau),
tau in {0.02, 0.05, 0.1, 0.2}. The filter uses the model's own predictions, never the true labels. Also: AUROC of '1 - P(shown)' for detecting wrong verdicts.
PREDICTIONS, written before running:
  V1  detection AUROC of wrong verdicts >= 0.75 (for the clean-trained model).
  V2  filtering raises the clean-trained model's accuracy on noisy test episodes by >= 2 points at its best tau (tau picked on validation episodes, not test).
  V3  the gain of filtering is smaller for the noise-trained model (it already discounts).
  Falsifier of V2: gain within the seed sd for every tau.

    python experiments/logic_verdict_filter.py --quick
    python experiments/logic_verdict_filter.py
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
from stml.model.logic_episodes import N_TYPES, episode_graph, episode_items, make_episodes, noisy_feedback, query_after_asks  # noqa: E402
from stml.model.logic_data import Item  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

TAUS = (0.02, 0.05, 0.1, 0.2)


def loo_scores(model, eps, device):
    """For each episode, time t and earlier claim s<t: P(shown verdict of s | visible premises + the other shown verdicts before t). Returns list[dict[(t, s)] -> p]."""
    items, keys = [], []
    for i, ep in enumerate(eps):
        shown = ep.shown if ep.shown is not None else ep.labels
        for t in range(len(ep.queries)):
            for s in range(t):
                items.append(query_after_asks(ep, [j for j in range(t) if j != s], s))
                keys.append((i, t, s, shown[s]))
    p = torch.softmax(logits_of(model, items, device), 1).cpu().numpy()
    out = [dict() for _ in eps]
    for (i, t, s, y), row in zip(keys, p):
        out[i][(t, s)] = float(row[y])
    return out


def filtered_items(eps, scores, tau):
    items = []
    for i, ep in enumerate(eps):
        for t in range(len(ep.queries)):
            keep = [scores[i][(t, s)] >= tau for s in range(t)] if tau > 0 else None
            it = Item(ep.level, ep.seed, ep.n, ep.queries[t], ep.labels[t], 0, episode_graph(ep, t, "fb", keep=keep), len(ep.visible), (), ())
            it.t = t
            items.append(it)
    return items


def late(m, items, device) -> float:
    y = np.array([x.label for x in items])
    t = np.array([x.t for x in items])
    return float(((logits_of(m, items, device).argmax(1).cpu().numpy() == y).astype(float))[t >= 7].mean())


def auroc(score, pos):
    from itertools import product  # noqa: F401
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--p", type=float, default=0.2)
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, draws, nv, nte, epochs, k = 300, 10, 50, 100, 10, 10
    if a.quick:
        a.seeds, nw, draws, nv, nte, epochs, k = 1, 8, 3, 8, 12, 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_verdict_filter" + ("_quick" if a.quick else ""), vars(a)) as run:
        draws_ep = [make_episodes(levels, range(nw), k, 0.5, rng_seed=100 + r) for r in range(draws)]
        clean10 = sum(draws_ep, [])
        noisy10 = sum((noisy_feedback(e, a.p, 1000 + r) for r, e in enumerate(draws_ep)), [])
        va = noisy_feedback(make_episodes(levels, range(10 ** 4, 10 ** 4 + nv), k, 0.5, rng_seed=2), a.p, 78)
        te = noisy_feedback(make_episodes(levels, range(1000, 1000 + nte), k, 0.5, rng_seed=3), a.p, 77)
        a_va = episode_items(va, "fb")
        res = {m: {"unfiltered": [], **{f"tau{t}": [] for t in TAUS}, "auroc": [], "val": {f"tau{t}": [] for t in TAUS}} for m in ("clean_trained", "noise_trained")}
        for seed in range(a.seeds):
            for name, ep in (("clean_trained", clean10), ("noise_trained", noisy10)):
                torch.manual_seed(seed)
                m = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
                fit(m, episode_items(ep, "fb"), a_va if name == "noise_trained" else episode_items(make_episodes(levels, range(10 ** 4, 10 ** 4 + nv), k, 0.5, rng_seed=2), "fb"), device, epochs=epochs, seed=seed, log=lambda s: None)
                sc_te, sc_va = loo_scores(m, te, device), loo_scores(m, va, device)
                res[name]["unfiltered"].append(late(m, filtered_items(te, sc_te, 0.0), device))
                for tau in TAUS:
                    res[name][f"tau{tau}"].append(late(m, filtered_items(te, sc_te, tau), device))
                    res[name]["val"][f"tau{tau}"].append(late(m, filtered_items(va, sc_va, tau), device))
                wrong, score = [], []
                for i, e in enumerate(te):
                    for (t, s), pr in sc_te[i].items():
                        wrong.append(e.shown[s] != e.labels[s])
                        score.append(1 - pr)
                res[name]["auroc"].append(auroc(np.array(score), np.array(wrong)))
            run.log(f"seed {seed}: " + ", ".join(f"{n} unf {res[n]['unfiltered'][-1]:.1%} " + " ".join(f"{t} {res[n][f'tau{t}'][-1]:.1%}" for t in TAUS) + f" auroc {res[n]['auroc'][-1]:.2f}" for n in res))
        L = ["# Leave-one-out verdict filter", "", f"{a.seeds} seeds; p = {a.p}; accuracy at claims 7-10 on noisy test episodes (mean, sd). Predictions V1-V3 in the script.", "",
             "| model | unfiltered | " + " | ".join(f"tau {t}" for t in TAUS) + " | detection AUROC |", "| --- | ---: | " + " | ".join("---:" for _ in TAUS) + " | ---: |"]
        for n, r in res.items():
            L.append(f"| {n} | {100 * np.mean(r['unfiltered']):.1f}% ({100 * np.std(r['unfiltered']):.1f}) | " + " | ".join(f"{100 * np.mean(r[f'tau{t}']):.1f}% ({100 * np.std(r[f'tau{t}']):.1f})" for t in TAUS) + f" | {np.mean(r['auroc']):.2f} ({np.std(r['auroc']):.2f}) |")
        L += ["", "## Gain of filtering over unfiltered (points; sd) and the tau chosen on validation", "", "| model | " + " | ".join(f"tau {t}" for t in TAUS) + " | tau by validation | gain at that tau |", "| --- | " + " | ".join("---:" for _ in TAUS) + " | ---: | ---: |"]
        for n, r in res.items():
            gains = {t: 100 * (np.array(r[f"tau{t}"]) - np.array(r["unfiltered"])) for t in TAUS}
            best = max(TAUS, key=lambda t: np.mean(r["val"][f"tau{t}"]))
            L.append(f"| {n} | " + " | ".join(f"{gains[t].mean():+.1f} ({gains[t].std():.1f})" for t in TAUS) + f" | {best} | {gains[best].mean():+.1f} |")
        run.finish({n: {k_: v for k_, v in r.items() if k_ != 'val'} for n, r in res.items()}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

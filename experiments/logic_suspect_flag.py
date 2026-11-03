"""Q23 (follow-up of Q22): KEEP a doubtful verdict but TELL the model it is doubtful. Q22: the leave-one-out check detects wrong verdicts (AUROC 0.90) but deleting the suspects repairs only
~25% of the loss from 20% noisy feedback, because suspects are also where hidden structure is revealed. A 'suspect' flag keeps the edge and lets the network learn how much to trust it.

Setting: 20% of the shown verdicts wrong, hidden-premise episodes (50% visible, 10 claims), levels 2-5. A scorer M0 (noise-trained, 300 worlds x 5 draws on worlds A) gives every shown verdict a leave-one-out
probability; M1 is trained on DIFFERENT worlds B (300 x 5 draws, noise redrawn) with scores from M0 (so the scores are not in-sample); test worlds are unseen. Never any true-label information in a flag.
Conditions on noisy test episodes, accuracy at claims 7-10:
  plain        M1 on unflagged graphs
  hard_filter  M1 on plain graphs, test graphs with suspects deleted (tau 0.05, as Q22)
  hard_both    M1 trained AND tested on graphs with suspects deleted
  flag         M1 trained and tested on graphs where suspect edges (P < 0.05) have their own edge types (verdict x suspect, 72 types)
  clean_ref    M1 trained and tested on clean feedback (the ceiling)
PREDICTIONS, written before running:
  S1  flag beats hard_filter by >= 1 point.
  S2  flag beats plain by >= 3 points.
  Falsifier of S1: flag within the seed sd of hard_filter.

    python experiments/logic_suspect_flag.py --quick
    python experiments/logic_suspect_flag.py
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
from logic_verdict_filter import late, loo_scores  # noqa: E402
from stml.model.logic_data import Item  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_graph, episode_items, make_episodes, noisy_feedback  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

TAU = 0.05


def build(eps, scores, mode):
    """mode: plain | hard | flag."""
    items = []
    for i, ep in enumerate(eps):
        for t in range(len(ep.queries)):
            if mode == "plain":
                g = episode_graph(ep, t, "fb")
            elif mode == "hard":
                g = episode_graph(ep, t, "fb", keep=[scores[i][(t, s)] >= TAU for s in range(t)])
            else:
                g = episode_graph(ep, t, "fb_s", flavours=[int(scores[i][(t, s)] < TAU) for s in range(t)])
            it = Item(ep.level, ep.seed, ep.n, ep.queries[t], ep.labels[t], 0, g, len(ep.visible), (), ())
            it.t = t
            items.append(it)
    return items


def train_model(items, val, mode, device, seed, epochs):
    torch.manual_seed(seed)
    m = LogicJudge(64, 4, n_types=N_TYPES["fb_s" if mode == "flag" else "fb"]).to(device)
    fit(m, items, val, device, epochs=epochs, seed=seed, log=lambda s: None)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--p", type=float, default=0.2)
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, draws, nv, nte, epochs, k = 300, 5, 50, 100, 10, 10
    if a.quick:
        a.seeds, nw, draws, nv, nte, epochs, k = 1, 8, 2, 8, 12, 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_suspect_flag" + ("_quick" if a.quick else ""), vars(a)) as run:
        def noisy_draws(w0, seed0):
            return sum((noisy_feedback(make_episodes(levels, range(w0, w0 + nw), k, 0.5, rng_seed=100 + r), a.p, seed0 + r) for r in range(draws)), [])
        worlds_a, worlds_b = noisy_draws(0, 1000), noisy_draws(10 ** 3, 2000)
        va = noisy_feedback(make_episodes(levels, range(10 ** 4, 10 ** 4 + nv), k, 0.5, rng_seed=2), a.p, 78)
        te = noisy_feedback(make_episodes(levels, range(1000 * 5, 1000 * 5 + nte), k, 0.5, rng_seed=3), a.p, 77)
        clean_b = sum((make_episodes(levels, range(10 ** 3, 10 ** 3 + nw), k, 0.5, rng_seed=100 + r) for r in range(draws)), [])
        clean_va = make_episodes(levels, range(10 ** 4, 10 ** 4 + nv), k, 0.5, rng_seed=2)
        clean_te = make_episodes(levels, range(1000 * 5, 1000 * 5 + nte), k, 0.5, rng_seed=3)
        names = ["plain", "hard_filter", "hard_both", "flag", "clean_ref"]
        res = {n: [] for n in names}
        for seed in range(a.seeds):
            m0 = train_model(episode_items(worlds_a, "fb"), episode_items(va, "fb"), "plain", device, seed, epochs)
            sc_b, sc_va, sc_te = loo_scores(m0, worlds_b, device), loo_scores(m0, va, device), loo_scores(m0, te, device)
            run.log(f"seed {seed}: scores ready")
            m_plain = train_model(build(worlds_b, sc_b, "plain"), build(va, sc_va, "plain"), "plain", device, seed, epochs)
            res["plain"].append(late(m_plain, build(te, sc_te, "plain"), device))
            res["hard_filter"].append(late(m_plain, build(te, sc_te, "hard"), device))
            m_hard = train_model(build(worlds_b, sc_b, "hard"), build(va, sc_va, "hard"), "plain", device, seed, epochs)
            res["hard_both"].append(late(m_hard, build(te, sc_te, "hard"), device))
            m_flag = train_model(build(worlds_b, sc_b, "flag"), build(va, sc_va, "flag"), "flag", device, seed, epochs)
            res["flag"].append(late(m_flag, build(te, sc_te, "flag"), device))
            m_clean = train_model(episode_items(clean_b, "fb"), episode_items(clean_va, "fb"), "plain", device, seed, epochs)
            res["clean_ref"].append(late(m_clean, episode_items(clean_te, "fb"), device))
            run.log(f"seed {seed}: " + ", ".join(f"{n} {res[n][-1]:.1%}" for n in names))
        L = ["# Suspect flag vs deletion", "", f"{a.seeds} seeds; p = {a.p}; accuracy at claims 7-10 on noisy test episodes (mean, sd). Predictions S1-S2 in the script.", "", "| condition | accuracy |", "| --- | ---: |"]
        for n in names:
            L.append(f"| {n} | {100 * np.mean(res[n]):.1f}% ({100 * np.std(res[n]):.1f}) |")
        L += ["", "| comparison | difference |", "| --- | ---: |"]
        for x, y in (("flag", "hard_filter"), ("flag", "plain"), ("flag", "hard_both"), ("hard_filter", "plain"), ("clean_ref", "flag")):
            d = 100 * (np.array(res[x]) - np.array(res[y]))
            L.append(f"| {x} minus {y} | {d.mean():+.1f} ({d.std():.1f}) |")
        run.finish(res, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

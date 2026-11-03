"""Q26 (out of the box): TEST-TIME TRAINING inside an episode. The feedback learner already reads revealed verdicts as graph edges (in-context). Can it also LEARN from them with gradient steps
at test time, for this one world? The revealed claims give free supervised examples: re-judge revealed claim s from the visible premises and the other revealed verdicts (leave-one-out) and
take a few gradient steps towards the shown verdict (cf. test-time training, Sun et al. 2020/2024; MAML-style adaptation).

Setting: hidden-premise episodes (50% visible, 10 claims, exact verdicts), judge trained on 300 worlds x 10 re-masked draws (the Q20 setting). At time t >= 2 the copy of the model takes K = 3 Adam steps on the leave-one-out
examples of the t revealed claims, then judges claim t; the adaptation accumulates over t within the episode and is discarded afterwards. Learning rate chosen on validation episodes among 3e-4, 1e-3, 3e-3.
Controls: the same procedure with SHUFFLED verdicts as targets (adaptation without information).
PREDICTIONS, written before running (accuracy at claims 7-10 on test episodes):
  T1  test-time training with the best validation lr beats the frozen judge by >= 1.5 points.
  T2  the shuffled-target control is not better than the frozen judge (it can only hurt).
  Falsifier of T1: gain within the seed sd at every lr.
Q26b (--lrs 1e-5,3e-5,1e-4): PREDICTION written after Q26 (every lr hurt, monotonically): no lr beats the frozen judge by more than the seed sd (best gain <= +0.5).

    python experiments/logic_test_time_training.py --quick
    python experiments/logic_test_time_training.py
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stml.model.logic_data as ld  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, make_episodes, query_after_asks  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

LRS = (3e-4, 1e-3, 3e-3)           # overridden by --lrs
K_STEPS = 3


def ttt_correct(model_cpu, state0, ep, items_t, lr, shuffle, rnd) -> list[int]:
    """1 if the adapted model judged claim t correctly, for each t of the episode (t < 2: frozen judge)."""
    model_cpu.load_state_dict(state0)
    opt = torch.optim.Adam(model_cpu.parameters(), lr=lr)
    dev = torch.device("cpu")
    out = []
    for t in range(len(ep.queries)):
        if t >= 2 and lr > 0:
            loo = [query_after_asks(ep, [j for j in range(t) if j != s], s) for s in range(t)]
            y = torch.as_tensor([ep.labels[s] for s in range(t)])
            if shuffle:
                y = y[torch.as_tensor(rnd.sample(range(t), t))]
            model_cpu.train()
            for _ in range(K_STEPS):
                loss = F.cross_entropy(model_cpu(loo, dev), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model_cpu.eval()
        with torch.no_grad():
            out.append(int(model_cpu([items_t[t]], dev).argmax(1).item() == ep.labels[t]))
    return out


def evaluate(model_cpu, state0, eps, lr, shuffle, seed) -> float:
    rnd = random.Random(seed)
    items = episode_items(eps, "fb")
    k = len(eps[0].queries)
    right = []
    for i, ep in enumerate(eps):
        right += ttt_correct(model_cpu, state0, ep, items[i * k:(i + 1) * k], lr, shuffle, rnd)[7:]
    return float(np.mean(right))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lrs", default="", help="comma-separated learning rates (Q26b: 1e-5,3e-5,1e-4)")
    a = ap.parse_args()
    global LRS
    if a.lrs:
        LRS = tuple(float(x) for x in a.lrs.split(','))
    ld.SIZE_FREE = True
    nw, draws, nv, nval, nte, epochs, k = 300, 10, 50, 25, 100, 10, 10
    if a.quick:
        a.seeds, nw, draws, nv, nval, nte, epochs, k = 1, 8, 2, 8, 3, 3, 2, 10
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_test_time_training" + ("_quick" if a.quick else ""), {**vars(a), "lrs": LRS, "k_steps": K_STEPS}) as run:
        tr = sum((make_episodes(levels, range(nw), k, 0.5, rng_seed=100 + r) for r in range(draws)), [])
        va_fit = make_episodes(levels, range(10 ** 4, 10 ** 4 + nv), k, 0.5, rng_seed=2)
        va = make_episodes(levels, range(2 * 10 ** 4, 2 * 10 ** 4 + nval), k, 0.5, rng_seed=7)
        te = make_episodes(levels, range(1000, 1000 + nte), k, 0.5, rng_seed=3)
        res = {"frozen": [], **{f"ttt_{lr}": [] for lr in LRS}, "shuffled_best": [], "best_lr": []}
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            m = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
            fit(m, episode_items(tr, "fb"), episode_items(va_fit, "fb"), device, epochs=epochs, seed=seed, log=lambda s: None)
            mc = copy.deepcopy(m).cpu()
            state0 = {k_: v.clone() for k_, v in mc.state_dict().items()}
            val_acc = {lr: evaluate(mc, state0, va, lr, False, seed) for lr in (0.0,) + LRS}
            best = max(LRS, key=lambda lr: val_acc[lr])
            res["frozen"].append(evaluate(mc, state0, te, 0.0, False, seed))
            for lr in LRS:
                res[f"ttt_{lr}"].append(evaluate(mc, state0, te, lr, False, seed))
            res["shuffled_best"].append(evaluate(mc, state0, te, best, True, seed))
            res["best_lr"].append(best)
            run.log(f"seed {seed}: frozen {res['frozen'][-1]:.1%}, " + ", ".join(f"lr {lr} {res[f'ttt_{lr}'][-1]:.1%}" for lr in LRS) + f", shuffled@{best} {res['shuffled_best'][-1]:.1%}; val {val_acc}")
        L = ["# Test-time training", "", f"{a.seeds} seeds; {len(te)} test episodes; accuracy at claims 7-10 (mean, sd). K = {K_STEPS} Adam steps per revealed set. Predictions T1-T2 in the script.", "", "| condition | accuracy |", "| --- | ---: |"]
        for n in ("frozen",) + tuple(f"ttt_{lr}" for lr in LRS) + ("shuffled_best",):
            L.append(f"| {n} | {100 * np.mean(res[n]):.1f}% ({100 * np.std(res[n]):.1f}) |")
        L += ["", "| comparison | difference |", "| --- | ---: |"]
        for lr in LRS:
            d = 100 * (np.array(res[f"ttt_{lr}"]) - np.array(res["frozen"]))
            L.append(f"| ttt lr {lr} minus frozen | {d.mean():+.1f} ({d.std():.1f}) |")
        d = 100 * (np.array(res["shuffled_best"]) - np.array(res["frozen"]))
        L.append(f"| shuffled targets (validation-best lr) minus frozen | {d.mean():+.1f} ({d.std():.1f}) |")
        L += ["", f"validation-selected lr per seed: {res['best_lr']}"]
        run.finish({k_: v for k_, v in res.items()}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

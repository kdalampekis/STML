"""Hyperparameter search for the feedback learner (rule 7b of docs/EXPERIMENT_QUEUE.md: a promising method gets tuned, then tested deeper).

The feedback learner (hidden-premise episodes, verdicts revealed and added to the graph) is data-limited, not depth-limited. Random search over learning rate (log-uniform
3e-4..3e-3), width {32, 64, 128}, GNN layers {2, 3, 4, 6}, epochs {6, 10, 16}, weight decay (log-uniform 1e-5..1e-2), batch {64, 128, 256}; 11 random configs + the default
(lr 1e-3, width 64, 4 layers, 10 epochs, wd 1e-4, batch 128). Selection uses VALIDATION episodes only (2 seeds per config). The best 3 and the default are then confirmed on the
TEST episodes with 3 NEW seeds.
PREDICTIONS, written before running:
  T1  the best validation config beats the default on test by >= 1 point but by < 4 points (the default is reasonable; the learner is data-limited, not tuning-limited).
  T2  the top configs are not the deeper or wider ones (depth did not help in the sweep); more epochs or lower weight decay matter more than width.
  Falsifier of T1: tuned minus default on test <= the seed sd, or the validation winner does not beat the default on test.

    python experiments/logic_hparam_search.py --quick
    python experiments/logic_hparam_search.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stml.model.logic_data as ld  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, make_episodes  # noqa: E402
from stml.model.logic_judge import LogicJudge, ece, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

DEFAULT = {"lr": 1e-3, "d": 64, "layers": 4, "epochs": 10, "wd": 1e-4, "batch": 128}


def sample_configs(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = [dict(DEFAULT)]
    for _ in range(n):
        out.append({"lr": float(np.exp(rng.uniform(np.log(3e-4), np.log(3e-3)))), "d": int(rng.choice([32, 64, 128])),
                    "layers": int(rng.choice([2, 3, 4, 6])), "epochs": int(rng.choice([6, 10, 16])),
                    "wd": float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-2)))), "batch": int(rng.choice([64, 128, 256]))})
    return out


def train_eval(cfg, tr, va, te, device, seed):
    torch.manual_seed(seed)
    m = LogicJudge(cfg["d"], cfg["layers"], n_types=N_TYPES["fb"]).to(device)
    fit(m, tr, va, device, epochs=cfg["epochs"], lr=cfg["lr"], batch=cfg["batch"], seed=seed, weight_decay=cfg["wd"], log=lambda s: None)
    out = {}
    for name, items in (("val", va), ("test", te)):
        y = np.array([x.label for x in items])
        lg = logits_of(m, items, device).cpu().numpy()
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        right = (p.argmax(1) == y).astype(float)
        t = np.array([x.t for x in items])
        out[name] = {"acc": float(right.mean()), "late": float(right[t >= 2 * len(set(t)) // 3].mean()), "ece": ece(p.max(1), right)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", type=int, default=600)
    ap.add_argument("--random-configs", type=int, default=11)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw = (a.worlds, 60, 100)
    if a.quick:
        nw, a.random_configs = (24, 8, 10), 2
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_hparam_search" + ("_quick" if a.quick else ""), {**vars(a), "default": DEFAULT}) as run:
        tr = episode_items(make_episodes(levels, range(nw[0]), 10, 0.5, rng_seed=1), "fb")
        va = episode_items(make_episodes(levels, range(nw[0], nw[0] + nw[1]), 10, 0.5, rng_seed=2), "fb")
        te = episode_items(make_episodes(levels, range(1000, 1000 + nw[2]), 10, 0.5, rng_seed=3), "fb")
        cfgs = sample_configs(a.random_configs)
        search = []
        for i, cfg in enumerate(cfgs):
            r = [train_eval(cfg, tr, va, te, device, s) for s in (0, 1)]
            search.append({"cfg": cfg, "val": float(np.mean([x["val"]["acc"] for x in r]))})        # test numbers are NOT used for selection
            run.log(f"config {i}{' (default)' if i == 0 else ''}: val {search[-1]['val']:.1%}  {cfg}")
        order = sorted(range(1, len(cfgs)), key=lambda i: -search[i]["val"])[:3]
        confirm = {"default": cfgs[0], **{f"top{k + 1}": cfgs[i] for k, i in enumerate(order)}}
        res = {}
        for name, cfg in confirm.items():
            res[name] = [train_eval(cfg, tr, va, te, device, s) for s in (10, 11, 12)]
            run.log(f"confirm {name}: test {np.mean([x['test']['acc'] for x in res[name]]):.1%}")
        L = ["# Hyperparameter search for the feedback learner", "", f"{len(cfgs)} configs (default + {a.random_configs} random), selected on validation episodes only (2 seeds); "
             "the best 3 and the default confirmed on test with 3 new seeds. Predictions T1-T2 are in the script.", "",
             "## Search (validation accuracy, mean of 2 seeds)", "", "| config | lr | width | layers | epochs | wd | batch | val acc |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for i in sorted(range(len(cfgs)), key=lambda i: -search[i]["val"]):
            c = cfgs[i]
            L.append(f"| {'default' if i == 0 else i} | {c['lr']:.1e} | {c['d']} | {c['layers']} | {c['epochs']} | {c['wd']:.1e} | {c['batch']} | {search[i]['val']:.1%} |")
        L += ["", "## Confirmation on test (3 new seeds; mean (sd))", "", "| config | test accuracy | late positions | ECE | difference to default (points) |", "| --- | ---: | ---: | ---: | ---: |"]
        base = np.array([x["test"]["acc"] for x in res["default"]])
        for name, r in res.items():
            acc = np.array([x["test"]["acc"] for x in r])
            d = 100 * (acc - base)
            L.append(f"| {name} | {100 * acc.mean():.1f}% ({100 * acc.std():.1f}) | {100 * np.mean([x['test']['late'] for x in r]):.1f}% | {np.mean([x['test']['ece'] for x in r]):.3f} | "
                     f"{d.mean():+.1f} ({d.std():.1f}) |")
        run.finish({"search": search, "confirm": {k: v for k, v in res.items()}, "configs": confirm}, "\n".join(L) + "\n")
        (run.dir / "best_config.json").write_text(json.dumps(confirm["top1"]))


if __name__ == "__main__":
    main()

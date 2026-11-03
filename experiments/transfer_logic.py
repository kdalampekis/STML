"""Cross-domain transfer: does what a learner picks up from ARC help it learn logic, and does it matter whether it learned
from human learning histories or from RL learning histories?

Stage 1 (source): pretrain the SHARED core of a relation-embedding encoder (stml/model/shared_gnn.py) on ARC, six ways:
    scratch       no pretraining (random core)
    human         imitate human learning histories (H-ARC, training puzzles only)
    answers       imitate correct answers, same number of steps
    rl_history    imitate the learning history of an RL agent, same number of steps
    rl_policy     the RL agent itself
    logic_small   positive control: logic levels 1-2 only (relations all / no), no ARC
Stage 2 (target): build a logic judge on a copy of each core (new input adapter and relation table, new head) and train on
N = 250 ... 5000 problems; "finetune" updates everything, "frozen" trains only adapter, relation table and head.
Compare accuracy on unseen worlds (levels 1-4, levels 3-4 which use the NEW relations some / some-not, and 12-kind chains).
`reference` = the original per-relation-weight encoder from scratch (is the shared encoder itself OK?).

    python experiments/transfer_logic.py --quick
    python experiments/transfer_logic.py --seeds 3
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stml.model.logic_data as ld  # noqa: E402
from stml.model.graphs import ARC_EDGE_TYPES, ARC_NODE_DIM  # noqa: E402
from stml.model.harc_data import answer_steps, attach_pools, build_graphs, human_steps, load_puzzles, read_records  # noqa: E402
from stml.model.logic_data import EDGE_TYPES, NODE_DIM, make_big_items, make_items  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import STMLLearner, pick_device  # noqa: E402
from stml.model.shared_gnn import DomainEncoder, SharedCore, transfer_core  # noqa: E402
from stml.model.train import evaluate, fit_imitation, train_rl  # noqa: E402
from stml.runlog import Run  # noqa: E402

D, LAYERS, D_REL = 64, 4, 32
SOURCES = ["scratch", "human", "answers", "rl_history", "rl_policy", "logic_small"]


def arc_model(core: SharedCore, device) -> STMLLearner:
    return STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=D, out_match=True,
                       gnn=DomainEncoder(core, ARC_NODE_DIM, len(ARC_EDGE_TYPES))).to(device)


def logic_model(core: SharedCore, device) -> LogicJudge:
    return LogicJudge(D, LAYERS, gnn=DomainEncoder(core, NODE_DIM, len(EDGE_TYPES))).to(device)


def pretrain_sources(a, run, device, seed: int) -> dict[str, SharedCore]:
    """Stage 1 for one pretraining seed. Returns the cores (CPU copies) and logs how good each ARC source learner is."""
    records = read_records(ROOT / "data/processed/harc.jsonl")
    puzzles = attach_pools(load_puzzles(ROOT / "data/raw/harc/arc_data"), records)
    if a.quick:
        puzzles = {k: puzzles[k] for k in sorted(puzzles)[:60]}
    build_graphs(puzzles, cache=None if a.quick else ROOT / "data/processed/harc_graphs.pkl", log=lambda s: None)
    names = sorted(puzzles)
    rnd = random.Random(0)                      # the ARC split is fixed; only initialisation and sampling vary with `seed`
    order = names[:]
    rnd.shuffle(order)
    train, test = order[: len(order) // 2], order[len(order) // 2:]
    h_train = human_steps(records, puzzles, set(train))
    budget = len(h_train)
    cores: dict[str, SharedCore] = {}

    def fresh():
        torch.manual_seed(1000 + seed)
        return SharedCore(D, LAYERS, D_REL, a.core_mode)

    def quality(tag: str, m: STMLLearner) -> None:
        r = evaluate(m, puzzles, test, device)
        run.log(f"seed {seed} source {tag}: ARC solved {r['solved']:.1%}, first try {r['first']:.1%}")
        run.metric({"seed": seed, "source": tag, "arc_solved": r["solved"], "arc_first": r["first"]})

    cores["scratch"] = fresh()
    m = fit_imitation(arc_model(fresh(), device), puzzles, h_train, device, epochs=a.epochs, seed=seed, log=lambda s: None)
    quality("human", m)
    cores["human"] = m.gnn.core.cpu()
    m = fit_imitation(arc_model(fresh(), device), puzzles, answer_steps(puzzles, set(train), budget), device,
                      epochs=a.epochs, seed=seed, log=lambda s: None)
    quality("answers", m)
    cores["answers"] = m.gnn.core.cpu()
    agent = arc_model(fresh(), device)
    hist = train_rl(agent, puzzles, train, device, episodes=a.rl_episodes, seed=seed, log=lambda s: None)
    quality("rl_policy", agent)
    cores["rl_policy"] = agent.gnn.core.cpu()
    take = np.random.default_rng(seed).choice(len(hist), size=min(budget, len(hist)), replace=False)
    m = fit_imitation(arc_model(fresh(), device), puzzles, [hist[i] for i in take], device, epochs=a.epochs, seed=seed,
                      log=lambda s: None)
    quality("rl_history", m)
    cores["rl_history"] = m.gnn.core.cpu()
    return cores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15, help="ARC pretraining epochs")
    ap.add_argument("--rl-episodes", type=int, default=6000)
    ap.add_argument("--target-epochs", type=int, default=10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--core-mode", default="film", choices=["concat", "film", "both", "lowrank", "hyper"], help="how the relation conditions messages (hyper matched the original encoder in dev_shared_encoder round 2)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    ns = (250, 500, 1000, 2000, 5000)
    nw = (300, 50, 100)
    if a.quick:
        a.seeds, a.epochs, a.rl_episodes, a.target_epochs, ns, nw = 1, 1, 160, 2, (200, 500), (30, 10, 20)
    device = pick_device(a.device)
    with Run("transfer_logic" + ("_quick" if a.quick else ""), {**vars(a), "ns": ns, "sources": SOURCES}) as run:
        pool = make_items([1, 2, 3, 4], range(nw[0]), rng_seed=1)
        val = make_items([1, 2, 3, 4], range(nw[0], nw[0] + nw[1]), rng_seed=2)
        small = make_items([1, 2], range(nw[0]), rng_seed=5)                       # positive-control source: only all / no
        small_val = make_items([1, 2], range(nw[0], nw[0] + nw[1]), rng_seed=6)
        test = make_items([1, 2, 3, 4], range(1000, 1000 + nw[2]), rng_seed=3)
        tests = {"levels 1-4": test, "levels 3-4 (new relations)": [i for i in test if i.level >= 3],
                 "12 kinds, chains": make_big_items(12, range(1000, 1000 + (nw[2] // 3 if not a.quick else 5)), rng_seed=12, chain=0.9)}
        run.log("test sets: " + ", ".join(f"{k}: {len(v)}" for k, v in tests.items()))
        acc: dict = {}                                                            # acc[mode][source][n][test] -> list over seeds
        arc_quality: dict = {}
        for seed in range(a.seeds):
            cache = ROOT / "data" / "processed" / "transfer_cores" / f"seed{seed}_{a.core_mode}{'_quick' if a.quick else ''}"
            if all((cache / f"{k}.pt").exists() for k in ("scratch", "human", "answers", "rl_history", "rl_policy")):
                cores = {}
                for k in ("scratch", "human", "answers", "rl_history", "rl_policy"):
                    cores[k] = SharedCore(D, LAYERS, D_REL, a.core_mode)
                    cores[k].load_state_dict(torch.load(cache / f"{k}.pt"))
                run.log(f"seed {seed}: pretrained cores loaded from {cache}")
            else:
                cores = pretrain_sources(a, run, device, seed)
                cache.mkdir(parents=True, exist_ok=True)
                for k, c in cores.items():
                    torch.save(c.state_dict(), cache / f"{k}.pt")     # resume point: pretraining is the expensive part
            torch.manual_seed(seed)
            lj = logic_model(SharedCore(D, LAYERS, D_REL, a.core_mode), device)                # positive control: logic levels 1-2 only
            fit(lj, small, small_val, device, epochs=a.target_epochs, seed=seed, log=lambda s: None)
            cores["logic_small"] = lj.gnn.core.cpu()
            order = pool[:]
            random.Random(seed).shuffle(order)
            for mode in ("finetune", "frozen"):
                for src in SOURCES + ["reference"]:
                    if src == "reference" and mode == "frozen":
                        continue
                    for n in ns:
                        torch.manual_seed(seed)
                        if src == "reference":                                    # the original per-relation encoder, from scratch
                            m = LogicJudge(D, LAYERS).to(device)
                        else:
                            m = logic_model(transfer_core(cores[src], freeze=(mode == "frozen")), device)
                        fit(m, order[:n], val, device, epochs=a.target_epochs, seed=seed, log=lambda s: None)
                        for t, items in tests.items():
                            y = np.array([it.label for it in items])
                            r = float((logits_of(m, items, device).argmax(1).cpu().numpy() == y).mean())
                            acc.setdefault(mode, {}).setdefault(src, {}).setdefault(n, {}).setdefault(t, []).append(r)
                run.log(f"seed {seed} mode {mode}: done")
            run.metric({"seed": seed, "snapshot": acc})
        # ------------------------------------------------------------------ report
        L = ["# Cross-domain transfer: ARC to logic", "",
             f"{a.seeds} seeds. Accuracy on unseen logic worlds after training on N problems (mean over seeds, sd in brackets). "
             "`finetune` = all weights trained; `frozen` = only the new input adapter, relation table and head trained "
             "(a test of the representation). Sources were pretrained on the ARC training half only.", ""]
        for mode in ("finetune", "frozen"):
            for t in tests:
                L += [f"## {mode}, {t}", "", "| source | " + " | ".join(f"N={n}" for n in ns) + " |",
                      "| --- | " + " | ".join("---:" for _ in ns) + " |"]
                for src in SOURCES + (["reference"] if mode == "finetune" else []):
                    L.append(f"| {src} | " + " | ".join(f"{np.mean(acc[mode][src][n][t]):.1%} ({np.std(acc[mode][src][n][t]):.1%})" for n in ns) + " |")
                L.append("")
        L += ["## Human vs RL-history source, paired by seed (points, finetune, levels 1-4)", "",
              "| N | human minus rl_history | human minus answers | human minus scratch | rl_history minus scratch |", "| ---: | ---: | ---: | ---: | ---: |"]
        for n in ns:
            g = lambda s: np.array(acc["finetune"][s][n]["levels 1-4"])
            f = lambda x: f"{100 * x.mean():+.1f} ({100 * x.std():.1f})"
            L.append(f"| {n} | {f(g('human') - g('rl_history'))} | {f(g('human') - g('answers'))} | {f(g('human') - g('scratch'))} | {f(g('rl_history') - g('scratch'))} |")
        run.finish({"acc": acc}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

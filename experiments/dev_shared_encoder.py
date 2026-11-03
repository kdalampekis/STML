"""Is the shared (relation-embedding) encoder as good as the original per-relation encoder? Logic judge, trained from scratch.

Round 2 variants (relation-specific weights that stay shareable): low-rank = shared base matrix plus a relation-specific rank-r term;
hypernetwork = the full per-relation matrix generated from the relation embedding by a shared network; film = best round-1 variant.

The first transfer run showed the shared encoder trailing the original one at every N, which confounds any transfer reading.
Variants: concat (relation embedding concatenated to the source state, the first version), film (the relation gates and shifts
the source state multiplicatively), both, and concat with a larger relation embedding. `original` = RelationalGNN.

    python experiments/dev_shared_encoder.py --quick
    python experiments/dev_shared_encoder.py
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
from stml.model.logic_data import EDGE_TYPES, NODE_DIM, make_big_items, make_items  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.model.shared_gnn import DomainEncoder, SharedCore  # noqa: E402
from stml.runlog import Run  # noqa: E402

VARIANTS = {"original": None, "film (round-1 best)": ("film", 32, 16), "low-rank r=16": ("lowrank", 32, 16),
            "low-rank r=64": ("lowrank", 32, 64), "hypernetwork": ("hyper", 32, 16)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    ns, nw, epochs = (250, 500, 1000, 2000), (300, 50, 100), 10
    if a.quick:
        a.seeds, ns, nw, epochs = 1, (200,), (30, 10, 20), 2
    device = pick_device(None)
    with Run("dev_shared_encoder" + ("_quick" if a.quick else ""), {**vars(a), "variants": list(VARIANTS)}) as run:
        pool = make_items([1, 2, 3, 4], range(nw[0]), rng_seed=1)
        val = make_items([1, 2, 3, 4], range(nw[0], nw[0] + nw[1]), rng_seed=2)
        tests = {"levels 1-4": make_items([1, 2, 3, 4], range(1000, 1000 + nw[2]), rng_seed=3),
                 "12 kinds, chains": make_big_items(12, range(1000, 1000 + (30 if not a.quick else 4)), rng_seed=12, chain=0.9)}
        res = {k: {n: {t: [] for t in tests} for n in ns} for k in VARIANTS}
        for seed in range(a.seeds):
            order = pool[:]
            random.Random(seed).shuffle(order)
            for name, cfg in VARIANTS.items():
                for n in ns:
                    torch.manual_seed(seed)
                    if cfg is None:
                        m = LogicJudge(64, 4).to(device)
                    else:
                        m = LogicJudge(64, 4, gnn=DomainEncoder(SharedCore(64, 4, cfg[1], cfg[0], cfg[2]), NODE_DIM, len(EDGE_TYPES))).to(device)
                    fit(m, order[:n], val, device, epochs=epochs, seed=seed, log=lambda s: None)
                    for t, items in tests.items():
                        y = np.array([it.label for it in items])
                        res[name][n][t].append(float((logits_of(m, items, device).argmax(1).cpu().numpy() == y).mean()))
            run.log(f"seed {seed} done")
        L = ["# Shared encoder variants vs the original encoder (logic, from scratch)", "", f"{a.seeds} seeds; accuracy on unseen worlds, mean (sd).", ""]
        for t in tests:
            L += [f"## {t}", "", "| variant | " + " | ".join(f"N={n}" for n in ns) + " |", "| --- | " + " | ".join("---:" for _ in ns) + " |"]
            for k in VARIANTS:
                L.append(f"| {k} | " + " | ".join(f"{np.mean(res[k][n][t]):.1%} ({np.std(res[k][n][t]):.1%})" for n in ns) + " |")
            L.append("")
        run.finish({k: {str(n): v for n, v in d.items()} for k, d in res.items()}, "\n".join(L))


if __name__ == "__main__":
    main()

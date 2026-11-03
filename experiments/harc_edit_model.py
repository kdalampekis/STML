"""Process-level imitation, step 2: predict the next edited CELL, and test whether a person's own failed attempt explains it.

H1's mechanism: after a wrong answer people repair locally. If that is real, a model that sees the grid the person last SUBMITTED
(and its difference to the current grid) should predict their next edit better than the same model without that information,
specifically on edits made after a failure. Variants (same network, same data, same training):
    uniform          every valid cell equally likely
    adjacent prior   mass on the 8 neighbours of the previous edit (a locality heuristic)
    no history       CNN on current grid, test input, last two edits, colour   (history channels zeroed)
    + failed attempt  the same, plus the last submitted grid, its difference map and a 'has failed' flag
    shuffled history  control: the history channels of a DIFFERENT sample (same distribution, no link to this edit)
Evaluated on edits from unseen puzzles; NLL per edit (nats), top-1 / top-5 accuracy; cluster bootstrap over puzzles for the paired
NLL difference on edits after a failure.

    python experiments/harc_edit_model.py --quick
    python experiments/harc_edit_model.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.model.edit_model import H, EditNet, build_rows, make_input  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run, table  # noqa: E402

VARIANTS = {"no history": (False, False), "+ failed attempt": (True, False), "shuffled history": (True, True)}


def batches(n: int, bs: int, rng=None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def take(d: dict, idx) -> dict:
    return {k: v[idx] for k, v in d.items()}


def train(d: dict, device, history: bool, shuffle: bool, epochs: int, seed: int, log) -> EditNet:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = EditNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(d["y"])
    for ep in range(epochs):
        tot = 0.0
        for idx in batches(n, 128, rng):
            b = take(d, idx)
            x, valid = make_input(b, device, history, shuffle)
            loss = F.cross_entropy(net(x, valid), torch.as_tensor(b["y"], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        log(f"  epoch {ep + 1}/{epochs}: train NLL {tot / n:.3f}")
    return net


@torch.no_grad()
def score(net, d: dict, device, history: bool, shuffle: bool) -> dict:
    net.eval()
    nll, top1, top5 = [], [], []
    for idx in batches(len(d["y"]), 256):
        b = take(d, idx)
        x, valid = make_input(b, device, history, shuffle)
        lg = net(x, valid)
        y = torch.as_tensor(b["y"], device=device)
        nll.append(F.cross_entropy(lg, y, reduction="none").cpu())
        top = lg.topk(5, 1).indices
        top1.append((top[:, 0] == y).cpu())
        top5.append((top == y[:, None]).any(1).cpu())
    return {"nll": torch.cat(nll).numpy(), "top1": torch.cat(top1).numpy(), "top5": torch.cat(top5).numpy()}


def heuristics(d: dict) -> dict:
    """Uniform and adjacent-to-last-edit baselines (exact NLL, no training)."""
    valid = (d["grid"] >= 0).reshape(len(d["y"]), -1)
    nv = valid.sum(1)
    uni_nll = np.log(nv)
    adj_nll, adj_top1, adj_top5 = [], [], []
    for i in range(len(d["y"])):
        x, y = d["pos1"][i]
        if x < 0:
            adj_nll.append(np.log(nv[i])); adj_top1.append(False); adj_top5.append(False)
            continue
        nb = [(yy * H + xx) for yy in range(max(0, y - 1), min(H, y + 2)) for xx in range(max(0, x - 1), min(H, x + 2))
              if valid[i, yy * H + xx]]
        p = np.full(H * H, 0.3 / max(1, nv[i] - len(nb))) * valid[i]
        for c in nb:
            p[c] = 0.7 / len(nb)
        adj_nll.append(-np.log(p[d["y"][i]] + 1e-12))
        adj_top1.append(d["y"][i] in nb[:1]); adj_top5.append(d["y"][i] in nb[:5])
    return {"uniform": {"nll": uni_nll, "top1": 1.0 / nv, "top5": np.minimum(5.0 / nv, 1.0)},
            "adjacent prior": {"nll": np.array(adj_nll), "top1": np.array(adj_top1, float), "top5": np.array(adj_top5, float)}}


def cluster_boot(diff: np.ndarray, tasks: np.ndarray, draws: int = 2000, seed: int = 0):
    """Mean and 95% CI of a per-edit difference, resampling puzzles."""
    rng = np.random.default_rng(seed)
    u = np.unique(tasks)
    sums = {t: (diff[tasks == t].sum(), (tasks == t).sum()) for t in u}
    s, c = np.array([sums[t][0] for t in u]), np.array([sums[t][1] for t in u])
    est = []
    for _ in range(draws):
        i = rng.integers(len(u), size=len(u))
        est.append(s[i].sum() / c[i].sum())
    return float(diff.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=120000)
    ap.add_argument("--test", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.train, a.test, a.epochs, a.seeds = 3000, 1500, 1, 1
    device = pick_device(a.device)
    with Run("harc_edit_model" + ("_quick" if a.quick else ""), {**vars(a), "device_used": str(device)}) as run:
        D = build_rows(ROOT / "data/raw/harc/data/data.csv", ROOT / "data/raw/harc/arc_data", a.train, a.test)
        tr, te = D["train"], D["test"]
        failed = te["n_wrong"] > 0
        run.log(f"edits: train {len(tr['y'])}, test {len(te['y'])} ({failed.mean():.1%} after a failure); puzzles in test {len(np.unique(te['task']))}")
        res = {"uniform": [], "adjacent prior": []}
        h = heuristics(te)
        for k in res:
            res[k].append(h[k])
        for name, (hist, shuf) in VARIANTS.items():
            res[name] = []
            for seed in range(a.seeds):
                run.log(f"{name}, seed {seed}")
                net = train(tr, device, hist, shuf, a.epochs, seed, run.log)
                res[name].append(score(net, te, device, hist, shuf))
        def summ(k, key, mask=None):
            return float(np.mean([(r[key] if mask is None else r[key][mask]).mean() for r in res[k]]))
        rows = {}
        for k in res:
            rows[k] = {"nll": summ(k, "nll"), "top1": summ(k, "top1"), "top5": summ(k, "top5"),
                       "nll_f": summ(k, "nll", failed), "top1_f": summ(k, "top1", failed), "nll_1": summ(k, "nll", ~failed), "top1_1": summ(k, "top1", ~failed)}
        cols = [("nll", "NLL all", "{:.3f}"), ("top1", "Top-1", "{:.1%}"), ("top5", "Top-5", "{:.1%}"),
                ("nll_1", "NLL first attempt", "{:.3f}"), ("nll_f", "NLL after a failure", "{:.3f}"), ("top1_f", "Top-1 after a failure", "{:.1%}")]
        # paired differences on edits after a failure (mean over seeds of the per-edit NLL)
        mean_nll = lambda k: np.mean([r["nll"] for r in res[k]], 0)
        stats = {}
        for a_, b_ in (("+ failed attempt", "no history"), ("shuffled history", "no history"), ("+ failed attempt", "shuffled history")):
            diff = (mean_nll(a_) - mean_nll(b_))
            stats[f"{a_} minus {b_}"] = {"after_failure": cluster_boot(diff[failed], te["task"][failed]),
                                         "first_attempt": cluster_boot(diff[~failed], te["task"][~failed])}
        L = ["# Process-level imitation, step 2: predicting the next edited cell", "",
             f"{len(tr['y']):,} training edits, {len(te['y']):,} test edits from unseen puzzles ({failed.mean():.1%} after a failure). "
             f"{a.seeds} seeds, {a.epochs} epochs. NLL in nats per edit (uniform guessing = {rows['uniform']['nll']:.2f}).", "",
             table(rows, cols), "",
             "## Paired NLL difference (negative = the first model is better), 95% CI over puzzles", "",
             "| comparison | after a failure | first attempt |", "| --- | ---: | ---: |"]
        for k, v in stats.items():
            f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
            L.append(f"| {k} | {f(v['after_failure'])} | {f(v['first_attempt'])} |")
        L += ["", "On first-attempt edits nothing has been submitted yet, so '+ failed attempt' and 'no history' receive identical inputs: those cells are a "
                  "check that the two networks are equally good where history cannot matter. The shuffled control also perturbs first-attempt edits "
                  "(it injects another sample's history), so its first-attempt difference is expected.", ""]
        run.finish({"rows": rows, "stats": stats}, "\n".join(L))


if __name__ == "__main__":
    main()

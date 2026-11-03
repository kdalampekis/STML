"""H2 in the logic domain: do structured critiques teach faster than a right/wrong signal?

Same judge, same problems, one pass over N training problems, different feedback after each answer:
  scalar       only whether the learner's sampled answer was right. Two ways to learn from that one bit:
               REINFORCE with a batch baseline, and binary cross-entropy on the chosen answer's probability
               (the stronger, lower-variance use of the same information)
  label        the true status (follows / contradicted / open): a full-information label, no 'why'
  label + why  the true status AND which premises the proof depends on (the referee's unsat core), learned
               through a per-premise head (the Socratic critique's 'which premises are responsible')
Accuracy is measured after N = 500 ... 20000 problems on unseen worlds: levels 1-4, level 5, and large chain
worlds (12 kinds, deep proofs). Size-independent node features throughout. The comparison label vs
label + why is the one that isolates the structured part of the critique; scalar vs label is mostly about
how much information the signal carries.

    python experiments/logic_h2_feedback.py --quick
    python experiments/logic_h2_feedback.py --seeds 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stml.model.logic_data as ld  # noqa: E402
from stml.model.logic_judge import LogicJudge, logits_of  # noqa: E402
from stml.model.logic_data import make_big_items, make_items  # noqa: E402
from stml.runlog import Run  # noqa: E402

ARMS = ("scalar (REINFORCE)", "scalar (BCE)", "label", "label + why")


def train_arm(arm: str, items: list, device, seed: int, batch: int = 16, lr: float = 1e-3, why_weight: float = 0.5) -> LogicJudge:
    torch.manual_seed(seed)
    m = LogicJudge(64, 4).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    m.train()
    for i in range(0, len(items), batch):          # one pass: every problem is seen once, with one piece of feedback
        chunk = items[i:i + batch]
        y = torch.as_tensor([it.label for it in chunk], device=device)
        if arm == "label + why":
            logits, core_logits = m(chunk, device, return_core=True)
            target = torch.zeros(sum(it.n_prem for it in chunk), device=device)
            off = 0
            for it in chunk:
                for j in it.core:
                    target[off + j] = 1.0
                off += it.n_prem
            loss = F.cross_entropy(logits, y) + why_weight * F.binary_cross_entropy_with_logits(core_logits, target)
        elif arm == "label":
            loss = F.cross_entropy(m(chunk, device), y)
        elif arm == "scalar (BCE)":                 # same single bit, used as a label for the chosen answer
            logits = m(chunk, device)
            a = torch.distributions.Categorical(logits=logits).sample()
            p = logits.softmax(1).gather(1, a[:, None]).squeeze(1).clamp(1e-6, 1 - 1e-6)
            r = (a == y).float()
            loss = -(r * p.log() + (1 - r) * (1 - p).log()).mean()
        else:                                       # scalar (REINFORCE): sample an answer, learn only from right/wrong
            logits = m(chunk, device)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            r = (a == y).float()
            loss = -((r - r.mean()).detach() * dist.log_prob(a)).mean() - 0.01 * dist.entropy().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    sizes = (500, 1000, 2000, 5000, 10000, 20000)
    nw, big = (300, 100), 40
    if a.quick:
        a.seeds, sizes, nw, big = 1, (200, 800), (30, 15), 6
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with Run("logic_h2_feedback" + ("_quick" if a.quick else ""), {**vars(a), "sizes": sizes, "arms": ARMS}) as run:
        pool = make_items([1, 2, 3, 4], range(nw[0]), rng_seed=1)
        tests = {"levels 1-4": make_items([1, 2, 3, 4], range(1000, 1000 + nw[1]), rng_seed=3),
                 "level 5": make_items([5], range(1000, 1000 + nw[1]), rng_seed=4),
                 "12 kinds, chains": make_big_items(12, range(1000, 1000 + big), rng_seed=12, chain=0.9)}
        run.log(f"pool {len(pool)} problems; tests " + ", ".join(f"{k}: {len(v)}" for k, v in tests.items()))
        res = {arm: {n: {t: [] for t in tests} for n in sizes} for arm in ARMS}
        for seed in range(a.seeds):
            order = pool[:]
            random.Random(seed).shuffle(order)
            for n in sizes:
                for arm in ARMS:
                    m = train_arm(arm, order[:n], device, seed)
                    for t, items in tests.items():
                        lg = logits_of(m, items, device).argmax(1).cpu().numpy()
                        res[arm][n][t].append(float((lg == np.array([it.label for it in items])).mean()))
            run.log(f"seed {seed} done")
            run.metric({"seed": seed, "snapshot": {arm: {str(n): {t: v[-1] for t, v in d.items()} for n, d in r.items()} for arm, r in res.items()}})
        lines = ["# H2 in the logic domain: scalar vs label vs label + why", "",
                 f"{a.seeds} seeds; one pass over N problems; accuracy on unseen worlds (mean over seeds, sd in brackets). "
                 "Chance is 33%, majority class 40-56%.", ""]
        for t in tests:
            lines += [f"## {t}", "", "| N problems | " + " | ".join(ARMS) + " |", "| ---: | " + " | ".join("---:" for _ in ARMS) + " |"]
            for n in sizes:
                lines.append(f"| {n} | " + " | ".join(f"{np.mean(res[arm][n][t]):.1%} ({np.std(res[arm][n][t]):.1%})" for arm in ARMS) + " |")
            lines.append("")
        lines += ["## Paired difference, label + why minus label (points, mean over seeds)", "", "| N | " + " | ".join(tests) + " |",
                  "| ---: | " + " | ".join("---:" for _ in tests) + " |"]
        for n in sizes:
            lines.append(f"| {n} | " + " | ".join(f"{100 * (np.mean(res['label + why'][n][t]) - np.mean(res['label'][n][t])):+.1f}" for t in tests) + " |")
        run.finish({arm: {str(n): d for n, d in r.items()} for arm, r in res.items()}, "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

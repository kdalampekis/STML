"""Training and evaluating the STML learner.

  Stage A  imitate learning histories (cross-entropy on the next choice, plus the
           confidence head learning whether that choice was right).
  Stage B  reinforcement learning with right/wrong feedback (REINFORCE with a
           moving baseline and an entropy bonus); every choice it makes is recorded,
           so its learning history can itself be imitated (the Algorithm Distillation
           control for H1).
  evaluate solve rate in 3 attempts, first-try accuracy, recovery after a wrong
           answer, human-likeness (log-likelihood of people's actual choices), and
           calibration of the confidence head (Brier score, ECE).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .harc_data import Puzzle, Step
from .network import STMLLearner

Log = Callable[[str], None]


def _group(steps: list[Step]) -> dict[str, dict[tuple, list[Step]]]:
    """puzzle -> history prefix -> steps; one transformer pass serves every step with that prefix."""
    out: dict[str, dict[tuple, list[Step]]] = defaultdict(lambda: defaultdict(list))
    for s in steps:
        out[s.puzzle][tuple(s.tried)].append(s)
    return out


def _step_losses(model: STMLLearner, enc: dict, groups: dict[tuple, list[Step]], conf_weight: float):
    ce, bce, n = 0.0, 0.0, 0
    for prefix, ss in groups.items():
        scores, conf = model.logits(enc, list(prefix))
        logp = F.log_softmax(scores, 0)
        idx = torch.tensor([s.choice for s in ss], device=scores.device)
        y = torch.tensor([float(s.correct) for s in ss], device=scores.device)
        ce = ce - logp[idx].sum()
        bce = bce + F.binary_cross_entropy_with_logits(conf[idx], y, reduction="sum")
        n += len(ss)
    return ce, bce, n


def fit_imitation(model: STMLLearner, puzzles: dict[str, Puzzle], steps: list[Step], device: torch.device,
                  epochs: int = 15, lr: float = 3e-4, batch: int = 16, conf_weight: float = 0.5,
                  seed: int = 0, log: Log = print, metric=None, aux_weight: float = 0.0,
                  aux_per_puzzle: int = 8) -> STMLLearner:
    """Stage A. aux_weight > 0 adds the ARCTraj progress regression on puzzles that have progress grids."""
    torch.manual_seed(seed)
    rnd = random.Random(seed)
    grouped = _group(steps)
    names = sorted(grouped)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(epochs):
        rnd.shuffle(names)
        tot_ce = tot_n = 0.0
        for b in range(0, len(names), batch):
            chunk = names[b:b + batch]
            for nm in chunk:
                p = puzzles[nm]
                p.aux_active = (rnd.sample(range(len(p.aux_graphs)), min(aux_per_puzzle, len(p.aux_graphs)))
                                if aux_weight > 0 else [])
            encs = model.encode_puzzles([puzzles[n] for n in chunk], device)
            ce = bce = aux = 0.0
            n = 0
            for name, enc in zip(chunk, encs):
                c, bc, k = _step_losses(model, enc, grouped[name], conf_weight)
                ce, bce, n = ce + c, bce + bc, n + k
                if "aux" in enc:
                    y = torch.as_tensor(puzzles[name].aux_y[puzzles[name].aux_active], device=device)
                    aux = aux + F.mse_loss(torch.sigmoid(model.progress(enc["aux"]).squeeze(1)), y, reduction="sum")
            for nm in chunk:
                puzzles[nm].aux_active = []
            loss = (ce + conf_weight * bce + aux_weight * aux) / max(n, 1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_ce += float(ce)
            tot_n += n
        msg = f"stage A epoch {ep + 1}/{epochs}: choice loss {tot_ce / max(tot_n, 1):.3f} over {int(tot_n)} steps"
        log(msg)
        if metric:
            metric({"stage": "A", "epoch": ep + 1, "choice_loss": tot_ce / max(tot_n, 1)})
    return model


def train_rl(model: STMLLearner, puzzles: dict[str, Puzzle], tasks: list[str], device: torch.device,
             episodes: int = 6000, lr: float = 1e-4, batch: int = 16, entropy: float = 0.01,
             conf_weight: float = 0.5, seed: int = 0, log: Log = print, metric=None) -> list[Step]:
    """Stage B. Returns every choice made while learning (an Algorithm Distillation source)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history: list[Step] = []
    baseline, done, recent = 0.0, 0, []
    model.train()
    while done < episodes:
        chunk = [tasks[i] for i in rng.integers(len(tasks), size=batch)]
        encs = model.encode_puzzles([puzzles[n] for n in chunk], device)
        loss = 0.0
        for name, enc in zip(chunk, encs):
            p = puzzles[name]
            tried: list[tuple[int, bool]] = []
            logps, ents, confs, oks = [], [], [], []
            reward = 0.0
            for k in range(min(3, len(p.candidates))):
                scores, conf = model.logits(enc, tried)
                dist = torch.distributions.Categorical(logits=scores)
                i = int(dist.sample())
                ok = p.candidates[i] == p.target
                logps.append(dist.log_prob(torch.tensor(i, device=device)))
                ents.append(dist.entropy())
                confs.append(conf[i])
                oks.append(float(ok))
                history.append(Step(name, list(tried), i, ok))
                if ok:
                    reward = 1.0 - 0.25 * k
                    break
                tried.append((i, False))
            adv = reward - baseline
            conf_t = torch.stack(confs)
            loss = loss - adv * torch.stack(logps).sum() - entropy * torch.stack(ents).sum() \
                + conf_weight * F.binary_cross_entropy_with_logits(conf_t, torch.tensor(oks, device=device), reduction="sum")
            recent.append(reward > 0)
            baseline += 0.02 * (reward - baseline)
        opt.zero_grad()
        (loss / batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        done += batch
        if done % (batch * 25) == 0:
            rate = float(np.mean(recent[-400:]))
            log(f"stage B episodes {done}/{episodes}: recent solve rate {rate:.1%}")
            if metric:
                metric({"stage": "B", "episodes": done, "recent_solve_rate": rate})
    return history


@torch.no_grad()
def play(model: STMLLearner, enc: dict, p: Puzzle, attempts: int = 3) -> tuple[bool, int, list[tuple[float, bool]]]:
    """Greedy play with right/wrong feedback. Returns (solved, attempts used, [(confidence, right)])."""
    tried: list[tuple[int, bool]] = []
    confs = []
    for k in range(min(attempts, len(p.candidates))):
        scores, conf = model.logits(enc, tried)
        i = int(torch.argmax(scores))
        ok = p.candidates[i] == p.target
        confs.append((float(torch.sigmoid(conf[i])), ok))
        if ok:
            return True, k + 1, confs
        tried.append((i, False))
    return False, attempts, confs


@torch.no_grad()
def evaluate(model: STMLLearner, puzzles: dict[str, Puzzle], tasks: list[str], device: torch.device,
             human: list[Step] | None = None, batch: int = 32) -> dict:
    model.eval()
    solved, first, conf_pairs = [], [], []
    loglik_all, loglik_rev = [], []
    grouped = _group(human or [])
    for b in range(0, len(tasks), batch):
        chunk = tasks[b:b + batch]
        encs = model.encode_puzzles([puzzles[n] for n in chunk], device)
        for name, enc in zip(chunk, encs):
            s, k, c = play(model, enc, puzzles[name])
            solved.append(s)
            first.append(s and k == 1)
            conf_pairs += c
            for prefix, ss in grouped.get(name, {}).items():
                scores, _ = model.logits(enc, list(prefix))
                lp = F.log_softmax(scores, 0)
                for st in ss:
                    v = float(lp[st.choice])
                    loglik_all.append(v)
                    if prefix:
                        loglik_rev.append(v)
    solved, first = np.array(solved), np.array(first)
    conf = np.array([c for c, _ in conf_pairs])
    right = np.array([float(o) for _, o in conf_pairs])
    bins = np.clip((conf * 10).astype(int), 0, 9)
    ece = sum(abs(conf[bins == k].mean() - right[bins == k].mean()) * (bins == k).mean()
              for k in range(10) if (bins == k).any())
    model.train()
    return {
        "solved": float(solved.mean()),
        "first": float(first.mean()),
        "recover": float(solved[~first].mean()) if (~first).any() else float("nan"),
        "human_loglik": float(np.mean(loglik_all)) if loglik_all else float("nan"),
        "human_loglik_revisions": float(np.mean(loglik_rev)) if loglik_rev else float("nan"),
        "brier": float(np.mean((conf - right) ** 2)) if len(conf) else float("nan"),
        "ece": float(ece),
        "_solved": solved.tolist(),
    }


def random_reference(puzzles: dict[str, Puzzle], tasks: list[str], human: list[Step]) -> dict:
    s = np.array([min(3, len(puzzles[t].candidates)) / len(puzzles[t].candidates) for t in tasks])
    f = np.array([1 / len(puzzles[t].candidates) for t in tasks])
    ll = [-np.log(len(puzzles[st.puzzle].candidates) - len(st.tried)) for st in human]
    llr = [-np.log(len(puzzles[st.puzzle].candidates) - len(st.tried)) for st in human if st.tried]
    return {"solved": float(s.mean()), "first": float(f.mean()), "recover": float(((s - f) / (1 - f + 1e-9)).mean()),
            "human_loglik": float(np.mean(ll)), "human_loglik_revisions": float(np.mean(llr)),
            "brier": float("nan"), "ece": float("nan")}


def people_reference(records: list[dict], tasks: set[str]) -> dict:
    by = defaultdict(list)
    for r in records:
        if r["task"]["id"] in tasks:
            by[r["task"]["id"]].append((r["outcome"]["solved"], r["attempts"][0]["feedback"]["correct"]))
    solved = float(np.mean([np.mean([s for s, _ in v]) for v in by.values()]))
    first = float(np.mean([np.mean([f for _, f in v]) for v in by.values()]))
    rec = [s for v in by.values() for s, f in v if not f]
    return {"solved": solved, "first": first, "recover": float(np.mean(rec)),
            "human_loglik": float("nan"), "human_loglik_revisions": float("nan"), "brier": float("nan"), "ece": float("nan")}

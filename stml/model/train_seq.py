"""Learning to learn across puzzles: the learner carries its experience from one puzzle to the next.

A sequence is several puzzles played one after another (H-ARC people did about five, in random order). With
context, the history transformer sees the earlier puzzles of the sequence, every attempt and its right/wrong
feedback, before the current one. Without context the same model treats every puzzle alone (the control).

  human_sequences       people's own puzzle sequences, restricted to a given set of puzzles
  fit_imitation_seq     Stage A on those sequences (teacher forcing: the context is the person's own history)
  train_rl_seq          meta-RL: sample sequences of random training puzzles, play them with own context
  evaluate_seq          the learner plays random sequences of unseen puzzles with its OWN context;
                        results are reported by position in the sequence
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .harc_data import Puzzle
from .network import STMLLearner

MAX_CONTEXT = 4


def human_sequences(records: list[dict], puzzles: dict[str, Puzzle], tasks: set[str]) -> list[list[tuple[str, list[tuple[int, bool]]]]]:
    """Per person: [(puzzle, [(candidate index, right?), ...]), ...] in the order they did them."""
    by = defaultdict(list)
    for r in records:
        name = r["task"]["id"]
        if name not in tasks or name not in puzzles:
            continue
        index = {c: i for i, c in enumerate(puzzles[name].candidates)}
        att = []
        for a in r["attempts"]:
            i = index.get(a["commit"]["answer"])
            if i is None or any(i == j for j, _ in att):
                break
            ok = bool(a["feedback"]["correct"])
            att.append((i, ok))
            if ok:
                break
        if att:
            by[r["learner"]["id"]].append((r["task"]["order"], name, att))
    return [[(n, a) for _, n, a in sorted(v)] for v in by.values()]


def _ctx(enc_list: list[dict], episodes: list[list[tuple[int, bool]]], j: int, use_context: bool):
    if not use_context or j == 0:
        return None
    lo = max(0, j - MAX_CONTEXT)
    return [(enc_list[k], episodes[k]) for k in range(lo, j)]


def fit_imitation_seq(model: STMLLearner, puzzles: dict[str, Puzzle], seqs: list, device: torch.device,
                      use_context: bool = True, epochs: int = 15, lr: float = 3e-4, batch: int = 8,
                      conf_weight: float = 0.5, seed: int = 0, log=print, metric=None) -> STMLLearner:
    torch.manual_seed(seed)
    rnd = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    seqs = list(seqs)
    model.train()
    for ep in range(epochs):
        rnd.shuffle(seqs)
        tot = n_tot = 0.0
        for b in range(0, len(seqs), batch):
            chunk = seqs[b:b + batch]
            flat = [puzzles[n] for s in chunk for n, _ in s]
            encs = model.encode_puzzles(flat, device)
            ce = bce = 0.0
            n = 0
            pos = 0
            for s in chunk:
                enc_s = encs[pos:pos + len(s)]
                pos += len(s)
                episodes = []
                for j, (name, att) in enumerate(s):
                    ctx = _ctx(enc_s, episodes, j, use_context)
                    for k, (i, ok) in enumerate(att):
                        scores, conf = model.logits(enc_s[j], list(att[:k]), ctx)
                        ce = ce - F.log_softmax(scores, 0)[i]
                        bce = bce + F.binary_cross_entropy_with_logits(conf[i], torch.tensor(float(ok), device=device))
                        n += 1
                    episodes.append(list(att))
            loss = (ce + conf_weight * bce) / max(n, 1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(ce.detach())
            n_tot += n
        log(f"stage A (sequences) epoch {ep + 1}/{epochs}: choice loss {tot / max(n_tot, 1):.3f} over {int(n_tot)} steps")
        if metric:
            metric({"stage": "A-seq", "epoch": ep + 1, "choice_loss": tot / max(n_tot, 1)})
    return model


def _play_episode(model, enc, p: Puzzle, ctx, device, sample: bool, attempts: int = 3):
    """One puzzle. Returns (episode as [(cand, right)], solved, attempts used, logps, entropies, confs, oks)."""
    tried: list[tuple[int, bool]] = []
    logps, ents, confs, oks = [], [], [], []
    for k in range(min(attempts, len(p.candidates))):
        scores, conf = model.logits(enc, tried, ctx)
        if sample:
            dist = torch.distributions.Categorical(logits=scores)
            i = int(dist.sample())
            logps.append(dist.log_prob(torch.tensor(i, device=device)))
            ents.append(dist.entropy())
        else:
            i = int(torch.argmax(scores))
        ok = p.candidates[i] == p.target
        confs.append(conf[i])
        oks.append(float(ok))
        if ok:
            return tried + [(i, True)], True, k + 1, logps, ents, confs, oks
        tried.append((i, False))
    return tried, False, attempts, logps, ents, confs, oks


def train_rl_seq(model: STMLLearner, puzzles: dict[str, Puzzle], tasks: list[str], device: torch.device,
                 use_context: bool = True, episodes: int = 6000, seq_len: int = 5, lr: float = 1e-4,
                 batch: int = 4, entropy: float = 0.01, conf_weight: float = 0.5, seed: int = 0,
                 log=print, metric=None) -> None:
    """Meta-RL (reward 1 - 0.25 (attempt - 1) if solved). `episodes` counts puzzles played, as in train_rl."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    baseline, done, recent = 0.0, 0, []
    model.train()
    while done < episodes:
        seqs = [[tasks[i] for i in rng.choice(len(tasks), size=seq_len, replace=False)] for _ in range(batch)]
        encs = model.encode_puzzles([puzzles[n] for s in seqs for n in s], device)
        loss = 0.0
        for si, s in enumerate(seqs):
            enc_s = encs[si * seq_len:(si + 1) * seq_len]
            eps: list = []
            for j, name in enumerate(s):
                ctx = _ctx(enc_s, eps, j, use_context)
                ep, solved, used, logps, ents, confs, oks = _play_episode(model, enc_s[j], puzzles[name], ctx, device, True)
                reward = (1.0 - 0.25 * (used - 1)) if solved else 0.0
                adv = reward - baseline
                loss = loss - adv * torch.stack(logps).sum() - entropy * torch.stack(ents).sum() \
                    + conf_weight * F.binary_cross_entropy_with_logits(torch.stack(confs), torch.tensor(oks, device=device), reduction="sum")
                baseline += 0.02 * (reward - baseline)
                recent.append(solved)
                eps.append(ep)
        opt.zero_grad()
        (loss / (batch * seq_len)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        done += batch * seq_len
        if done % (batch * seq_len * 15) == 0:
            rate = float(np.mean(recent[-400:]))
            log(f"stage B (sequences) episodes {done}/{episodes}: recent solve rate {rate:.1%}")
            if metric:
                metric({"stage": "B-seq", "episodes": done, "recent_solve_rate": rate})


@torch.no_grad()
def evaluate_seq(model: STMLLearner, puzzles: dict[str, Puzzle], tasks: list[str], device: torch.device,
                 use_context: bool = True, seq_len: int = 5, repeats: int = 4, seed: int = 0, batch: int = 8) -> dict:
    """Random sequences of unseen puzzles, played greedily with the learner's own context. By position."""
    model.eval()
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(repeats):                       # every puzzle appears `repeats` times, at random positions
        order = list(rng.permutation(len(tasks)))
        seqs += [[tasks[i] for i in order[b:b + seq_len]] for b in range(0, len(order) - seq_len + 1, seq_len)]
    solved = defaultdict(list)
    first = defaultdict(list)
    for b in range(0, len(seqs), batch):
        chunk = seqs[b:b + batch]
        encs = model.encode_puzzles([puzzles[n] for s in chunk for n in s], device)
        for si, s in enumerate(chunk):
            enc_s = encs[si * seq_len:(si + 1) * seq_len]
            eps: list = []
            for j, name in enumerate(s):
                ctx = _ctx(enc_s, eps, j, use_context)
                ep, ok, used, *_ = _play_episode(model, enc_s[j], puzzles[name], ctx, device, False)
                solved[j].append(ok)
                first[j].append(ok and used == 1)
                eps.append(ep)
    model.train()
    pos = sorted(solved)
    return {"solved_by_position": [float(np.mean(solved[j])) for j in pos],
            "first_by_position": [float(np.mean(first[j])) for j in pos],
            "solved": float(np.mean([x for j in pos for x in solved[j]])),
            "first": float(np.mean([x for j in pos for x in first[j]])),
            "n_sequences": len(seqs)}

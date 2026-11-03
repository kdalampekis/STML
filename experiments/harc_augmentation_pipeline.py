"""Q29 (the missing control of Q28): H1 with EQUAL DATA. Is imitating human learning histories before RL better than imitating ANSWERS before RL, under the same augmentation and the same RL budget?

Same puzzles, splits (seeds 0-2), augmentation (3 copies per training puzzle: one dihedral transform + one colour permutation applied to every grid and candidate) and pure learner as harc_augmentation.py.
Conditions, solved in 3 attempts on unseen puzzles (RL: 6000 episodes on the training puzzles and their 3 augmented copies; same epochs and splits as Q27/Q28):
  S_aug    imitate human learning on the augmented steps, then RL (the Q28 pipeline, rerun for a paired comparison)
  SA_aug   imitate ANSWERS on the augmented puzzles (same number of steps as S_aug's imitation phase), then the same RL
  R_aug    RL from scratch on the same augmented puzzles (reference)
PREDICTIONS, written before running (Q28: S_aug 60.4, R_aug 51.9; Q27: imitate answers with augmentation 59.3, imitate humans 57.3):
  C1  SA_aug is within +-2 points of S_aug (human histories add nothing over answers in the full pipeline).
  C2  both S_aug and SA_aug beat R_aug by >= 4 points.
  Falsifier of C1: S_aug beats SA_aug by more than 2 points and more than the split sd.
RESULT of the first run (splits 0-2): S_aug 62.5, SA_aug 59.6, R_aug 54.0; S_aug - SA_aug = +2.9 (1.0); C1 was falsified.
Q30 (--seed0 3 --seeds 3 --synthetic-failures): replication on fresh splits (3-5) and a mechanism control. SF_aug = imitation of ANSWERS where every puzzle is first answered wrongly (a random wrong candidate, feedback 'wrong')
  and then correctly, with the same number of steps as the human imitation, then the same RL. PREDICTIONS, written before running:
  D1  S_aug beats SA_aug by >= 2 points on the fresh splits (replication); falsifier: S_aug - SA_aug <= +1.
Q40 (--arctraj --seeds 3, pure family, splits 0-2): MORE HUMAN DATA. The ARCTraj first-attempt steps of the training puzzles are added to the H-ARC human steps (ARCTraj only on training puzzles); the answers arm has the same total number of steps.
  Reference (Q29, H-ARC only): humans-then-RL 62.5, answers-then-RL 59.6, RL from scratch 54.0; lead over answers +2.9. Q38 showed the human-specific lead needs the volume of human steps (+0.1 at 25%, -1.1 at 10%).
  PREDICTIONS, written before running: R1 adding ARCTraj raises the lead of humans-then-RL over answers-then-RL to >= +4 (more human volume, more human-specific benefit); R2 humans-then-RL itself rises by >= 1.5 over Q29. Falsifier of R1: the lead stays <= +3.
Q38 (--human-frac 0.1 and 0.25, pure family, splits 0-2): HOW MUCH HUMAN DATA IS NEEDED? The imitation phase uses only a random fraction of the human steps (and, in SA_aug, the same number of answer steps); the RL phase is unchanged.
  This sizes the benefit for a human study of limited size. Reference (full human steps, Q29): humans-then-RL minus answers-then-RL +2.9; humans-then-RL 62.5, answers-then-RL 59.6, RL from scratch 54.0.
  PREDICTIONS, written before running: Q1 with 25% of the human steps the lead over answers-then-RL is still >= +1.5 (the benefit saturates early); Q2 with 10% it is <= +1.5 and humans-then-RL is still >= 3 points above RL from scratch (the imitation phase helps even with little data).
Q37 (--pretrain-task --seeds 3, pure family, splits 0-2): PUZZLE-LEVEL pretext on the demonstration pairs of the split's training puzzles (and their augmented copies): the task vector built from the other pairs plus the held-out INPUT must predict a descriptor of the held-out OUTPUT
  (shape, palette size, density, background share, symmetries, same shape as the input), the regularities the hand features of Q34 encode. Trains gnn, pair and task_in. Reference (Q29, no pretext): humans-then-RL 62.5, answers-then-RL 59.6, RL from scratch 54.0.
  PREDICTIONS, written before running (the hybrid result suggests puzzle-level regularities were the missing perception; masked-node pretraining, Q36, did not help):
  P-c  the pretext raises RL from scratch and answers-then-RL by >= 2 points (it supplies the regularities the hand features gave).
  P-d  the lead of humans-then-RL over answers-then-RL falls below +2 (if the human-history advantage was a weak-perception effect).
  Falsifier of P-c: no pipeline gains more than its split sd.
Q36 (--pretrain --seeds 3, pure family, splits 0-2): the object-graph encoder is first trained by MASKED NODE-ATTRIBUTE PREDICTION on the grids of the split's training puzzles and their augmented copies (6 epochs, no labels), then used
  to initialise every pipeline. This is the 'learn to perceive' test the hybrid result (Q34) left open. Reference (Q29, same splits, no pretraining): humans-then-RL 62.5, answers-then-RL 59.6, RL from scratch 54.0. PREDICTIONS, written before running:
  P-a  pretraining raises humans-then-RL and answers-then-RL by >= 2 points and RL from scratch by >= 2 (better perception helps every pipeline).
  P-b  the lead of humans-then-RL over answers-then-RL falls below +2 (as perception improves the human-history advantage shrinks, as with the hand features in Q34).
  Falsifier of P-a: no pipeline gains more than its split sd.
Q34 (--family hybrid --rl-history --seeds 3, splits 0-2): the HYBRID learner (pure + the 12 hand features of baseline_v1 as an input and a linear shortcut) in the same four pipelines, with augmentation. This is goal 3 of the project brief:
  'does a learner that can actually perceive the puzzles benefit from human learning histories?'. Reference: baseline_v1 (linear, 12 features) solves 65.8-67.1%, 34% right first; the hybrid learner of harc_h1_stml_v2 solved 55.5% (humans),
  61.3% (answers), 64.0% (RL from scratch), 61.8% (humans then RL) without augmentation and without matched imitation lengths. PREDICTIONS, written before running:
  H-a  hybrid humans-then-RL with augmentation reaches >= 68% solved (above the linear baseline).
  H-b  hybrid humans-then-RL beats hybrid answers-then-RL by >= 2 points (H1 holds for a learner that perceives).
  H-c  hybrid RL from scratch (augmented) reaches >= 62%.
Q33 (--seed0 9 --parts, with augmentation, fresh splits 9-11): which part of the human histories carries the effect? Conditions (all: imitation phase of the same length, then the same RL): S_aug (full histories), SA_aug (answers),
  S_first (only first attempts of the human histories, resampled to the same number of steps), S_rev (only revision steps, i.e. attempts after a failure, resampled), S_shuf (full steps but the attempt histories permuted among the steps of the same puzzle, so the
  history no longer matches the choice). PREDICTIONS, written before running (Q31: S_aug minus SA_aug +5.3):
  G1  S_rev beats S_first by >= 1.5 points (the revisions carry the learning-from-feedback signal).
  G2  S_first still beats SA_aug by >= 1 (which wrong answers people give is information).
  G3  S_shuf is >= 2 below S_aug (the match between history and choice matters).
Q32 (--k-aug 0 --rl-history --seeds 5): the same pipelines WITHOUT augmentation (splits 0-4). PREDICTIONS, written before running (Q31 with augmentation: humans-then-RL minus answers-then-RL +5.3, minus RL-history +4.6):
  F1  without augmentation humans-then-RL still beats RL-history-then-RL and answers-then-RL, by less than with augmentation (<= +2.5 over answers).
  F2  every pipeline scores below its augmented version.
Q31 (--seed0 6 --rl-history): fresh splits 6-8. SRL_aug = imitate the RL agent's OWN learning history (the choices it made while learning from scratch on the same augmented puzzles; a random subset of the same number of steps as the
  human imitation; Algorithm Distillation, the original H1 control), then the same RL. PREDICTIONS, written before running:
  E1  S_aug (humans-then-RL) beats SRL_aug by >= 2 points.
  E2  S_aug beats SA_aug by >= 1 on the fresh splits (third replication; pooled with Q29/Q30 gives 9 splits).
  Falsifier of E1: S_aug - SRL_aug <= +1.
  D2  SF_aug lies between SA_aug and S_aug (part of the gain is simply training the history channel on failed attempts; the rest, if any, is human-specific).

    python experiments/harc_augmentation_pipeline.py --quick
    python experiments/harc_augmentation_pipeline.py
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from harc_augmentation import K_AUG, augmented  # noqa: E402
from stml.model.graphs import ARC_EDGE_TYPES, ARC_NODE_DIM  # noqa: E402
from stml.model.hand_features import N_HAND, attach_hand, hand_for_grids  # noqa: E402
from stml.model.arctraj_data import arctraj_steps  # noqa: E402
from stml.model.harc_data import Step, answer_steps, attach_pools, build_graphs, human_steps, load_puzzles, read_records  # noqa: E402
from stml.model.network import RelationalGNN, STMLLearner, pick_device  # noqa: E402
from stml.model.pretrain import pretrain_gnn, pretrain_task  # noqa: E402
from stml.model.train import evaluate, fit_imitation, train_rl  # noqa: E402
from stml.runlog import Run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--rl-episodes", type=int, default=6000)
    ap.add_argument("--rl-restarts", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pretrain", action="store_true", help="Q36: self-supervised masked-node pretraining of the object-graph encoder on the grids of the split's training puzzles (and their augmented copies)")
    ap.add_argument("--pretrain-task", action="store_true", help="Q37: puzzle-level pretext (leave-one-pair-out prediction of the held-out output's shape / palette / density / symmetry) on the training puzzles' demonstration pairs")
    ap.add_argument("--arctraj", action="store_true", help="Q40: add the ARCTraj first-attempt steps of the training puzzles to the human steps (more human data)")
    ap.add_argument("--human-frac", type=float, default=1.0, help="Q38: fraction of the human steps (and, equal length, answer steps) used in the imitation phase")
    ap.add_argument("--family", choices=["pure", "hybrid"], default="pure", help="Q34: hybrid = + the 12 hand features of baseline_v1 as input and linear shortcut")
    ap.add_argument("--parts", action="store_true", help="Q33: which part of the human histories carries the effect (S_first, S_rev, S_shuf)")
    ap.add_argument("--k-aug", type=int, default=3, help="Q32: augmented copies per training puzzle (0 = none)")
    ap.add_argument("--seed0", type=int, default=0, help="first split seed (Q30 uses 3 = fresh splits)")
    ap.add_argument("--rl-history", action="store_true", help="Q31: add SRL_aug, imitation of the RL agent's own learning history (Algorithm Distillation control), then RL")
    ap.add_argument("--synthetic-failures", action="store_true", help="Q30: add SF_aug, answers imitation with synthetic failed attempts")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.rl_episodes, a.rl_restarts, a.d = 1, 2, 160, 1, 64
    global K_AUG
    K_AUG = a.k_aug
    device = pick_device(None)
    conds = ["S_aug", "SA_aug"] + (["SF_aug"] if a.synthetic_failures else []) + (["SRL_aug"] if a.rl_history else []) + (["S_first", "S_rev", "S_shuf"] if a.parts else []) + ([] if a.parts else ["R_aug"])
    with Run("harc_augmentation_pipeline" + ("_hybrid" if a.family == "hybrid" else "") + ("_pretrain" if a.pretrain else "") + ("_pretask" if a.pretrain_task else "") + (f"_frac{a.human_frac}" if a.human_frac < 1.0 else "") + ("_arctraj" if a.arctraj else "") + ("_quick" if a.quick else ""), {**vars(a), "conditions": conds, "k_aug": K_AUG}) as run:
        records = read_records(ROOT / "data/processed/harc.jsonl")
        arc_records = read_records(ROOT / "data/processed/arctraj.jsonl") if a.arctraj else None
        puzzles = attach_pools(load_puzzles(ROOT / "data/raw/harc/arc_data"), records)
        if a.quick:
            puzzles = {k: puzzles[k] for k in sorted(puzzles)[:40]}
        build_graphs(puzzles, cache=None if a.quick else ROOT / "data/processed/harc_graphs.pkl", log=run.log)
        hybrid = a.family == "hybrid"
        if hybrid:
            attach_hand(puzzles, ROOT / "data/raw/harc/arc_data", records)
        raw_hand = {n: p_.hand for n, p_ in puzzles.items()} if hybrid else None
        names = sorted(puzzles)
        cache: dict[tuple[str, int], object] = {}
        per_split = {c: [] for c in conds}

        pre_state: dict = {}

        def new_model(seed: int) -> STMLLearner:
            torch.manual_seed(seed)
            m_ = STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=a.d, out_match=True, hand_dim=N_HAND if hybrid else 0, wide=hybrid).to(device)
            if pre_state.get('state'):
                m_.gnn.load_state_dict(pre_state['state'])
            if pre_state.get('task'):
                for k_ in ('gnn', 'pair', 'task_in'):
                    getattr(m_, k_).load_state_dict(pre_state['task'][k_])
            return m_

        def best_rl(allp, pool, train, seed, tag, with_history=False):
            best = None
            for k in range(a.rl_restarts):
                m = new_model(1000 * seed + k)
                hist = train_rl(m, allp, pool, device, episodes=a.rl_episodes, seed=1000 * seed + k, log=lambda s: None)
                score = evaluate(m, allp, train, device)["solved"]          # restart chosen on the ORIGINAL training puzzles
                run.log(f"{tag} restart {k + 1}: training-puzzle solve rate {score:.1%}")
                if best is None or score > best[0]:
                    best = (score, m, hist)
            return best[1] if not with_history else (best[1], best[2])

        for seed in range(a.seed0, a.seed0 + a.seeds):
            rnd = random.Random(seed)
            order = names[:]
            rnd.shuffle(order)
            train, test = order[: len(order) // 2], order[len(order) // 2:]
            for n in train:
                for i in range(K_AUG):
                    if (n, i) not in cache:
                        cache[(n, i)] = augmented(puzzles[n], i, random.Random(zlib.crc32(f"{n}:{i}".encode())))
            aug_names = [f"{n}#a{i}" for n in train for i in range(K_AUG)]
            if hybrid:                 # standardise with the TRAINING puzzles' candidates only; features with no spread keep unit scale
                X = np.concatenate([raw_hand[n] for n in train])
                mu, sd = X.mean(0), X.std(0)
                sd = np.where(sd < 1e-6, 1.0, sd)
                for n in names:
                    puzzles[n].hand = ((raw_hand[n] - mu) / sd).astype(np.float32)
                for (n, i), q in cache.items():
                    if n in set(train):
                        if getattr(q, "raw_hand", None) is None:
                            q.raw_hand = hand_for_grids(q.train, q.test_input, q.target, q.candidates)
                        q.hand = ((q.raw_hand - mu) / sd).astype(np.float32)
            allp = dict(puzzles)
            allp.update({q.name: q for (n, i), q in cache.items() if n in set(train)})
            if a.pretrain:
                gs = []
                for n_ in list(train) + aug_names:
                    q_ = allp[n_]
                    gs += [g for pair in q_.graph_train for g in pair] + [q_.graph_test_in] + list(q_.graph_cands)
                torch.manual_seed(seed)
                g0 = RelationalGNN(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=a.d, layers=3)
                pre_state['state'] = pretrain_gnn(g0, gs, device, epochs=1 if a.quick else 6, seed=seed, log=run.log)
                run.log(f"pretrained the encoder on {len(gs)} grids of the training puzzles")
            if a.pretrain_task:
                torch.manual_seed(seed)
                m0 = STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=a.d, out_match=True).to(device)
                pre_state['task'] = pretrain_task(m0, [allp[n_] for n_ in list(train) + aug_names], device, epochs=1 if a.quick else 6, seed=seed, log=run.log)
                del m0
            h_train = human_steps(records, puzzles, set(train))
            if a.arctraj:
                extra = arctraj_steps(arc_records, puzzles, set(train))
                run.log(f"ARCTraj adds {len(extra)} steps to the {len(h_train)} H-ARC steps")
                h_train = h_train + extra
            if a.human_frac < 1.0:
                h_train = random.Random(seed).sample(h_train, int(len(h_train) * a.human_frac))
            h_aug = h_train + [Step(f"{s.puzzle}#a{i}", s.tried, s.choice, s.correct) for s in h_train for i in range(K_AUG)]
            h_test = human_steps(records, puzzles, set(test))
            run.log(f"--- split {seed + 1}/{a.seeds}: {len(train)} training, {len(test)} test puzzles")

            def record(c, m):
                r = evaluate(m, allp, test, device, human=h_test)
                per_split[c].append(r)
                run.log(f"{c}: solved {r['solved']:.1%}, first try {r['first']:.1%}")

            m_s = fit_imitation(new_model(seed), allp, h_aug, device, epochs=a.epochs, seed=seed, log=lambda s: None)
            train_rl(m_s, allp, train + aug_names, device, episodes=a.rl_episodes, seed=seed, log=lambda s: None)
            record("S_aug", m_s)
            m_a = fit_imitation(new_model(seed), allp, answer_steps(allp, set(train) | set(aug_names), len(h_aug)), device, epochs=a.epochs, seed=seed, log=lambda s: None)
            train_rl(m_a, allp, train + aug_names, device, episodes=a.rl_episodes, seed=seed, log=lambda s: None)
            record("SA_aug", m_a)
            if a.synthetic_failures:
                srnd = random.Random(seed)
                pool = sorted(n for n in set(train) | set(aug_names) if len(allp[n].candidates) > 1)      # a wrong candidate must exist
                sf = []
                while len(sf) < len(h_aug):
                    n = pool[len(sf) // 2 % len(pool)]
                    t = allp[n].target_index
                    wrong = srnd.choice([i for i in range(len(allp[n].candidates)) if i != t])
                    sf += [Step(n, [], wrong, False), Step(n, [(wrong, False)], t, True)]
                m_f = fit_imitation(new_model(seed), allp, sf, device, epochs=a.epochs, seed=seed, log=lambda s: None)
                train_rl(m_f, allp, train + aug_names, device, episodes=a.rl_episodes, seed=seed, log=lambda s: None)
                record("SF_aug", m_f)
            if a.parts:
                prnd = random.Random(seed)
                resample = lambda steps: [steps[prnd.randrange(len(steps))] for _ in range(len(h_aug))]
                first = [x for x in h_aug if not x.tried]
                rev = [x for x in h_aug if x.tried]
                by_p: dict[str, list] = {}
                for x in h_aug:
                    by_p.setdefault(x.puzzle, []).append(x)
                shuf = []
                for lst in by_p.values():          # permute the attempt histories among the steps of the same puzzle: the history no longer matches the choice
                    tr_ = [x.tried for x in lst]
                    prnd.shuffle(tr_)
                    shuf += [Step(x.puzzle, t_, x.choice, x.correct) for x, t_ in zip(lst, tr_)]
                for tag, steps in (("S_first", resample(first)), ("S_rev", resample(rev)), ("S_shuf", shuf)):
                    m_p = fit_imitation(new_model(seed), allp, steps, device, epochs=a.epochs, seed=seed, log=lambda s: None)
                    train_rl(m_p, allp, train + aug_names, device, episodes=a.rl_episodes, seed=seed, log=lambda s: None)
                    record(tag, m_p)
            elif a.rl_history:
                m_r, hist = best_rl(allp, train + aug_names, train, seed, "R_aug", with_history=True)
                record("R_aug", m_r)
                take = np.random.default_rng(seed).choice(len(hist), size=min(len(h_aug), len(hist)), replace=False)
                m_ad = fit_imitation(new_model(seed), allp, [hist[i] for i in take], device, epochs=a.epochs, seed=seed, log=lambda s: None)
                train_rl(m_ad, allp, train + aug_names, device, episodes=a.rl_episodes, seed=seed, log=lambda s: None)
                record("SRL_aug", m_ad)
            else:
                record("R_aug", best_rl(allp, train + aug_names, train, seed, "R_aug"))
            run.log(f"steps: first {len([x for x in h_aug if not x.tried])}, revisions {len([x for x in h_aug if x.tried])}") if a.parts else None
            (run.dir / "partial_per_split.json").write_text(json.dumps({c: [{k: v for k, v in r.items() if not k.startswith('_')} for r in rs] for c, rs in per_split.items()}))
        sv = {c: np.array([r["solved"] for r in per_split[c]]) for c in conds}
        fs = {c: np.array([r["first"] for r in per_split[c]]) for c in conds}
        L = ["# Humans-then-RL vs answers-then-RL under augmentation (pure STML learner)", "", f"{a.seeds} splits; solved in 3 attempts on unseen puzzles (mean, sd over splits). Predictions C1-C2 in the script.", "",
             "| condition | solved in 3 | right first |", "| --- | ---: | ---: |"]
        for c in conds:
            L.append(f"| {c} | {100 * sv[c].mean():.1f}% ({100 * sv[c].std():.1f}) | {100 * fs[c].mean():.1f}% ({100 * fs[c].std():.1f}) |")
        L += ["", "| comparison | solved difference (points; sd) |", "| --- | ---: |"]
        for x, y in [p_ for p_ in (("S_aug", "SA_aug"), ("S_aug", "R_aug"), ("SA_aug", "R_aug"), ("SF_aug", "SA_aug"), ("S_aug", "SF_aug"), ("S_aug", "SRL_aug"), ("SRL_aug", "SA_aug"), ("SRL_aug", "R_aug"), ("S_rev", "S_first"), ("S_first", "SA_aug"), ("S_rev", "SA_aug"), ("S_shuf", "S_aug"), ("S_shuf", "SA_aug")) if p_[0] in conds and p_[1] in conds]:
            d = 100 * (sv[x] - sv[y])
            L.append(f"| {x} minus {y} | {d.mean():+.1f} ({d.std():.1f}) |")
        L += ["", "Reference (Q27/Q28, same splits): imitate human learning with augmentation 57.3%, imitate answers with augmentation 59.3%, humans-then-RL 60.4%, RL from scratch (augmented) 51.9%."]
        run.finish({c: [{k: v for k, v in r.items() if not k.startswith('_')} for r in rs] for c, rs in per_split.items()}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

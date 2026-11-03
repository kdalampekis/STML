# STML: Socratic Topological Meta-Learning

This file is the project brief for Claude Code. Read it fully before doing anything in this
folder. It explains the idea, the hypotheses, the architecture and its math, the code, the
game, the datasets, the results so far (including what went wrong), the current goal, and
the rules for working here.

Owner: Kostas Bekis. Folder: `~/Research_Projects/STML` on a MacBook (Apple M5, macOS).
Everything runs **on this Mac**, using the Apple GPU through PyTorch's Metal backend (`mps`).
Logs, datasets and results stay in this folder. No git for now.

---

## 1. The idea in one paragraph

Standard imitation learning copies **what** people answer. STML copies **how** people learn:
the wrong guesses, the moment a contradiction is noticed, and the revision that follows. The
learner reasons in an explicit graph, not in text. A Socratic critic tells it *why* something
failed, not just *that* it failed. A SAT solver acts as an independent referee that certifies
every verdict and is never visible to the learner. The learner is first trained to imitate
human learning trajectories (Stage A), then pushed past human efficiency with reinforcement
learning (Stage B).

The original chain of reasoning (the owner's): understand how people learn → translate that
process into something a machine can learn (imitation + RL) → a machine that learns like
people can understand context (transfer) → if it understands, it can produce new knowledge →
if it produces knowledge, it can form its own concepts and, eventually, its own language. A
fourth idea: a learner that **knows what it does not know**, saying "I don't know yet"
instead of inventing an answer, while still trying to build the missing knowledge.

## 2. Hypotheses

- **H1, process imitation.** A learner trained on human learning trajectories (mistakes and
  corrections included) learns new domains faster and transfers better than the same learner
  trained on RL-agent learning histories (Algorithm Distillation) or on scalar reward.
- **H2, diagnostic feedback.** Structured critiques (which rule failed, which premises are
  responsible, a countermodel) teach faster than a right/wrong signal, for the same learner.
- **H3, knowing what it does not know.** With knowledge stored as an explicit verified graph
  and a reward that makes honesty pay, the learner says "I don't know yet" when it has neither
  a proof nor a counterexample, and stays calibrated on problems harder than its training.

Kill criteria: if human trajectories give no advantage over RL histories, report it and keep
H2/H3; if critiques never beat scalar reward; if abstention does not track real difficulty.

## 3. Related work (closest)

- Algorithm Distillation (Laskin et al., ICLR 2023): imitate RL agents' learning histories →
  in-context RL. STML imitates **human** histories instead.
- MLC (Lake & Baroni, Nature 2023): human-like generalization by meta-learning, without
  imitating human behavioural data.
- Critique-GRPO, Text2Grad (2025): RL from text critiques for LLMs. STML: structured critiques,
  non-linguistic graph learner.
- Abductive Learning (Dai et al., NeurIPS 2019); AlphaGeometry (Nature 2024): neural proposer +
  symbolic checker.
- Kadavath et al. 2022 and R-Tuning (NAACL 2024): LLMs saying "I don't know". STML defines
  "don't know" as the absence of a certificate.
- Name clash to avoid: "RLEF" is Meta's RL from Execution Feedback (ICML 2025).

Neuroscience / cognitive-science background (62 papers, 27 PDFs, a résumé of what each finding means for STML): `literature/`
(start with `literature/SUMMARY.md`; the architecture and pretraining design derived from it is `docs/ARCHITECTURE_V2.md`; `literature/INDEX.md` lists every paper; `python scripts/fetch_literature.py` rebuilds it).

Full proposal (living document): https://claude.ai/code/artifact/c35b18e7-137c-4dfd-87a6-014cc1326c5d

## 4. Architecture and math

### 4.1 The logic domain (the main sandbox)

Categorical logic (syllogisms) over invented kinds ("wugs", "daxes"), i.e. basic set theory:

| Statement | Set reading |
| --- | --- |
| All A are B | A ⊆ B |
| No A are B | A ∩ B = ∅ |
| Some A are B | A ∩ B ≠ ∅ |
| Some A are not B | A \ B ≠ ∅ |

Every kind is non-empty. A world is a set of inhabited "regions" (exact subsets of kinds).
Universal statements force regions empty; existential ones need an inhabited region. So
satisfiability is exact, entailment is `premises ∧ ¬claim` unsatisfiable, and every claim is
**follows / contradicted / open**. Small-model property: a countermodel needs at most one
individual per existential statement.

Proof rules (Barbara, Celarent, Darii, Ferio, Baroco, Bocardo, plus conversion, identity,
existence) are **tested sound and complete** against the semantics, so every true claim has
a named proof and every open claim has a concrete countermodel.

### 4.2 The SAT referee

Variables x[j][i] = "individual j is a K_i"; one witness individual per existential statement;
All/No become clauses over all individuals; Some/Some-not fix the witness. Polynomial size;
PySAT + CaDiCaL. UNSAT → unsat core (which premises a verdict needs); SAT → countermodel.
Agreed with the semantics on 13,252 of 13,252 checks; solves 40-kind worlds in milliseconds.
**The learner never calls the referee** (otherwise "I know" is trivial and H3 is void).

### 4.3 The STML learner (`stml/model/`)

```
state graph(s) ──► relational GNN ──► graph embeddings
                                         │
task token (ARC: mean over example pairs [in, out, out−in]; + test input)
candidate tokens (each possible answer, read against task and test input)
                                         │
history transformer (causal): [task, attempt 1 + feedback, attempt 2 + feedback, …]
                                         │
     choice head: score per candidate (what to try next)
     confidence head: P(this answer is right) (what it knows, for H3)
```

- **Graphs** (`graphs.py`). ARC grid → object graph: nodes = 4-connected one-colour objects
  (colour, size, bounding box, centroid, background flag); edge types touch, same_color,
  same_shape, aligned_row, aligned_col. Logic world → concept graph: nodes = kinds (no names,
  so renaming changes nothing); 16 edge types = {All, No, Some, Some-not} × {given, proven,
  refuted, open}.
- **Relational GNN** (`network.py`): h_v ← h_v + ReLU(LN(W₀h_v + Σ_r mean_{u∈N_r(v)} W_r h_u)),
  3 layers, mean+max pooling. Pure PyTorch (no graph library) so it runs on Metal.
- **History transformer**: 2 layers, 4 heads, causal mask; tokens = candidate embedding +
  feedback embedding + attempt position. This is where learning from one's own mistakes
  happens in context.
- **Stage A** (`train.py: fit_imitation`): cross-entropy on the next choice in a learning
  history + BCE for the confidence head on whether that choice was right.
- **Stage B** (`train.py: train_rl`): REINFORCE, reward 1 − 0.25·(attempt−1) if solved, moving
  baseline, entropy bonus; confidence head trained on its own outcomes. Every choice is
  recorded so its learning history can be imitated (the Algorithm Distillation control).
- **Device**: `pick_device()` → `mps` (Apple Metal) if available, else `cuda`, else `cpu`.

### 4.4 Human learning model (Pillar 1, for the cognitive learner, not built yet)

Surprise δ_t = o_t − ô_t(G_t); accumulated S_t. Small surprise → assimilation (adjust beliefs);
S_t > τ → accommodation: G_{t+1} = argmin edit(G_t, G') + λ·violations + μ·MDL(G'). Socratic
question q* = argmax information gain about which assumption is wrong. Reported confidence
κ ∈ {1..5} is a noisy readout of belief.

### 4.5 H3 answer rule

Four answers: follows / contradicted / open / "I don't know yet". Reward +1 correct, −λ wrong,
−k for "don't know" (0 < k < λ). Commit only if p > (λ − k)/(1 + λ); e.g. k = 0.2: λ = 1 → 0.40,
λ = 4 → 0.76. Train at several λ and report the risk–coverage curve. A correct answer without a
certificate counts as a lucky guess, not knowledge.

## 5. The game: Wugs and Daxes (human data)

`game/template.html`, built by `scripts/build_game.py` into `game/index.html`. 5 levels × 3
worlds (Chains, Exclusion, Some, Exceptions, Transfer). Players propose statements that must
follow, rate confidence 1–5 before each answer, can mark the premises they used and say why.
The critic answers with a status, a Socratic question, and on request a proof (named rules) or
a countermodel world with the individual that breaks the claim. After each world: a test round
of 4 statements answered with **must be true / must be false / can't tell / I don't know yet**
(H3 data), then a one-sentence reflection.

- Published version (the owner is playing it now): https://claude.ai/artifact/74tWZmQeXUT5P43UYdjt7h
  Logs from that page go to the page's private research store (only the owner can read all of
  them). Claude on claude.ai can export them; Claude Code cannot reach that store. The page's
  **Download log** button gives the same log as a file: put it in `logs/<participant>/`.
- Local version: `python scripts/play_local.py` → http://127.0.0.1:8000; every move is saved
  automatically to `logs/<participant code>/<session>_<world>.json`.
- Log schema `stml-log/1` (see `stml/trajectories.py`). Analyse with
  `python scripts/analyze_log.py logs/<participant>`.
- The owner's own run is a usability pilot, **not data** (he knows the rules and hypotheses).

## 6. Datasets

All sources are converted to one format, **`stml-traj/1`** (`stml/datasets/common.py`): one
record per learner × task, with attempts (actions, commit, feedback, explanation, analysis),
timed explanations, epistemic judgments and outcome. The premise (itself testable): the human
learning loop (hypothesis → act → commit → feedback → revise → explain) has the same shape in
every domain.

### H-ARC (in `data/raw/harc/`, converted by `stml/datasets/harc.py`)

LeGris, Vong, Lake & Gureckis, *Scientific Data* 2025. Data from OSF (https://osf.io/bh8yq);
task files from https://github.com/Le-Gris/h-arc (`arc_data/`). Facts checked on the real data:

- 586,266 actions, 1,729 participants (1,393 finished), 800 tasks; up to 3 attempts with
  right/wrong feedback; an explicit `submit` action ends each attempt.
- **Every action row carries the full output grid**, so distance to the answer is known at
  every step (`d` on each action in our format).
- The **first rule description is written during attempt 1, before any feedback**; after the
  final submit people either write a revised description (2,771) or keep the first (6,061).
  Stored as timed `explanations` (initial / revised), with a quality flag.
- Validation: per-task solve rate 76.2% on training tasks, exactly the published figure
  (evaluation tasks 65.3% vs 64.2% published; we keep only finishers).
- Findings: first-attempt accuracy rises 50% → 58% from a person's 1st to 5th puzzle (random
  task order: **learning to learn**); after a wrong answer 52% of revisions get closer and 30%
  further; 48% of wrong submissions came after the person had been closer earlier in the same
  attempt (hypothesis switching, not hill-climbing); after a mistake people keep the grid size
  86% of the time and keep a median 93% of cells (local repair).

Convert: `python scripts/convert_datasets.py harc --data data/raw/harc/data --arc data/raw/harc/arc_data`
→ `data/processed/harc.jsonl` (≈48 MB, 7,871 records).

### ARCTraj (in `data/raw/arctraj/ARCTraj.csv`, converted by `stml/datasets/arctraj.py`)

Kim et al., Hugging Face `SejinKimm/ARCTraj`, CC-BY-NC. `python scripts/download_arctraj.py` fetches
the CSV directly (`load_dataset()` returns only the Croissant metadata card, not the data). Real layout,
checked: 10,670 sessions, 101 users, 401 task ids; each row = one session with a JSON list of actions
(SelectCell, Paint, Paste, Move, Undo, Redo, Submit, ...), each carrying the full grid after the action.
**Every session has exactly one Submit and the `succeed` flag is False on all of them**, so correctness is
recomputed against the first test output of the ARC training task (73% solved). 476 sessions have an empty
task id and 1 is unparseable; 10,193 records are converted to `data/processed/arctraj.jsonl`
(`python scripts/convert_datasets.py arctraj --arc data/raw/harc/arc_data`).
**There is no wrong-answer -> feedback -> revision loop in it** (one attempt per session), so it can add
first-attempt choices and intermediate-grid progress labels, not learning-from-feedback histories.

### Planned

Sudoku-Bench CTC (2,565 expert solves with spoken reasoning + moves); CMU DataShop tutor logs
(step-level, with errors and hints, including logic tutors).

## 7. Experiments so far

### 7.1 First experiment: `experiments/baseline_v1/` (linear choice model, hand features)

Protocol (kept for all H-ARC experiments): unseen puzzles, 50/50 puzzle split per seed, 5
splits; for each puzzle the learner chooses among the answers people actually gave (median 8;
the correct one is present in 787/800 puzzles); 3 attempts with right/wrong feedback. Same model
and same number of training steps in every condition; only the data source differs.

Results (from a run in Claude's cloud workspace, after the bug fix below; **re-run locally to
get logs**):

| Condition | Solved in 3 | Right first | Human-likeness | (revisions) |
| --- | ---: | ---: | ---: | ---: |
| Imitate human learning | 65.8% | 34.0% | **−2.00** | **−1.90** |
| Imitate answers | 66.5% | 34.2% | −2.77 | −3.16 |
| Imitate RL learning history | 66.8% | 34.4% | −2.79 | −3.12 |
| RL agent itself (best of 5 restarts) | 66.6% | 34.1% | −3.96 | −4.58 |
| Random choice | 45.8% | 18.9% | −2.23 | −2.36 |
| People | 71.4% | 54.9% | – | – |

Human-likeness = mean log-likelihood of people's actual next choices on unseen puzzles
(higher = more human-like). Differences in solving are all within ±1.2 points; every 95% CI
includes 0. **H1 is not supported in this setup.**

What we learned:
- The human-trained model is the **only** one that predicts how people revise better than
  chance. Human revision behaviour is systematic and learnable (it prefers answers similar to
  the one that just failed: local repair).
- The bottleneck is representation: every model gets ~34% right first time vs 55% for people,
  because it sees puzzles only through 12 hand-made features. A learner that cannot perceive
  the rule cannot benefit from a better learning strategy, whoever it imitates.
- Performance collapses on puzzles with many candidate answers (16+ answers: 34% solved), which
  is exactly where perception matters.

What went wrong (keep these in mind; each was caught and fixed):
1. **Unfair baseline.** The first run showed human histories +4.3 points over RL histories with
   a CI excluding 0. It came entirely from 2 of 5 splits where REINFORCE got stuck (~50%).
   Fix: RL restarts, best chosen on training puzzles only. Then the gap vanished.
2. **Wrong metric.** "Mean probability of the human choice" rewarded an overconfident model.
   Fix: log-likelihood (a proper score).
3. **Scaling bug.** The RL agent standardized features using fresh-start states only, where the
   history features are always 0 → divided by ~0 → its weight on "similarity to the failed
   answer" was 8,425. Fix: features with no spread keep unit scale (`_scale`).
4. **Design limits that remain:** the candidate pool comes from people (it leaks human
   knowledge into every condition and makes recovery easy as the pool shrinks); the model was
   memoryless across tasks, so the learning-to-learn effect (50% → 58%) could not be used.

### 7.2 Current experiment: `experiments/harc_h1_stml.py` (our architecture)

Same protocol, but the learner is the STML learner: it reads the real grids through the
object-graph GNN and conditions on its attempt history with the transformer. Conditions:
imitate human learning; imitate answers; RL from scratch (best of restarts, chosen on training
puzzles); imitate the RL agent's learning history (same step budget); **STML pipeline: imitate
humans, then RL**. Reports solve rate, first-try, recovery, human-likeness, and confidence
calibration (Brier, ECE).

**Result** (run `2026-10-13_184916_harc_h1_stml`, 5 splits, 105 min): pure GNN learner, 845k parameters.
Solved in 3 / right first: imitate human 50.9% / 21.5%; imitate answers 52.4% / 22.8%; imitate RL history
50.9% / 23.0%; RL from scratch 54.0% / 25.3%; STML (humans then RL) 53.1% / 22.5%; random 45.8% / 18.9%;
people 71.4% / 54.9%. Worse than the 12-hand-feature baseline (66%, 34%). Imitating human learning is behind
RL from scratch by 3.1 points (CI excludes 0). Overfits: RL restarts reach 62-68% on training puzzles, 52-56%
on unseen. Confidence is poorly calibrated for imitate-answers (ECE 0.78) and the STML pipeline (0.43).
**H1 is not supported.**

### 7.3 Round 2 on ARC: ARCTraj and hand features (`experiments/harc_h1_stml_v2.py`, `dev_stml_v2.py`)

Dev ablation on validation puzzles carved out of training puzzles (dev seeds >= 100; test halves of seeds
0-4 never touched): ARCTraj steps, ARCTraj progress regression, smaller model: all within noise (about +-5
points on 79 puzzles). Giving the network the 12 hand features of baseline_v1 (the "hybrid" family: as input
and as a linear shortcut to the choice score) helps; output-match input (candidate minus mean training-output
embedding) is a small plus. Final runs `2026-10-14_125206_harc_h1_stml_v2_pure` / `..._hybrid` (5 splits, 7
conditions each, 2 RL restarts). Solved in 3 (pure / hybrid): imitate human 49.8 / 55.5; + ARCTraj 51.6 / 57.6;
imitate answers 54.1 / 61.3; imitate RL history 51.9 / 61.3; RL from scratch 52.8 / **64.0**; STML 55.4 / 61.8;
STML + ARCTraj 54.2 / 62.3. Findings: (1) the graph encoder alone barely perceives the puzzles; with hand features
it gets close to but not above the linear baseline (it adds nothing measurable on top of the 12 features);
(2) **H1 reversed**: imitating human learning is the worst learning condition in both families (CIs exclude 0);
(3) ARCTraj gives +1.8 / +2.0 points, CIs include 0; (4) human-trained models still predict how people revise
(hybrid revisions log-lik -2.09 vs -2.36 random); (5) calibration: RL from scratch best (ECE 0.06 pure), imitate
answers terrible (0.72-0.77). The first launch of these runs was killed mid-split-4 when a session ended
(`2026-10-14_014325_*`, kept, not used); runs started afterwards are detached and write
`partial_per_split.json` after every split.

### 7.4 Logic domain, H3: graph judge (`experiments/logic_h3_judge.py`, `logic_h3_scaling.py`, `logic_h3_certificate.py`, `logic_h3_thresholds.py`)

Judge = relational GNN over the premise graph of a world (nameless kinds; 8 edge types = 4 forms x
forward/backward), answers follows / contradicted / open, plus a "why" head naming the premises of its proof.
Labels from the exact semantics; large worlds (up to 32 kinds) are labelled by the SAT referee (generator
checked against the semantics on 930/930 claims). The judge never calls the referee.
- `logic_h3_judge` (`2026-10-14_161820_*`): trained on levels 1-4, tested on unseen worlds: 100% (levels 1-4),
  98.5% (level 5, never trained). Too easy to test H3.
- `logic_h3_scaling` (`..._162714_*`, size-free rerun `..._163259_*`): small-world judge on 6-24 kinds. Accuracy
  falls (24 kinds: 88% on trees, 65% on long chains) while confidence stays 91-97%. Part of the first drop was
  an artifact of size-dependent node features (n/12, degree/n): removed with `SIZE_FREE`. Real failure: proof
  depth (1-2 premises 99-100% right, 9+ premises 76% right, 92% confident).
- `logic_h3_certificate` (`..._175255_*`): two training regimes (small; extended with 8-12 kinds), tested on
  12-32 kinds. Extended training lifts 24-kind chains 66% -> 91% and 32-kind 59% -> 82%. Gates compared: confidence,
  5-model ensemble, predicted proof length, and the learner's own **certificate checks** (sufficiency: re-judge
  with only the premises it named; necessity: remove each named premise). **Sufficiency** beats confidence on AURC
  and AUROC in 5/5 seeds wherever the judge makes real errors (e.g. small-trained 32-kind chains AURC 0.285 -> 0.203,
  AUROC 0.66 -> 0.76) and beats the ensemble on chains with one model; it is no better where errors are almost
  absent. **Necessity and proof length do not work** (AUROC < 0.5): the named premise sets are not minimal.
- `logic_h3_thresholds` (`..._195306_*`): a threshold fixed on in-distribution validation worlds barely abstains
  on big worlds and scores below always-commit at lam=4. With some harder validation worlds or few-shot
  recalibration (25-200 answered problems in the new setting) the honest score recovers most of the oracle
  ceiling, e.g. 32-kind chains lam=4, extended training, sufficiency gate: -0.14 -> +0.44 (25 problems) vs oracle
  +0.49 and always-commit +0.14. Caveat: the few-shot simulation assumes right/wrong feedback on the m problems
  whether or not the learner would commit; a real learner only sees feedback on what it commits to.
**H3 is partly supported**: the learner's own sufficiency self-check detects its errors better than softmax
confidence or an ensemble, and shift-aware thresholds turn that into better decisions; turning it into a
guarantee is unsolved.

### 7.5 Logic domain, H2: feedback types (`experiments/logic_h2_feedback.py`, `2026-10-14_164241_*`)

Same judge, one pass over N problems, N = 500...20000, four feedback types: scalar (REINFORCE), scalar (binary
cross-entropy on the chosen answer), true verdict, verdict + which premises are responsible (referee core).
The true verdict beats a right/wrong bit by about 8 points at N=500 (levels 1-4: 67% vs 59%), gap closing by
N~10000; REINFORCE is far behind. The "why" adds +0.6 (levels 1-4) and +1.4 (level 5) points on average (positive
in 5/5 seeds), nothing on 12-kind chains (-0.1). H2 is weakly supported on information content, unproven on the
structured part.

### 7.6 Learning to learn across puzzles (`experiments/harc_learning_to_learn.py`, `2026-10-14_163948_*`)

Hybrid family; the transformer sees the earlier puzzles of a sequence (attempts + feedback); trained by
meta-RL on sequences of 5 and by imitating people's own sequences; tested on random sequences of 5 unseen puzzles
with the learner's own context. **Null**: first-try accuracy is flat by position (slopes +0.1 to +0.25 points
per position, sd 0.4-0.8, same as the no-context controls); context does not raise overall accuracy. People rise
50.0% -> 58.0% over positions 1-5 (probably partly adaptation to the interface, which the learner does not
need). Small budget (6000 episodes), so not evidence against learning to learn in general.

### 7.7 Literature, architecture design, study sizing, process-level imitation (2026-10-15)

- `literature/` (62 papers, 27 PDFs) and `docs/ARCHITECTURE_V2.md`: eight findings from the neuroscience/cognitive science of
  learning mapped to components (surprise channel, fast episodic store + prioritized replay, structure/content split, meta-RL on
  task families, second-order metacognition, hypercorrection weighting, hypothesis sampling + local edits) and a staged
  pretraining curriculum (structure first, then interleaved surprising material). Design only; nothing there is shown to
  improve results yet.
- **Study sizing** (`stml/synthetic.py`, `experiments/sim_study_power.py`, run `2026-10-15_001937_sim_study_power`): synthetic
  participants play the real game (15 episodes, 60 test statements). ALL EFFECT SIZES ARE ASSUMED (no human data yet):
  a diagnostic-feedback effect of about +8 points on the final test needs about 40-50 completers per arm for 80% power
  (50-63 enrolled per arm at 80.6% completion), +4 points needs about 200 per arm (249 enrolled); with no effect the false-positive
  rate is 5%. A 30-50 participant pilot is a calibration/feasibility pilot, not a hypothesis test (15-25 per arm gives
  about 25-40% power for an 8-point effect). Hypercorrection (high- vs low-confidence errors) is not detectable with N up to
  300 in the current game design (power 0.22 at N=300 for a 2.6-point effect, 0.50 for 4.6 points), and a direct re-test did
  not help in this model; it is decided by design, not by N. Confidence-correctness AUROC > 0.5 is trivially detectable.
  Recalibrate `Population` with the first real logs.
- **Process-level imitation, step 1** (`experiments/harc_process_imitation.py`, run `2026-10-15_002212_harc_process_imitation`):
  next-operation prediction on unseen puzzles (gradient boosting, 14 classes). Frequency only 63.4% accuracy; + last op 79.4%;
  + feedback/attempt history 80.7%; + timing 83.0% (the largest single gain in log-loss, 0.82 -> 0.63); + coarse grid features
  83.2%. After a wrong submit people change colour 37% of the time (10% overall), reset the grid 11.5% (2.1%), copy from the
  input 8.1% (1.2%). 71% of consecutive paints are adjacent to the previous one and 88% keep the colour. So the process is
  highly structured, but mostly mechanically; next-op accuracy does not show whether it carries *solving* information. Note:
  adding the last 3 ops raised accuracy but worsened log-loss (0.725 -> 0.863), a sign of overfit probabilities in the
  boosted model. Next: predict the next cell/colour with a grid encoder, then the cross-domain transfer test (needs a
  relation-embedding encoder first, because the current GNN has per-domain edge-type weights that cannot be shared).

### 7.8 Shared encoder, transfer, edit model (2026-10-15; first pass, single seed unless stated)

- **Autonomous loop:** `docs/EXPERIMENT_QUEUE.md` holds the rules, the queue and the log for the scheduled experiment loop (cron job, session-only, 7-day expiry).
- **Machine note:** running two GPU jobs plus a browser filled the 8 GB swap on this Mac; jobs gained seconds of CPU per hour.
  Run heavy experiments one or two at a time and check `sysctl vm.swapusage`. `transfer_logic.py` now caches pretrained cores in
  `data/processed/transfer_cores/` so a restart resumes.
- **Relation-embedding encoder** (`stml/model/shared_gnn.py`, 3 modes): needed for cross-domain transfer because the per-edge-type
  GNN cannot share weights between ARC (5 types) and logic (8). `dev_shared_encoder` (`2026-10-15_135705_dev_shared_encoder`, 3 seeds):
  logic from scratch, original encoder 75.6% / 85.2% / 93.2% / 97.2% at N = 250 / 500 / 1000 / 2000; shared encoder with
  concatenated relation embedding 67.0 / 76.2 / 87.7 / 93.9; with multiplicative (FiLM) conditioning 69.1 / 79.5 / 89.5 / 95.4. Sharing
  costs accuracy: FiLM closes about a third of the gap but not all of it (12-kind chains: 56.1 vs 55.7 at N=250, 67.4 vs 69.9 at N=2000).
- **Shared encoder, round 2** (`2026-10-16_102801_dev_shared_encoder`, 3 seeds): a HYPERNETWORK core (the per-relation d x d matrices
  generated from the relation embedding by a shared network, `SharedCore(mode="hyper")`) matches the original per-relation encoder
  within noise: levels 1-4 78.5 / 85.8 / 93.0 / 96.7 vs original 75.6 / 85.2 / 93.2 / 97.0 at N = 250 / 500 / 1000 / 2000; 12-kind chains
  56.9 / 57.9 / 64.3 / 68.7 vs 55.7 / 59.8 / 65.3 / 69.9 (seed sd 1-5 points). Low-rank relation-specific terms (r = 16, 64) were WORSE than FiLM
  (65.8 / 73.8 / 85.9 / 92.5 at r = 16). So the earlier gap came from missing per-relation capacity, not from sharing; the shared
  generator keeps the core transferable. Use `--core-mode hyper` for transfer from now on.
- **Transfer ARC -> logic, first pass** (`2026-10-15_134307_transfer_logic`, concat core, 1 seed, so differences under about 4 points are
  not interpretable): fine-tuning from any ARC-pretrained core (human histories, answers, RL history, RL policy) was within about
  +-3 points of no pretraining at every N, with no consistent order between human and RL-history sources. With the core frozen, every
  pretrained core beat a random frozen core by 4-8 points on levels 1-4, again with no consistent order among sources; the logic-only
  positive control was best (88.7% vs 84.0% random at N=5000). No evidence of an advantage of human over RL histories.
- **Transfer ARC -> logic, FiLM core, 3 seeds** (`2026-10-15_142558_transfer_logic`, 58 min). Fine-tuning: no ARC source is consistently
  better than no pretraining (differences within about +-3 points, mostly inside the seed sd of 1-4); at N=250 human and RL-history
  sources are +3.1 and +4.5 points above scratch, gone by N=500. Human minus RL-history is negative at 4 of 5 N (-1.4, -2.9, -1.3,
  +0.3, -1.0; paired sd 0.4-1.7), so no human advantage, if anything the reverse. Even the logic-only positive control is about equal to
  scratch under fine-tuning, so fine-tuning mostly washes pretraining out; the ORIGINAL per-relation encoder from scratch beats every
  shared-encoder row by 4-8 points at low N (75.6% vs 67-72% at N=250), so the encoder matters more than the source. Frozen core: every
  pretrained core beats a random frozen core by 4-8 points on levels 1-4 (N=250: 51.9% random vs 56.6-60.3% ARC sources; N=5000: 83.9% vs
  86.2-87.7%) and 4-5 points on the new relations, with no consistent order among human / answers / RL sources (human best at N=250, 60.3%,
  but sd 2-4 and the gaps shrink to within 1.5 points at larger N). Reading: any trained ARC network gives better-than-random
  relational features, and the kind of learning history does not matter at this scale. Limits: 3 seeds, one ARC split, ARC learners
  that themselves only solve about 57% of puzzles.
- **Cell-level edit model** (`2026-10-15_135610_harc_edit_model`, 1 seed): the CNN predicts the next edited cell with top-1 34.5%
  (adjacent-to-last-edit heuristic 3.0%, uniform 1.1%). Adding the last submitted (wrong) grid lowered NLL after a failure by 0.114
  [0.073, 0.165], but the SHUFFLED-history control lowered it by 0.142, and the two variants also differed by 0.063 on first-attempt
  edits where their inputs are identical. So the gain is training noise, not information from the failed attempt.
  **3-seed rerun** (`2026-10-15_152519_harc_edit_model`, 107k train / 56k test edits from unseen puzzles): top-1 34.1% (no history), 34.6%
  (+ failed attempt), 34.9% (shuffled history) vs 3.0% for the adjacency heuristic. After a failure, '+ failed attempt' minus 'no history'
  is -0.078 nats [-0.106, -0.054] but the shuffled-history control gives the same -0.078 [-0.096, -0.061], and '+ failed attempt' minus
  'shuffled history' is -0.000 [-0.018, +0.018]. Both also improve first-attempt edits (-0.047, -0.082) where history cannot matter, so
  merely feeding extra history-shaped channels acts as a regulariser. **Null: a person's own failed attempt carries no detectable
  information about their next edit beyond what the CNN already gets from the current grid.** The process is highly predictable
  (top-5 66%), but not because of feedback from failure.

- **Surprise-weighted training on the logic judge** (`stml/model/surprise.py`, `experiments/logic_surprise.py`, run
  `2026-10-16_103005_logic_surprise`, 3 seeds, 7 conditions x N = 250..5000). Predictions were written in the script before running.
  P1 (surprise or hypercorrection weighting beats plain cross-entropy by >= 2 accuracy points at N <= 1000): NOT supported; paired
  differences on levels 1-4 are -2.7 to +1.5 points with no consistent sign and within the seed sd (0.4-1.7). P2 (better calibration under
  shift): partly; on 12-kind chains ECE is lower than plain at every N for surprise b=4 (0.064 vs 0.083 at N=250, 0.074 vs 0.131 at 500)
  and at 4 of 5 N for hypercorrect b=4, but with a cost in-distribution (hypercorrect b=4 ECE 0.201 vs 0.054 at N=250). P3 (does it beat the
  standard control?): NO; focal loss gives the lowest chain ECE at N >= 500 (0.065, 0.085, 0.131, 0.125) with no accuracy gain either, so
  surprise weighting adds nothing beyond known hard-example reweighting. Prioritised replay by surprise HURT at small N (-7 to -11 accuracy
  points at N <= 1000; neutral from N = 2000). Conclusion: C1/C6 as loss weights are not supported in logic; the falsifier (no effect beyond seed sd,
  nothing beyond focal) was met. Caveats: 3 seeds, a small beta grid, and 35 comparisons per metric, so isolated gaps (e.g. +2 on level 5 at N=250) are expected by chance.
- **Transfer ARC -> logic with the hypernetwork core** (`2026-10-16_103411_transfer_logic`, 3 seeds, 39 min; the FiLM run above was confounded by a
  4-8 point encoder deficit, which this fixes: shared encoder from scratch 78.0% at N=250 vs original 75.6%). **Fine-tuning: no source differs from
  no pretraining by more than 0.6 points at any N** (human minus RL-history +0.6 / +0.1 / -0.1 / -0.0 / +0.1 at N = 250..5000, paired sd 0.1-0.4;
  human minus scratch +0.2 / -0.3 / -0.2 / -0.2 / +0.0), and the logic-only positive control is also equal to scratch. **Frozen core:** ARC-pretrained cores
  are within about 0-2 points of a random frozen core (N=250: 58.7% random vs 59.6-61.2% ARC sources; N=5000: 90.2% vs 90.2-90.8%), so the earlier
  '+4-8 points over random' was an artefact of a weak encoder whose random features were poor; the logic-only control is +1-4 points over random
  (62.7% vs 58.7% at N=250), showing the test can see small same-domain benefits. **Conclusion: no cross-domain transfer from ARC to logic, from any
  kind of learning history (human, RL, answers), at this scale.** Untested reasons: the ARC learners are weak (about 57% of puzzles), ARC and logic may share little
  relational structure, and logic is learnable from scratch from a few hundred problems.

- **Signed surprise as an INPUT to a learner that learns from feedback** (`stml/model/logic_episodes.py`, `experiments/logic_surprise_input.py`, first run
  `2026-10-16_132403_logic_surprise_input`, 3 seeds, 400 test episodes x 10 claims). Setting: game worlds (levels 2-5), learner sees ~50% of the premises,
  judges 10 claims in a row, each true verdict is revealed and added to the world graph as an edge; surprise = the history-free judge's signed error
  (expected / underestimated / overestimated), 2-fold cross-fitted so flags do not leak. Predictions were written first. **Q1 true:** feedback helps, accuracy rises from
  66.9% at the first claim to 77.8% at claims 8-10; feedback minus no feedback is +9.0 / +12.4 points at claims 5-7 / 8-10 (paired sd 0.6-0.8); showing which claims were asked
  WITHOUT the verdicts does nothing (-0.8). Oracle (all premises) 99.6%, so feedback closes only about a fifth of the gap. **Q2 false:** feedback + surprise flags (as 72 edge types)
  minus feedback alone is -0.7 overall (-1.2, -1.6 late). **Follow-up (Q3, written after seeing that), same worlds, rerun
  `2026-10-16_132941_logic_surprise_input`:** the same flags delivered as 3 per-kind node features with the 24 edge types kept: `fb_n` minus `fb` = +0.4 overall (-1.1 at claims 5-7,
  +0.8 at 8-10), and its shuffled control `fb_n_shuf` minus `fb_n` = +0.5 (the SHUFFLED flags are slightly better). In this rerun `fb_s` minus `fb` = -0.8 and `fb_s` minus its shuffled
  control = +0.8 (sd 1.1); the +1.7 of the first run was therefore within run-to-run noise (about 1 point) and I withdraw the claim that the flags carry information.
  **Conclusion: a signed-surprise signal given as an input adds nothing beyond the revealed verdict; the learning comes from the feedback itself.** (Run-to-run noise of the
  same configuration is about 1 point, so differences under 2 are not interpretable.)

- **What limits learning from feedback** (`experiments/logic_feedback_scaling.py`, `2026-10-16_133616_logic_feedback_scaling`, 3 seeds, one factor at a time around
  300 worlds / 4 layers / 10 claims / 50% visible). Predictions written first, all met. Gain of the feedback learner over the static one at the last third of the claims
  (gap closed in brackets): training worlds 100 / 300 / 1000 = +8.1 (22%) / +11.9 (35%) / +14.3 (45%), still rising (**data-limited**); GNN layers 2 / 4 / 6 = +13.4 / +11.9 /
  +11.1 (**not depth-limited**); claims per episode 5 / 10 / 20 = +5.2 (15%) / +11.9 (35%) / +22.0 (70%) (**more revealed evidence helps most**; at 20 claims the learner closes 70% of the
  gap); visible fraction 0.3 / 0.5 / 0.7 = +15.1 / +11.9 / +6.2 (less visible, more to gain). Practical rules so far: feed the learner more feedback per episode and more training episodes,
  not more depth; shallow (2-layer) graph models are enough here.

- **Does the learner know which claims it can answer now?** (`stml/model/logic_episodes.py: knowable`, `experiments/logic_knowable.py`, `2026-10-16_141524_logic_knowable`, 3 seeds,
  400 test episodes x 10 claims, 50% of premises visible, fb graph with 1000 training worlds per level). Exact label: a claim is *knowable* at step t if a proof of its verdict exists
  from the visible premises plus the revealed true claims (about 18% of claims at t=0, 30% at t=7-9). **K1 false:** the verdict model's confidence separates knowable from unknowable
  claims with AUROC 0.92 (0.98 at t=0 falling to 0.86 at t=7-9), not < 0.75. It is 99.2% accurate on knowable claims and 62% -> 76% (t=0 -> t=7-9) on unknowable ones (mostly by guessing
  'open', which is never knowable). **K2 half true:** a dedicated 'proof available' head reaches AUROC 1.000 (an exact function of the graph), but gating on it LOSES to confidence gating under the
  standard honest reward: +0.321 vs +0.402 at lam=2 and +0.089 vs +0.270 at lam=4 (answer-all +0.325 / -0.125; oracle gate +0.101 / +0.097). **K3 true:** knowable share and accuracy rise
  with feedback (69% -> 83%). Interpretation: the standard honest reward (+1 right, -lam wrong) PAYS lucky guesses, so abstaining on unknowable claims (still 70% right by luck) looks costly; CLAUDE.md 4.5
  says a correct answer without a certificate is a lucky guess, which the reward did not implement. Follow-up (post hoc, motivated by 4.5): a luck-adjusted reward.

- **Luck-adjusted reward (Q0b, post hoc, rerun `2026-10-16_142024_logic_knowable`).** Reward: right and knowable +1, right but unknowable 0, wrong -lam, abstain -0.2. The standard-reward numbers replicate
  (certificate gate +0.325 / +0.090 vs confidence gate +0.393 / +0.264 at lam = 2 / 4). Luck-adjusted: certificate gate +0.100 / +0.096 / +0.092, confidence gate +0.115 / +0.084 / +0.061, oracle gate
  +0.102 / +0.099 / +0.094 at lam = 1 / 2 / 4; answer-all +0.030 / -0.193 / -0.639. **K4 false as pre-registered** (certificate beats confidence by 0.012 at lam = 2 and 0.031 at lam = 4, not >= 0.05, and
  confidence is better at lam = 1), but the learned proof-availability head is within 0.004 of the oracle gate. The reward ceiling is low (~+0.10) because only 26% of claims are knowable and every abstention costs 0.2.
  Conclusion: a proof-availability head is a near-perfect abstention gate; raw confidence already captures much of its value and is free; with a reward that pays luck, gating on proof availability looks worse than it is.

- **Active querying** (`experiments/logic_active_query.py`, `2026-10-16_143330_logic_active_query`, 3 seeds, 400 episodes, pool of 12 claims, 6 asks, 8 held-out claims judged afterwards, one feedback model trained on
  random asks and shared by all policies). Accuracy on the held-out claims after 6 asks (paired difference to random, sd): random 83.0%; **entropy 88.3% (+5.3, 0.3); margin 88.1% (+5.1, 0.3)**; infogain_oracle 85.5% (+2.5, 0.3).
  A1 true (oracle beats random by 2.5). **A2 false:** uncertainty sampling beats random by 5 points (predicted: no better; the idea that uncertain claims are mostly unknowable and teach little is wrong).
  A3 true (provable held-out claims 37.5% oracle vs 30.6% random). Surprise: plain uncertainty sampling beats the greedy 'make the most claims provable' oracle in accuracy (88.3% vs 85.5%) although the oracle
  makes more claims provable (32% vs 24% after one ask), so provability is a poor proxy for what the learner gains. Mechanism follow-up queued (Q1b).

- **Why uncertainty-based asking wins** (Q1b, `experiments/logic_active_query_mechanism.py`, `2026-10-16_143926_logic_active_query_mechanism`, 3 seeds). Predictions B1-B2 held. Gain over random after 6 asks: entropy
  +5.2 (0.6); **entropy restricted to unprovable candidates +5.1** (98% of the gain); entropy restricted to provable candidates -1.1; random provable -3.2; random unprovable +2.3. So the advantage comes from
  asking claims whose answer cannot yet be derived (an already-provable claim teaches nothing it could not derive, and wastes budget); among the unprovable ones, uncertainty roughly doubles the gain. Design implication for
  the human game: 'propose statements that follow' invites asking what one already believes provable, which is the low-information kind.

- **Game-log readiness (Q-L, 2026-10-16).** Built and tested the whole path BEFORE the real logs: `stml/synthetic_logs.py` turns a synthetic participant into a real `stml-log/1`
  document (written only to temp folders, never to `logs/`); `stml/game_analysis.py` + `scripts/analyze_game.py` compute per-participant endpoints and the planned analyses (E1 confidence-AUROC vs 0.5,
  E2 hypercorrection, E3 diagnostic vs scalar feedback if the session records `arm`, learning and abstention curves over the 15 worlds, abstention by truth and proof size), a sizing update from the
  OBSERVED between-person sd, and a table of observed values next to what the simulation assumed. Tests (`tests/test_game_analysis.py`): logs go through the real loader and converter; endpoints recomputed
  from the log equal the simulation's exactly; the analysis recovers the built-in effects; it degrades gracefully with one participant. **When the logs arrive: `python scripts/analyze_game.py logs/`**
  (creates `runs/<time>_game_analysis/report.md`, never modifies the logs). NOTE for the game: to record the feedback arm, put `"arm": "scalar" | "diagnostic"` into the session (`session.arm` or the
  `session_start` event); otherwise E3 cannot be run. The game as built has no arm manipulation (everyone gets the diagnostic critic), so E3 needs a game change if you want that experiment.

- **Hyperparameter search for the feedback learner (H1)** (`experiments/logic_hparam_search.py`, `2026-10-16_145253_logic_hparam_search`; 12 configs on validation episodes only, 2 seeds; top 3 + default confirmed
  on test with 3 new seeds). **T1 false:** best validation configs beat the default on test by +0.5 (0.7) and +0.7 (0.7) points and the third by -0.0, all inside the seed sd; validation accuracy of all 12 configs spans
  73.1-77.0% and the default (76.3%) is in the upper half. **T2 mostly false:** the winners were wide+long (128 wide, 16 epochs, batch 256) and deep+heavily regularised (6 layers, wd 1.5e-3), but the gaps are noise.
  Conclusion: the learner is not tuning-limited (it is data-limited); keep the defaults (lr 1e-3, width 64, 4 layers, 10 epochs, wd 1e-4, batch 128). Next, per rule 7b, the robustness of the promising ACTIVE-QUERYING result (H2).

- **Active querying, deeper (H2)** (`experiments/logic_active_query_deeper.py`, `2026-10-16_151014_logic_active_query_deeper`, 3 seeds, 300-400 test episodes). Entropy-asking minus random-asking (points, sd):
  budget 3 / 6 / 10 = +5.0 (0.2) / +5.3 (0.6) / +1.7 (0.2); pool 8 / 12 / 20 = +3.1 (0.6) / +5.3 (0.6) / +5.7 (0.2); model default / wide-long / shallow = +5.3 / +5.0 (0.4) / +5.4 (1.1); on-policy trained model +5.9 (0.4)
  vs equal-data random-order control +5.3 (0.5), difference +0.7 (0.2). D1 mostly true (positive and >= 2 at every budget, but it SHRINKS from 6 to 10 asks: with 10 of 12 candidates asked, random covers nearly the whole pool);
  D2 true (grows with pool, saturating after about 12); D3 true (not an artefact of one configuration); D4 true (on-policy training does not hurt; +0.7 is near the ~1 point run-to-run noise).
  Conclusion: asking where the model is uncertain beats random asking by about 5 accuracy points whenever the budget is a minority of the pool (3-6 of 12, 6 of 20), across model sizes and depths; the advantage vanishes as the
  budget approaches the pool size. This supports the Socratic question as an uncertainty/information choice (F7), but the 'information gain' that matters is where the learner cannot yet derive the answer (Q1b), not the greedy
  'make more claims provable' proxy (Q1).

- **Abstention gates under shift in the hidden-premise regime (Q3)** (`experiments/logic_knowable_shift.py`, `2026-10-16_153956_logic_knowable_shift`, 3 seeds; trained on levels 2-4 with 50% of premises visible). AUROC for
  'a proof exists now' (confidence of the verdict model / the proof-availability head): in-dist 0.918 / 0.999; 30% visible 0.940 / 1.000; 70% visible 0.891 / 1.000; level 5 0.808 / 1.000; level 5 + 30% visible 0.870 / 1.000.
  Luck-adjusted reward (lam = 4, in-dist thresholds) confidence / head / oracle: level 5 -0.119 / +0.003 / +0.016; level 5 + 30% visible -0.153 / -0.105 / -0.100; 70% visible +0.198 / +0.221 / +0.222; 30% visible +0.030 / +0.050 / +0.048.
  **S1 half true** (the head stays at 1.000; confidence does NOT fall by 0.05 under visible-fraction shifts, only under the level-5 shift); S2 true in direction; **S3 true** (head within 0.013 of the oracle in every shifted set, confidence 0.135 and 0.053
  below the oracle on the two level-5 sets). Few-shot recalibration (100 items) did not help either gate (head -0.015 vs +0.003 at level 5), so the confidence gate's failure on larger unseen worlds is not a threshold problem.
  Conclusion: a proof-availability head is a shift-robust abstention gate in this regime; raw confidence is not. Limit: shifts so far reach 7 kinds; larger worlds are tested next (Q3b).

- **Proof-availability head on larger worlds and deep chains (Q3b)** (`experiments/logic_knowable_big.py`, `2026-10-16_155440_logic_knowable_big`, 3 seeds, trained on levels 2-4 with <= 6 kinds, tested on 8 and 10 kinds, trees and long chains, 10-12 premises).
  Verdict accuracy falls to 67% / 68% (trees) and 55% / 52% (chains), yet AUROC of the head for 'a proof exists now' is 0.997 / 0.999 / 0.988 / 0.989 (8 tree, 10 tree, 8 chain, 10 chain) against 0.808 / 0.774 / 0.737 / 0.765 for the verdict model's confidence.
  Luck-adjusted reward at lam = 4 (head / confidence / oracle): 8 tree -0.046 / -0.099 / -0.018; 10 tree -0.106 / -0.183 / -0.080; 8 chain -0.131 / -0.144 / -0.114; 10 chain -0.158 / -0.148 / -0.116.
  R1 true (head AUROC >= 0.95 everywhere); R2 mostly true (3 of 4 below 0.80); R3 mostly true (head within 0.042 of the oracle; confidence 0.08-0.10 below it on trees but only ~0.03 on chains).
  Conclusion: a learner can know what it cannot prove even when it is poor at the task itself, up to the 10-kind worlds tested. Caveat: the head is trained on exact 'knowable' labels from a symbolic engine; human data will not provide them
  (the game's critic could, via the support it computes).

- **The pipeline (Q4)** (`experiments/logic_pipeline.py`, `2026-10-16_161007_logic_pipeline`, 3 seeds; ask 6 of 12 candidate claims by uncertainty or at random, learn from the revealed verdicts, judge 8 held-out claims, abstain with a gate;
  trained on levels 2-5, tested in-distribution and on 8-kind chains and 10-kind trees; two encoders). **In-distribution baseline (original encoder):** verdict accuracy 86.5% with entropy asks vs 81.5% with random asks; luck-adjusted reward at lam = 4
  (abstain = -0.2): answer-all -0.187, confidence gate +0.206, head gate +0.206, oracle gate +0.208 (random asks: -0.448 / +0.122 / +0.135 / +0.140). The hypernetwork encoder: 85.4% / +0.202 head gate. Under the STANDARD reward the confidence gate beats the head gate
  (+0.515 vs +0.291), because that reward pays lucky guesses. **Large worlds:** verdict accuracy collapses to 57-60% (8-kind chains) and 72-75% (10-kind trees); every gate's reward is near zero or negative (8-kind chains, entropy asks: confidence -0.046, head +0.025, oracle +0.063); entropy asks help only +1.7 to +2.9 points.
  P1 true in-distribution (+5.0), weak on large worlds; **P2 false in-distribution** (head = confidence = oracle within 0.002), true only on 8-kind chains (+0.071); P3 true in-distribution, indistinguishable on large worlds; P4 mostly true (<= 1.1 points in-distribution, -2.5 on 10-kind trees).
  Conclusion: in the regime the learner was trained for, ask-by-uncertainty + feedback + a proof gate reaches the oracle abstention reward; the gate matters under shift (Q3, Q3b) while the limit on larger worlds is the verdict learner's own capacity/training range, which is the next follow-up (Q4b: extended training).

- **Pipeline with extended training (Q4b)** (`experiments/logic_pipeline.py --extended --encoders original`, `2026-10-16_162429_logic_pipeline_extended`, 3 seeds; training also on 8- and 10-kind worlds, trees and long chains, 400 each).
  Verdict accuracy with entropy / random asks: in-dist 87.2% / 82.0% (was 86.5 / 81.5); **8-kind chains 78.5% / 74.8% (was 59.4 / 57.5: +19.1 / +17.3)**; 10-kind trees 76.7% / 74.4% (was 74.9 / 72.0: +1.8 / +2.4).
  Luck-adjusted reward (lam = 4), entropy asks, confidence / head / oracle: in-dist +0.235 / +0.218 / +0.220; 8-kind chains +0.146 / +0.163 / +0.177 (was -0.046 / +0.025 / +0.063); 10-kind trees -0.044 / -0.036 / -0.025.
  **E1 true for chains** (accuracy +19, entropy asks beat random by 3.7) and **false for 10-kind trees** (rise < 4: they were not collapsed); **E2 false** (head minus confidence = +0.017 on chains, +0.008 on trees): once the verdict model is trained in range,
  its confidence is reliable and the head's advantage mostly disappears, consistent with Q3 (the head matters under shift). In-dist accuracy moved < 1 point as predicted. Conclusion: the large-world collapse was a TRAINING-RANGE problem for chains;
  training on the target range restores accuracy, the benefit of asking by uncertainty, and the gates. NOTE: the 'oracle' gate is NOT an upper bound under the luck-adjusted reward (a confident unprovable answer that is right earns 0 > -0.2 for abstaining):
  the in-dist confidence gate (+0.235) beats it (+0.220).

- **Feedback type in the sequential regime (Q8)** (`stml/model/logic_flags.py`, `experiments/logic_feedback_type.py`, `2026-10-16_164022_logic_feedback_type`, 3 seeds, 400 test episodes x 10 claims, 50% of premises visible).
  Accuracy at claims 8-10 (all positions): static 67.0% (68.3%); asked-only 66.3% (67.4%); BINARY 'was the history-free judge's guess right' 69.8% (69.3%); binary with shuffled flags 66.5% (67.1%); full VERDICT 81.3% (76.6%).
  F1 true: verdict minus binary = +9.3 / +11.5 points at claims 5-7 / 8-10 (paired sd 1.6 / 0.3). F2 true but small: binary minus static = +1.7 / +2.8 (sd 1.9 / 1.1), binary minus its shuffled control = +3.0 / +3.3 (so the single bit is real information).
  F3 true: binary gains about 15-19% of what the verdict gains (+2.8 vs +14.4 at claims 8-10). Machine analogue of the scalar vs diagnostic arm: a right/wrong signal teaches about a fifth of what the verdict does; overall verdict benefit +8.3 points,
  consistent with the 7-8 point effect assumed in the study-sizing simulation (people may differ).

- **Does 'which premises were used' add to the verdict? (Q8b)** (`experiments/logic_feedback_why.py`, `2026-10-16_165451_logic_feedback_why`, 3 seeds; 74% of the visible premises are marked by the end of an episode). Accuracy at claims 8-10:
  static 66.9%, verdict 81.9%, verdict + marks 79.6%, verdict + RANDOM marks 80.0%. **G1 false:** marks minus verdict = -0.8 / -2.4 points at claims 5-7 / 8-10 (sd 0.6 / 0.7); [CORRECTION after Q5: in a second run the same comparison was +0.2 / +0.4, so 'slightly hurt' was noise of 1-2 points; the safe statement is 'adds nothing', range -0.9..+0.4]. G2 true: random marks minus verdict
  = -0.9 / -1.9; real vs random marks -0.4 to +0.1. So the verdict carries the benefit, and extra structure on the feedback edges (here premise marks; before: signed surprise) costs about as much in diluted training data as it gives.
  For the game: a diagnostic-vs-scalar contrast should be about whether the verdict is shown, not about proof detail (which may still matter for human understanding, not for this learner). Hypothesis queued (Q5): the null is a data-dilution effect.

- **Data scaling and the dilution test (Q5)** (`experiments/logic_feedback_data_scaling.py`, `2026-10-16_170925_logic_feedback_data_scaling`, 3 seeds, 400 test episodes). Late-claim (7-10) accuracy at 600 / 2000 training worlds per level: static 69.8% / 71.3%;
  verdict 81.3% / 85.1%; verdict + marks 81.5% / 85.4%. **H1 true:** the verdict learner gains +3.8 points from 600 to 2000 worlds (static +1.5), still rising (fb minus static: +11.5 -> +13.7). **H2 false:** verdict+marks minus verdict = +0.2 (sd 1.2) at 600 worlds and +0.4 (0.7) at 2000,
  so the nulls for structured feedback (signed surprise, premise marks) are NOT a data-dilution effect. Note the run-to-run noise: the same comparison at 600 worlds was -0.8..-2.4 in Q8b; late-position differences below about 2 points are not interpretable.

- **Replication on fresh worlds (Q9)** (`experiments/logic_replication.py`, `2026-10-16_172518_logic_replication`; 3 model seeds x 5 independent test sets of unseen worlds, 100 per level, 20 claims each; models trained on levels 2-5, 600 worlds per level).
  (a) feedback gain at claims 15-20: +20.9 points (sd over test sets 0.6, over seeds 0.1; all 15 cells +19.7..+21.8); (b) entropy minus random asking after 6 asks: +4.7 (0.2, 0.1; cells +3.5..+5.5); (c) head gate minus confidence gate, luck-adjusted reward (lam = 4): in-distribution
  +0.014 (0.004, 0.004; cells +0.001..+0.026); 8-kind chains +0.014 (0.015, 0.013; cells -0.066..+0.065). **Ra half true** (positive everywhere, but larger than the earlier +8..+14, which used claims 7-10 of 10; the match is the K = 20 sweep's +22.0), **Rb true**, **Rc in-distribution true and chain part FALSE** (head beats
  confidence by >= 0.03 in only 1 of 5 chain test sets), **Rd true** (world-sampling sd <= 0.6 points; smaller than expected). CORRECTION: the head's REWARD advantage on 8-kind chains (+0.071 in Q4 with entropy asks, single test set of 100 episodes) is not reliable; with 800 judged claims per cell the reward noise is about +-0.03..0.06.
  The head's AUROC advantage (0.99 vs 0.74-0.81, Q3/Q3b) is a much larger effect and stands. The ~1 point run-to-run noise seen across scripts is therefore not world sampling; it comes from training-set composition and initialisation.

- **Q9b: better-powered chain-world gate test** (`experiments/logic_replication.py --only-c --nper-chain 400`, `2026-10-16_174002_logic_replication_chain400`, 3 seeds x 5 sets x 400 episodes of 8-kind chains). Head gate minus confidence gate (luck-adjusted, lam = 4): **+0.092** (sd over sets 0.016, over seeds 0.023; all 15 cells >= +0.051);
  in-dist +0.026 (cells +0.008..+0.050). In Q9 (100 episodes per set, retrained models, same first 100 worlds per set) the same comparison was +0.014 (cells -0.066..+0.065) and in-dist +0.014. So the SIGN is positive in both, but the size moved by ~0.08 between two TRAININGS (vs a within-run seed sd of 0.023):
  the out-of-range chain behaviour (models trained on levels 2-5, tested on 8 kinds) is unstable across trainings, and the threshold fitted on in-distribution validation is sensitive to it. The pre-registered Q9b prediction (small positive gap) held for the sign; the size is NOT precisely estimated (+0.01..+0.09). Q9c measures the training-to-training distribution.

- **Q9c: training-to-training variance of the chain gap** (`logic_replication.py --seeds 8 --sets 1 --only-c --nper-chain 400`, `2026-10-16_175419_logic_replication_chain400`; 8 independent trainings, ONE fixed 400-episode chain set). Head minus confidence gate (luck-adjusted, lam = 4):
  8-kind chains **+0.053, sd over trainings 0.034** (cells -0.013..+0.091); in-distribution +0.016 (sd 0.011, cells +0.001..+0.040). Pre-registered prediction (sd >= 0.04) narrowly MISSED (0.034), but the out-of-range gap varies about 3x more between trainings than the in-distribution one, as expected.
  Reading: the head gate's reward advantage on 8-kind chains is positive on average (about +0.05; standard error 0.012 over 8 trainings), is not guaranteed for any single training (1 of 8 trainings slightly negative), and the Q9b +0.092 and Q9 +0.014 are both within the range of single trainings. Report it as '+0.05 +- 0.03 across trainings'.

- **Q6: surprise-gated memory** (`experiments/logic_gated_memory.py`, `2026-10-16_180935_logic_gated_memory`, 3 seeds, 400 test episodes x 10 claims). Only verdicts the history-free judge got WRONG are stored as graph edges (32% of claims). Accuracy at claims 7-10:
  static 67.0%, keep-all 81.4%, store-surprising 76.5%, capacity-matched random store 73.6%, store-expected (the opposite gate) 73.2%. **M1 borderline true:** surprising minus random +2.9 (paired sd 1.8; per-seed about +3.3, +2.8, +0.1, so one seed shows nothing); **M2 true:** keep-all minus surprising +4.8 (1.1);
  **M3 true:** expected minus random -0.3. The gate keeps 1/3 of the memory and about two thirds of the feedback gain (+9.5 of +14.3), more per stored item than random or expected storage, but it is NOT free (-4.8 vs keep-all). The M1 evidence is weak (3 seeds); rerun with 8 seeds queued as Q6b.
  **Q6b (8 seeds, `2026-10-16_182345_logic_gated_memory`):** surprising minus random-same-size at claims 7-10 = **+2.2** (paired sd 1.4, SE 0.5, per seed +1.6 +2.8 -0.7 +3.4 +2.5 +3.6 +2.2 +1.9; 95% CI about +1.0..+3.3), at claims 4-6 +3.0 (1.0); keep-all minus surprising +4.8 (1.0); expected minus random +0.1. Pre-registered prediction (>= +2, sd <= 1.5, no seed below +1) held on mean and sd, one seed (-0.7) violated 'no seed below +1'.
  So the surprise gate is a small but real advantage over a random memory of the same size (CI excludes 0), well short of keep-all; it is a capacity argument (a third of the memory, two thirds of the gain), not an accuracy gain. Not promising enough for a hyperparameter search (effect ~1.6 seed sd).

- **Q10: learned asking policy** (`experiments/logic_learned_asking.py`, `2026-10-16_183959_logic_learned_asking`, 3 seeds, 400 test episodes; frozen feedback judge, REINFORCE policy over candidate features trained on 400 unseen worlds per level, 80 iterations x 128 episodes; reward = held-out accuracy after 6 asks of a pool of 12).
  Accuracy: random 83.1%, hand entropy rule 88.2%, **learned (with judge probabilities/entropy/margin) 88.1%**, learned with structure features only 83.4%. **L1 true:** learned minus entropy -0.2 (sd 0.2). **L2 true:** structure-only minus entropy -4.8 (0.4), minus random +0.3. Reading: RL finds nothing better than 'ask where the judge is uncertain'
  with these features (the hand rule is at the ceiling of this feature set), and without the judge's uncertainty there is nothing to learn from (structure alone is no better than random). This supports 'the Socratic question = the learner's own uncertainty', not a structural heuristic. Limit: features are low-dimensional; a policy reading the graph embedding might differ.

- **Q11: noisy feedback** (`experiments/logic_noisy_feedback.py`, `2026-10-16_185430_logic_noisy_feedback`, 3 seeds, 400 test episodes, 600 training worlds per level; each revealed verdict replaced with probability p by one of the two wrong verdicts, the learner is not told which). Accuracy at claims 7-10: static 66.7%; clean feedback 80.3% (gain +13.6);
  matched-noise training p = 0.1 / 0.2 / 0.3: 76.8 / 71.9 / 70.1 (gain +10.1 / +5.1 / +3.4 = 74% / 38% / 25% of the clean gain); trained clean, tested noisy: 76.5 / 72.9 / 67.6 (gain +9.7 / +6.2 / +0.9).
  **N1 false in magnitude, true in direction:** the gain falls much faster than predicted (prediction 85/65/45%), but stays positive at p = 0.3 for the matched learner. **N2 half true:** matched minus clean-trained = +0.3 (0.9) at p = 0.1, -1.0 (1.8) at p = 0.2 (not significant), +2.5 (0.5) at p = 0.3.
  Reading: learning from a teacher is fragile: 20% wrong verdicts remove about 60% of the benefit, and a learner trained on exact feedback is about as good as one trained on noisy feedback up to 20% noise, i.e. the matched learner does not learn to discount unreliable edges (except at 30%, +2.5). The exact critic of the game is safe; any
  setting where the verdicts come from people or other models needs a mechanism that checks a new verdict against the rest of the graph. Q11b tests whether more training data teaches discounting.

- **Q11b: does more data teach discounting?** (`logic_noisy_feedback.py --worlds 2000 --ps 0.0,0.2,0.3`, `2026-10-16_190852_logic_noisy_feedback`, 3 seeds). Clean gain +14.9. Matched-noise training: p = 0.2 gain +8.2 (**55%** of clean, was 38% with 600 worlds), p = 0.3 +4.8 (32%, was 25%); clean-trained learner +6.0 / +1.6.
  Matched minus clean-trained **+2.1 (0.3) at p = 0.2 and +3.1 (1.3) at p = 0.3**. Prediction (share >= 50% at p = 0.2; matched beats clean-trained by >= 2 at both) held. So robustness to a wrong teacher is learnable but data-hungry: with 3.3x data the learner starts to discount edges that conflict with the rest of the graph.

- **Q12: surprise gate under noisy feedback** (`logic_gated_memory.py --noise 0.2 --seeds 4`, `2026-10-16_192410_logic_gated_memory_noise0.2`; 600 worlds per level; 20% of revealed verdicts wrong, surprise judged against the shown verdict). Accuracy at claims 7-10: static 66.9%, keep-all 74.7%, store-surprising 69.3%, capacity-matched random 70.2%, store-expected 71.5%.
  **N3 true:** surprising minus random **-0.9 (sd 1.8)** (was +2.2 with exact feedback), i.e. the advantage disappears; the opposite gate (store verdicts that AGREE with the judge) is +1.3 over random (sd 1.6, not significant). Reading: with an unreliable teacher, 'surprise' is as often a teacher error as a genuine discovery, so a surprise-gated store loses its edge;
  a surprise-gated memory is only worth having if the feedback is trustworthy (the game's exact critic) or if surprising items are verified first. This is a boundary condition on C2/C6 in `docs/ARCHITECTURE_V2.md`.

- **Q13: recalibrating the abstention threshold from COMMITTED answers only** (`experiments/logic_selective_recalibration.py`, `2026-10-16_195341_logic_selective_recalibration`; offline replay of the saved scores of `2026-10-14_195306_logic_h3_thresholds`: 5 judge seeds, small and extended training, trees and chains of 16/24/32 kinds, lam = 1, 2, 4: 36 cells, 50 random stream orders each; threshold re-fitted every 25 items, start = Bayes threshold (lam+0.2)/(1+lam) of the calibrated confidence).
  Mean reward per item over the 36 cells: fixed +0.593, selective (committed answers only) +0.609, selective with exploration e = 0.05 / 0.1 / 0.2: +0.614 / +0.614 / +0.610, full labels (reference, not deployable) +0.629, oracle threshold +0.647. **R1 true:** in the 8 cells where full-label recalibration beats the fixed threshold by > 0.02 (all overconfident chains under small-world training), selective recalibration recovers 94% of the gain (e.g. chain-32 lam 4: -0.682 fixed, -0.180 selective, -0.169 full labels).
  **R2 true on average** (best exploration only +0.005 over no exploration), but with a nuance the prediction missed: where the fixed threshold is already about right (extended training, trees) selective recalibration with e = 0 LOSES 0.02-0.04 (the threshold ratchets up from early noisy labels and cannot come back down without information), and exploration of 5-10% removes most of that loss.
  Reading: a deployed learner can repair overconfidence from its own committed answers, at the price of a small loss when it was already calibrated; a little exploration (5-10%) is cheap insurance. Caveats: cells share the same 5 models and worlds (not independent), the stream is exchangeable, no uncertainty on the cell means.

- **Q13b: hysteresis against the ratchet** (`logic_selective_recalibration.py --margins ...`, `2026-10-16_195603_logic_selective_recalibration` and `2026-10-16_195929_logic_selective_recalibration`; same 36 cells, committed answers only, no exploration). A re-fitted threshold is adopted only if it beats the current one by a margin on the observed items.
  Share of selective_0's loss removed in the 12 calibrated cells (extended, lam <= 2) / share of its gain kept in the 8 overconfident cells: margin 0.02: 61% / 99%; **0.05: 77% / 100%** (prediction >= 70% / >= 80% held); 0.1: 88-91% / 99%; 0.2: 97% / 93%; 0.3: 99% / 87%; 0.5: 100% / 73%.
  Mean reward over all 36 cells: fixed +0.593, selective_0 +0.610, margin 0.05 +0.622, **0.1 +0.625-0.626 (89% of the full-label +0.629's gain)**, 0.2 +0.625, 0.3 +0.624, 0.5 +0.619; oracle +0.647. So a margin of 0.05-0.2 closes most of the ratchet loss at no cost to the repair; the flat optimum 0.1-0.2 is chosen on the same replay, so treat the exact number as a default, not a tuned result. A deployable rule: start at the Bayes threshold, re-fit from committed answers, move only when the gain is clear.

- **Q14: asking by uncertainty with noisy feedback** (`experiments/logic_noisy_asking.py`, `2026-10-16_200437_logic_noisy_asking`, 3 seeds, 400 test episodes, p = 0.2 of the SHOWN verdicts of asked claims wrong). Accuracy on 8 held-out claims after 6 asks, random / entropy / difference: exact feedback 83.6 / 88.0 / **+4.4** (0.3); matched-noise judge 75.9 / 79.4 / **+3.5** (0.7); clean-trained judge on noisy feedback 74.8 / 78.0 / **+3.2** (0.1).
  **Y1 true** (>= +3 held), **Y2 true** (accuracy falls 7.7 / 8.6 points). The advantage of asking where the model is uncertain shrinks by about a fifth but survives noise, so it does not depend on an exact teacher; the learner as a whole loses ~8 points of accuracy to 20% wrong verdicts.

- **Q15: lemmas as a scratchpad** (`experiments/logic_lemma_scratchpad.py`, `stml/model/logic_lemma.py`, `2026-10-16_202022_logic_lemma_scratchpad`, 3 seeds, 25 fresh test worlds per cell, 21 balanced target claims each; fully visible worlds, 6-layer judge trained on 5/6/8 kinds, trees and chains, 150 worlds each). The judge writes its confident 'follows' claims (p > 0.9) as 8 extra edge types and re-judges, up to 3 rounds.
  Accuracy plain judge A / lemma-trained round 0 / own lemmas after 3 rounds / true-lemma oracle (30%): tree-16 91.9 / 90.4 / 94.0 / 93.7; chain-12 92.4 / 91.5 / 95.2 / 94.9; chain-16 84.0 / 84.1 / 88.9 / 90.5; 8-kind sets 98-99.6 everywhere. Random false lemmas (same count): 65-83% (**-13 to -20 points**).
  Self-lemma gain over round 0 (points, sd over seeds): tree-8 +0.7, tree-12 +1.6 (0.6), tree-16 +3.6 (1.0), chain-8 +0.8, chain-12 +3.7 (3.0), chain-16 +4.8 (2.2); over the plain judge A: +0.3, +0.8, +2.0 (0.2), -0.1, +2.8 (2.9), +4.9 (3.9). Lemma precision 99% / 94-96% / 89-93% at 8 / 12 / 16 kinds.
  **P1 false in size** (oracle lemmas add only +0.4..+6.4: the plain judge is already at 84-98%, so the ceiling is low). **P2 half true:** +3.7 over round 0 at chain-12 (>= 3) but +2.8 over A (< 3; sd about 3), precision 95% (>= 90); on chain-16 +4.8 / +4.9 with sd 2-4. **P3 false:** the tree-16 gain (+3.6) is about three quarters of the chain-16 gain, not half. **P4 strongly true:** false lemmas hurt much more than true ones help.
  Reading: the self-written scratchpad reaches the level of true lemmas (chain-12 94.9 = oracle 94.9; chain-16 88.9 vs 90.5) with no sign of error compounding (accuracy rises or holds over rounds), and the gain grows with world size and depth (0 at 8 kinds, +2..+5 at 16). It is a small but consistent plus on the largest worlds, not a fix of a collapse: with 6 layers and
  fully visible premises the plain judge does not collapse at 16 kinds (84-92%). The strong sensitivity to false lemmas (the same fragility as noisy feedback in Q11) is the main risk; lemma training costs about 1 point at round 0. Next: Q15b, larger worlds (24/32 kinds), more rounds and thresholds.

- **Q15b: lemmas on larger worlds** (`logic_lemma_scratchpad.py --test-sizes 16,24,32 --ntest 12 --rounds 5 --taus 0.8,0.9,0.97 --no-random`, `2026-10-16_202652_logic_lemma_scratchpad`, 3 seeds, 12 fresh test worlds per cell; the same 5-8-kind-trained models as Q15). Self-lemma gain after 5 rounds over the plain judge A (tau 0.9; points, sd over seeds) / over the lemma-trained round 0 / true-lemma oracle over round 0 / lemma precision after 5 rounds:
  chain-16 +5.4 (4.6) / +7.5 (3.7) / +9.4 / 93%; **chain-24 +4.1 (1.8) / +4.4 / +7.7 / 82%; chain-32 +1.1 (0.8) / +0.4 / +6.6 / 75%**; tree-16 +2.0 (0.6) / +4.9 / +3.0 / 93%; tree-24 +0.8 / +0.8 / +2.1 / 85%; tree-32 +0.5 / +1.2 / +1.5 / 82%. Plain-judge accuracy at 16 / 24 / 32 kinds: chains 83.7 / 80.0 / 75.4, trees 94.2 / 94.6 / 93.0.
  Thresholds (chain-16 / 24 gain over A, precision after 5 rounds): tau 0.8: +6.0 / +4.2 (92% / 80%); 0.9: +5.4 / +4.1 (93% / 82%); 0.97: +4.4 / +3.7 (95% / 85%). More rounds never hurt: accuracy rises over rounds 1-3 and is flat afterwards (no compounding).
  **Prediction FALSE** (gain >= +6 at 24-kind and >= +8 at 32-kind chains; got +4.1 and +1.1; falsifier met: the 32-kind gain is below the 16-kind gain). Held: higher tau gives higher precision and a smaller gain; no error compounding up to 5 rounds.
  Reading: the scratchpad's gain peaks at about 16 kinds and fades with size because the judge's own lemmas get less reliable (precision 93% -> 82% -> 75% on chains, and a 0.97 confidence is not enough: calibration breaks out of range) while true lemmas would still give +6.6..+9.4. So the headroom is there but locked behind lemma precision; the lemma-trained judge also costs 2-3 points at round 0 on 16-kind sets.
  A useful lemma writer would need verification (calibration at the new size or an independent check), the same lesson as for noisy feedback (Q11) and the abstention threshold (Q13).

- **Q16: auxiliary world-model loss (predict the hidden premises)** (`experiments/logic_aux_worldmodel.py`, `stml/model/logic_aux.py`, `2026-10-16_215823_logic_aux_worldmodel`, 3 seeds, 400 test episodes x 10 claims, aux weight 0.5, hidden-premise episodes with 50% of the premises visible; 300 and 1000 training worlds per level). The run took 13.7 hours because the Mac was on battery in Low Power Mode (about 0.5% of normal CPU; the same script ran a seed in 61 s when plugged in).
  Accuracy at claims 7-10 / overall, fb vs fb + aux: 300 worlds 77.4 / 73.7 vs 77.8 / 74.4 (**+0.3 (0.8) late, +0.7 (0.8) overall**); 1000 worlds 82.8 / 77.6 vs 82.4 / 77.6 (**-0.3 (0.7) late, -0.1 (0.4) overall**). Shuffled-target control minus fb: -0.1 / -0.2 overall. Without feedback (static) the loss changes nothing (-0.2 at 300, -0.1 at 1000). Positions 1-3 show +1.5 (1.7) and +1.0 (0.3) for fb_aux over fb but position 0 is -1.2 at 1000 and the pattern is not consistent, so I do not read it as an effect.
  **W1 false** (predicted >= +2 at claims 7-10 at 300 worlds; got +0.3), **W2 moot** (no gain at 300 to halve), **W3 true** (the control adds nothing), **W4 false** (no gain without feedback). **Null: asking the encoder to predict the hidden premises does not relieve the data limit of the feedback learner.** Plausible reason: the verdict loss on 10 claims per world already forces the encoder to infer the same hidden structure, so the extra link-prediction target is redundant supervision.
  Data scaling stays the one lever that works (fb minus static: +11.8 late at 300 worlds, +15.1 at 1000).

- **Q17: re-masking the same worlds as free data** (`experiments/logic_remask_augmentation.py`, `2026-10-17_132732_logic_remask_augmentation`, 3 seeds, 400 test episodes x 10 claims, 300 training worlds per level, hidden-premise episodes with verdicts as edges; wall time 7.2 h because the Mac was in Low Power Mode on battery until the end). 'Re-mask' = new random draw of which premises are visible and which claims are asked, for the same worlds.
  Accuracy at claims 7-10 / overall: base (1200 episodes, one per world) 78.1 / 74.0; **remask3 (3600 episodes from the same 1200 worlds) 81.5 / 77.1**; worlds3 (3600 episodes from 3x distinct worlds) 82.0 / 76.7; **remask10 (12000 episodes) 84.7 / 79.4**. Paired: remask3 minus base +3.4 (0.9) late, +3.1 (0.3) overall; worlds3 minus base +3.9 (1.1) / +2.8 (0.7); remask10 minus remask3 +3.2 (0.4) / +2.4 (0.6); worlds3 minus remask3 +0.5 / -0.4.
  **F1 true** (>= +1.5), **F2 false** (re-masking recovers 88% of the gain of 3x distinct worlds, predicted <= 60%), **F3 false** (remask10 is +3.2 over remask3, not equal). Re-masking is therefore as good as new worlds at this size and keeps helping at 10 draws, for free (no new world generation, no labelling). CAVEAT: the re-masked sets also give more gradient steps (3x, 10x); the equal-steps control is Q17b.

- **Q17b: equal-steps control for re-masking** (`logic_remask_augmentation.py --conds base,base_x3,base_x10,remask3,remask10`, `2026-10-17_205222_logic_remask_augmentation`, 3 seeds, plugged in). Accuracy at claims 7-10 / overall: base 78.4 / 74.3; **base trained 3x longer 78.2 / 74.1; 10x longer 77.2 / 73.6** (more steps do not help, 10x slightly hurts: -1.3 late); remask3 82.5 / 77.6; remask10 84.8 / 79.5.
  Paired: remask3 minus base_x3 +4.3 (0.9) late, +3.4 (0.2) overall; **remask10 minus base_x10 +7.6 (0.7) late, +6.0 (0.3) overall**; remask10 minus remask3 +2.3 (0.3) late, +1.9 (0.6) overall. **F4 true** (the longer-trained baseline recovers none of the gain), **F5 true** (+1.9 >= +1). So the benefit is the DIVERSITY of the episodes, not the number of gradient steps. (Q17's base differs from Q17b's base by 0.3 points: run-to-run noise.)
  Reading: for the data-limited feedback learner, re-drawing which premises are hidden and which claims are asked from the same worlds is a free substitute for new worlds (88% of the gain of 3x distinct worlds at 3 draws) and keeps helping up to 10 draws (+6.0 points overall over an equal-steps baseline). It is the strongest cheap lever found so far; Q18 tests its scaling (draws, world count) per rule 7b.

- **Q18: scaling of re-masking** (`experiments/logic_remask_scaling.py`, `2026-10-17_210911_logic_remask_scaling`, 3 seeds, 400 test episodes x 10 claims, plugged in, 25 min). Accuracy at claims 7-10 (all claims in brackets) for W worlds per level x R re-masked draws (R x W x 4 episodes, 10 epochs):
  100 worlds: R=1 72.1 (68.6), R=3 78.0 (73.3), R=10 83.1 (77.1), R=30 84.0 (79.4); 300 worlds: R=1 78.1 (74.2), R=3 81.8 (77.2), R=10 84.8 (79.9), **R=30 87.1 (81.1)**; reference 1000 distinct worlds, one episode each: 83.5 (77.6).
  Paired (late): 100 worlds R3-R1 +5.9 (0.2), R10-R3 +5.1 (1.2), R30-R10 +0.9 (0.7); 300 worlds R3-R1 +3.7, R10-R3 +3.0 (0.3), **R30-R10 +2.3 (0.7)**; **300 worlds x30 minus 1000 distinct worlds +3.6 (1.0)**.
  **G1 true** (R30-R10 = +2.3 >= 0.5 and < R10-R3 = +3.0: diminishing returns; at 100 worlds saturation already at R10-R30), **G2 true** (300 worlds re-masked 30 times beat 1000 distinct worlds by 3.6; even 100 worlds x30 (84.0) match 1000 distinct worlds (83.5)), **G3 true** (R10 over R1: +11.0 at 100 worlds, +6.7 at 300).
  Reading: for this learner, the information that matters is the variety of (hidden subset, asked claims) given a world, not the number of worlds; 100 worlds re-masked 30 times are worth about 1000 distinct worlds. In the human study this means every world a person plays can be turned into many training episodes by re-masking (a direct way around the small-human-data limit), as long as the world's full premise set and verdicts are known. Next: does randomising the visible fraction across draws also give robustness to a shifted fraction (Q19), and does capacity matter once data is plentiful (Q20)?

- **Q19: randomising the visible fraction across re-masked draws** (`experiments/logic_remask_vis.py`, `2026-10-17_213910_logic_remask_vis`, 3 seeds, 300 worlds x 10 draws, plugged in). Accuracy at claims 7-10 for training with a fixed 50% visible / draws cycling through 30-70% visible, tested at 30% / 50% / 70% visible: fixed 82.1 / 85.4 / 90.5; mixed 81.8 / 84.7 / 90.1; mixed minus fixed -0.3 (0.5) / -0.7 (0.5) / -0.4 (0.2).
  **H1 false** (predicted >= +3 on the shifted tests), **H2 true** (-0.7 at 50%). Null: the learner trained at 50% visible is already as good at 30% and 70% visible as a learner trained on the whole range, so domain randomisation of the visible fraction adds nothing. (Accuracy follows how much is visible, 82 -> 85 -> 90, not the training fraction.)

- **Q20: capacity once data is plentiful** (`experiments/logic_remask_capacity.py`, `2026-10-17_215356_logic_remask_capacity`, 3 seeds, 300 worlds x 10 draws). Accuracy at claims 7-10 / overall: width 64 + 4 layers 84.9 / 79.7; width 128 + 4 layers 85.0 / 79.9; width 64 + 6 layers 84.7 / 79.7; width 128 + 6 layers 84.8 / 79.6. Paired: width 128 minus 64 = +0.1 (0.1) late; 6 layers minus 4 = -0.2 (0.5).
  **K1 false** (predicted >= +1), **K2 true**. Even with 12000 episodes the learner is not capacity-limited; the default 64 x 4 is enough. The remaining error is not a model-size problem: at claims 7-10 the learner is 99% right on claims that are provable from what it has seen and about 76% on the rest, where the verdict can only be guessed from the prior over worlds (Q0). The ceiling is therefore informational, not architectural.

- **Q21: re-masking x noisy teacher** (`experiments/logic_remask_noise.py`, `2026-10-17_222401_logic_remask_noise`, 3 seeds, 300 worlds x 10 draws, 20% of the shown verdicts wrong, noise redrawn in every draw; accuracy at claims 7-10 on episodes with noisy feedback). static (no feedback) 68.1; clean-feedback learner on clean test 85.3 (gain +17.2); **clean-trained on noisy test 76.1 (gain +7.9, 46% of the clean gain)**; noisy feedback with ONE draw 71.3 (+3.2, 19%); **noisy feedback with 10 redrawn draws 78.2 (+10.0, 58%)**; noisy_R10 minus clean_R10 on the noisy test +2.1 (0.6).
  **J1 false** (predicted >= 70% of the clean gain; 58%, though above the 55% falsifier line), **J2 false** (+2.1 < 3). Re-masking helps a noisy teacher a lot in absolute terms (noisy gain +3.2 -> +10.0, i.e. free data again), but it does not make the learner robust: 20% wrong verdicts still cost 42% of the gain, and discounting wrong edges is only slightly better than not (+2.1). The remaining fragility needs a mechanism that checks verdicts, not only more diverse data (Q22).

- **Q22: leave-one-out verdict check** (`experiments/logic_verdict_filter.py`, `2026-10-17_223920_logic_verdict_filter`, 3 seeds, 300 worlds x 10 draws, 20% wrong verdicts; each revealed verdict s is re-judged from the visible premises and the other earlier verdicts; the edge is dropped at time t if the model gives the shown verdict probability < tau; the learner never uses true labels). Accuracy at claims 7-10 on noisy test episodes:
  clean-trained model unfiltered 76.3 -> tau 0.02 / 0.05 / 0.1 / 0.2: 78.2 / **78.6** / 78.3 / 77.2; noise-trained model unfiltered 78.4 -> 78.5 / 78.9 / 78.9 / 78.1. Detection AUROC of wrong verdicts: **0.90 (clean-trained), 0.91 (noise-trained)**. Gain of filtering with tau picked on validation episodes (0.05 and 0.1): clean-trained **+2.3 (0.4)**, noise-trained +0.6.
  **V1 true** (AUROC >= 0.75), **V2 true** (+2.3 >= 2), **V3 true** (+0.6 < +2.3). A very good detector (AUROC 0.90) but a small repair: the filtered learner reaches 78.6 against 85.3 with clean feedback, i.e. about a quarter of the 9-point loss. Reason (consistent with Q12): the verdicts that look wrong to the learner are also the ones that carry hidden structure, so dropping them throws away information; a surprising verdict is ambiguous between 'error' and 'discovery'. Q23 tests keeping the edge but telling the model it is suspect.

- **Q23: suspect flag instead of deletion** (`experiments/logic_suspect_flag.py`, `2026-10-17_230159_logic_suspect_flag`, 3 seeds, 20% wrong verdicts, 300 worlds x 5 draws for the scorer M0 and, on different worlds, for M1; flags use only M0's leave-one-out probability < 0.05). Accuracy at claims 7-10 on noisy test episodes: plain 76.8; deletion at test time only 77.8; **deletion at training and test 78.5**; **suspect flag 77.4**; clean feedback (ceiling) 85.2.
  flag minus hard filter -0.3 (0.3); flag minus plain +0.6 (0.4); flag minus deletion-both -1.0 (0.3); clean minus flag +7.7. **S1 false, S2 false.** Keeping the edge and flagging it is no better than deleting it, and no deletion/flag scheme recovers more than about 2 of the ~8.4 points that 20% wrong verdicts cost. Together with Q11, Q11b, Q21 and Q22: a noisy teacher is robustly harmful; more data (+), re-masking (+), a good detector (AUROC 0.9) and deletion (+1.7) each recover a part, nothing recovers it.
  Practical consequence: the feedback source must be reliable (the game's critic is exact); the one remaining untested remedy is redundancy (asking the same claim twice), which costs budget.

- **Q24: the pipeline with re-masked training** (`experiments/logic_pipeline.py --remask 5 --encoders original`, `2026-10-17_232722_logic_pipeline_remask5`, 3 seeds, 600 worlds per level re-masked 5 times, not extended; ask 6 of 12 by uncertainty, learn from verdicts, judge 8 held-out claims, abstain with a gate). Against Q4 (one episode per world, original encoder):
  **in-distribution verdict accuracy with entropy asks 86.5 -> 90.1% (+3.6)**, random asks 81.5 -> 84.6 (+3.1); luck-adjusted reward (lam = 4, abstain -0.2): answer-all -0.187 -> -0.042; **confidence gate +0.206 -> +0.259; head gate +0.206 -> +0.223**; oracle gate +0.208 -> +0.222. 8-kind chains (not trained on): 59.4 -> 59.0% (no change); 10-kind trees 74.9 -> 77.0 (+2.1).
  **(a) true** (+3.6 >= +2), **(b) true** (OOD gains 0 and +2.1 < +3.6), **(c) FALSE:** head-minus-confidence gap in distribution is now -0.036 (it was 0.000): with more varied training the verdict model's own confidence becomes the better gate (+0.259 vs +0.223) and exceeds the 'oracle' gate (which is not an upper bound under the luck-adjusted reward). The head keeps its advantage only out of range (8-kind chains: head +0.050, confidence +0.032, oracle +0.081).
  Updated guidance: with plentiful (re-masked) training data use the verdict model's confidence as the in-distribution gate; keep the proof-availability head for shifted or larger worlds (Q3, Q3b, Q9c). Re-masking does not teach size generalisation (8-kind chains unchanged); training on the target size does (Q4b); Q25 combines both.

- **Q25: pipeline with extended AND re-masked training** (`logic_pipeline.py --extended --remask 5 --encoders original`, `2026-10-18_000850_logic_pipeline_extended_remask5`, 3 seeds; 8- and 10-kind worlds added to the training set and everything re-masked 5 times). Against Q4b (extended, one episode per world) with entropy asks: in-distribution accuracy **87.2 -> 89.4 (+2.2)**; **8-kind chains 78.5 -> 81.3 (+2.8)**; 10-kind trees 76.7 -> 77.0 (+0.3). Luck-adjusted reward (lam = 4): in-dist confidence / head / oracle gate +0.254 / +0.224 / +0.226 (Q4b +0.235 / +0.218 / +0.220); 8-kind chains +0.193 / +0.178 / +0.188 (Q4b +0.146 / +0.163 / +0.177); 10-kind trees -0.014 / -0.021 / -0.012.
  (a) in-dist >= 90%: **false, narrowly (89.4)**; (b) 8-kind chains >= 81 true (81.3), 10-kind trees >= 79 **false (77.0)**; (c) head not worse than confidence on chains: **false** (-0.015). So re-masking adds about +2..+3 points where the model has seen the world size; 10-kind trees do not move (75-77% in Q4, Q4b, Q24, Q25), which looks like an informational ceiling (most claims in a 10-kind world with 50% of the premises visible are unprovable from what can be asked).
  Best pipeline so far (entropy asks, extended + re-masked training, confidence gate): in-dist 89.4% accuracy, luck-adjusted reward +0.254 (lam = 4); 8-kind chains 81.3%, +0.193; 10-kind trees 77.0%, -0.014 (abstaining is about break-even there).

- **Q26: test-time training within an episode** (`experiments/logic_test_time_training.py`, `2026-10-18_003942_logic_test_time_training`, 3 seeds, 100 test episodes, judge from 300 worlds x 10 draws; at time t >= 2 three Adam steps on the leave-one-out examples of the t revealed verdicts, accumulated over t, then judge claim t). Accuracy at claims 7-10: frozen judge 85.0; lr 3e-4 82.4 (-2.5 (0.7)); lr 1e-3 77.9 (-7.1); lr 3e-3 68.1 (-16.8); shuffled targets at the validation-best lr 80.8 (-4.2 (1.0)).
  **T1 false** (predicted >= +1.5; every lr hurts, monotonically with lr), **T2 true** (shuffled targets hurt too, 1.7 points more than the true targets: the true verdicts carry a little information, but the update costs more than it gives). Reading: the in-context graph edges already deliver what the revealed verdicts can teach; adapting the weights on a handful of examples mostly damages a model trained on 12000 episodes. A smaller-lr check is Q26b.
  **Q26b** (`--lrs 1e-5,3e-5,1e-4`, `2026-10-18_005347_logic_test_time_training`, 3 seeds): frozen 84.9; lr 1e-5 84.9 (-0.1 (0.3)); 3e-5 84.4 (-0.5 (0.4)); 1e-4 84.0 (-0.9 (0.8)); shuffled targets at the validation-best lr 84.9 (+0.0). The pre-registered prediction (no lr beats frozen by more than the seed sd; monotone trend towards frozen) held: **there is no positive regime for test-time training here.**

- **Q27: rule-preserving augmentation on ARC** (`experiments/harc_augmentation.py`, `2026-10-18_011014_harc_augmentation`, 3 splits of H-ARC with the same seeds 0-2 as harc_h1_stml_v2, pure STML learner d = 128, 15 epochs, 3 augmented copies per training puzzle: one dihedral transform + one colour permutation applied to every grid of the puzzle, candidates included; test puzzles untouched). Solved in 3 attempts on unseen puzzles (right first in brackets):
  imitate human learning, plain 51.4% (22.1); same trained 4x longer 50.3% (19.6); **with augmentation 57.3% (24.8)**; imitate answers, plain 53.1% (24.4); **with augmentation 59.3% (28.0)**. Paired: H_aug minus H_base **+5.9 (1.7)**; H_base_x4 minus H_base -1.1 (1.5); H_aug minus H_base_x4 +7.0 (1.4); A_aug minus A_base +6.2 (2.0); **H_aug minus A_aug -2.0 (3.5)**; H_base minus A_base -1.8 (3.9).
  **A1 true** (>= +3), **A2 true** (more steps alone: -1.1), **A3 true at the boundary** (-2.0, within the +-2 band and with sd 3.5: human histories are NOT better than answers, with or without augmentation). The finding from logic transfers: diversity of the training instances, not gradient steps, removes a good part of the overfitting of the pure learner (+5.9..+6.2 points). It does not reach the 12-hand-feature linear baseline (65.8-67.1%), so perception is still the bottleneck. H1 stays unsupported on ARC.

- **Q28: ARC, RL with augmentation, and the augmented humans-then-RL pipeline** (`experiments/harc_augmentation_rl.py`, `2026-10-18_022436_harc_augmentation_rl`, 3 splits, pure learner, RL 6000 episodes with 2 restarts chosen on the ORIGINAL training puzzles, same RL step budget with and without augmentation). Solved in 3 attempts on unseen puzzles (right first): RL from scratch 54.1% (22.8); **RL from scratch with augmentation 51.9% (22.8)**; **STML pipeline with augmentation (imitate augmented human histories, then RL on training + augmented puzzles) 60.4% (27.7)**.
  Paired: R_aug minus R_base **-2.1 (1.4)**; S_aug minus R_aug **+8.5 (2.3)**; S_aug minus R_base +6.3 (0.9). Training-puzzle solve rates of the RL restarts: 66.9-68.7% plain vs 56.5-58.5% with augmentation (less overfitting) but no gain on unseen puzzles: the same number of RL episodes spread over 4x as many puzzles is too little learning per puzzle.
  **B1 false** (augmentation does not help RL from scratch with this budget), **B2 false** (S_aug beats R_aug by 8.5). BUT the comparison that matters for H1 is missing: S_aug has the imitation phase on 4x the human steps, and imitation of ANSWERS with augmentation reached 59.3% by itself (Q27). The pipeline's 60.4% is only +1.1 over that and +3.1 over imitating augmented human histories alone (57.3%). The right control, answers-then-RL under the same augmentation, is Q29.

- **Q29: humans-then-RL vs answers-then-RL under equal augmentation** (`experiments/harc_augmentation_pipeline.py`, `2026-10-18_032702_harc_augmentation_pipeline`, 3 splits (seeds 0-2), pure learner, 3 augmented copies per training puzzle, imitation phase of equal length, then 6000 RL episodes on training + augmented puzzles). Solved in 3 attempts on unseen puzzles (right first): **imitate human histories, then RL 62.5% (31.2)**; **imitate answers, then RL 59.6% (27.2)**; RL from scratch (augmented) 54.0% (23.7). Per split (humans / answers / RL): 64.2 / 60.2 / 57.9; 64.0 / 60.9 / 52.0; about 59.3 / 57.7 / 52.1.
  Paired: **S_aug minus SA_aug +2.9 (1.0)** (+4.0, +3.1, +1.6: positive in all three splits); S_aug minus R_aug +8.5 (2.4); SA_aug minus R_aug +5.7 (2.7).
  **C1 FALSE** (predicted within +-2; the falsifier is met: +2.9 > 2 and > the split sd), **C2 true** (both beat RL from scratch by >= 4). This is the FIRST positive H1 signal on ARC: with identical data volume, augmentation and RL budget, an imitation phase on human learning histories (with their wrong attempts and revisions) gives a better starting point for RL than an imitation phase on answers, by about 3 points and 2 points more right-first. Earlier null results (Q27 without RL: humans minus answers -2.0; the v2 runs) did not combine human histories with augmentation and RL. CAVEATS: 3 splits, an effect of 1.6-4.0 points against run-to-run variation of about 3 points in single runs (the same condition scored 61.4 in Q28 and 64.2 in Q29 on split 1), and the mechanism is unknown: human histories contain failures and revisions, so the history channel of the transformer is trained, while answer imitation never shows it a failed attempt. Q30 replicates on fresh splits and adds the control 'answers with synthetic failures'.
  **Q30 (replication on fresh splits 3-5, `2026-10-18_042442_harc_augmentation_pipeline`, same protocol, plus 'answers with synthetic failures')**: humans-then-RL 61.4% (29.6 right first), answers-then-RL 59.3% (27.5), **answers with synthetic failed attempts, then RL 58.5% (28.9)**, RL from scratch (augmented) 53.3% (25.4). Per split humans / answers / synthetic failures / RL: 58.6 / 58.1 / 59.4 / 54.6; 61.9 / 60.9 / 58.9 / 52.5; 63.7 / 58.9 / 57.4 / about 52.9.
  Paired: S_aug minus SA_aug **+2.1 (1.9)** (+0.5, +1.0, +4.8); SF_aug minus SA_aug -0.8 (1.5); S_aug minus SF_aug +2.9 (2.9); S_aug minus R_aug +8.1 (2.9).
  **D1 true, marginally** (>= +2; falsifier <= +1 not met, but 2 of 3 fresh splits are within +1), **D2 false** (synthetic failures do not sit between answers and humans; they are slightly below answers): training the history channel on made-up failures does not reproduce the gain, so the advantage is not just 'having seen failed attempts'.
  **Pooled over the 6 splits (Q29 + Q30):** humans-then-RL minus answers-then-RL = +4.0, +3.1, +1.6, +0.5, +1.0, +4.8, mean **+2.5**, sd 1.6, standard error 0.65, so roughly +1 to +4 (95%); positive in 6 of 6 splits. CAVEATS: the splits are random 50/50 halves of the same 787 puzzles (not independent), single runs vary by about 3 points, and the advantage is small. Reading: H1 gets modest support on ARC for the first time, in a pipeline (augmentation + imitation + RL) that no earlier test used; it is NOT an answer to the original question (human vs RL-agent histories), which is Q31.

- **Q31: the original H1 comparison on ARC: human histories vs the RL agent's own learning history vs answers, each followed by RL, under augmentation** (`harc_augmentation_pipeline.py --seed0 6 --rl-history`, `2026-10-18_055520_harc_augmentation_pipeline`, fresh splits 6-8, pure learner, 3 augmented copies per training puzzle, imitation phases of equal length, then 6000 RL episodes). Solved in 3 attempts on unseen puzzles (right first): **imitate human histories then RL 64.5% (32.7)**; imitate answers then RL 59.1% (28.3); **imitate the RL agent's own history (Algorithm Distillation) then RL 59.9% (27.2)**; RL from scratch (augmented) 54.7% (22.6).
  Paired: **S_aug minus SRL_aug +4.6 (1.6)** (+4.8, +2.5, +6.4); **S_aug minus SA_aug +5.3 (0.7)** (+5.8, +4.3, +5.9); SRL_aug minus SA_aug +0.8 (0.9); S_aug minus R_aug +9.8 (2.6).
  **E1 true** (>= +2), **E2 true** (>= +1). POOLED humans-then-RL minus answers-then-RL over the 9 splits of Q29/Q30/Q31: +4.0, +3.1, +1.6, +0.5, +1.0, +4.8, +5.8, +4.3, +5.9: mean **+3.4**, sd 1.9, standard error 0.63 (95% about +2.1..+4.7), positive in 9 of 9 splits.
  **H1 receives its first clear support: on ARC, with equal data volume, equal augmentation and equal RL budget, imitating HUMAN learning histories (wrong attempts and revisions included) before RL beats imitating the RL agent's own learning history by 4.6 points and imitating answers by 3-5 points; the Algorithm Distillation control is no better than answers.** Q30's synthetic-failure control (answers with made-up wrong attempts: no gain) says it is not just exposure to failed attempts. Caveats: random 50/50 halves of the same 787 puzzles (splits not independent); single runs vary by about 3 points; the advantage appears only in the augmentation + imitation + RL pipeline (without RL, human histories were not better than answers, Q27); one task domain, one learner family, only the pure family tested (the hybrid family with hand features is untested); the human data are H-ARC's, collected in a different interface. Next: does it need the augmentation (Q32), and which part of the human histories carries it (Q33).

- **Q32: does the H1 effect need augmentation?** (`harc_augmentation_pipeline.py --k-aug 0 --rl-history --seeds 5`, `2026-10-18_072415_harc_augmentation_pipeline`; splits 0-4, the same four pipelines WITHOUT augmentation; the labels in the log still say '_aug'). Solved in 3 attempts on unseen puzzles: **humans-then-RL 56.4% (right first 24.6)**; answers-then-RL 53.4% (23.5); RL-agent-history-then-RL 53.2% (24.9); RL from scratch 53.8% (24.2).
  Paired: humans-then-RL minus answers-then-RL **+3.0 (4.1)** (+6.9, +3.6, -4.5, +6.6, +2.4: positive in 4 of 5 splits); minus RL-history-then-RL +3.2 (3.2); minus RL from scratch +2.7 (3.0); answers-then-RL minus RL from scratch -0.4 (3.1); RL-history-then-RL minus RL from scratch -0.6 (2.7).
  **F1 borderline false** (predicted a lead of at most +2.5 over answers without augmentation; +3.0, sd 4.1), **F2 true for the imitation pipelines** (56.4 / 53.4 / 53.2 vs 61-65 / 59 / 60 with augmentation, though on different splits), RL from scratch unchanged (53.8 vs 53-55).
  Reading: the human-history advantage does NOT need augmentation: +3.0 here, +3.4 pooled over 9 augmented splits. Augmentation makes it far more reliable (9/9 positive, sd 1.9 vs 4/5, sd 4.1) and lifts every imitation pipeline. Without augmentation, imitating ANSWERS or the RL agent's OWN history before RL is no better than RL from scratch (-0.4, -0.6); only human histories help (+2.7 over scratch). Pooled over all 14 splits humans-then-RL beats answers-then-RL in 13. Q33 asks which part of the human histories carries it.

- **Q33: which part of the human histories carries the effect?** (`harc_augmentation_pipeline.py --seed0 9 --parts`, `2026-10-18_082431_harc_augmentation_pipeline`, fresh splits 9-11, with augmentation, imitation phases of equal length, then RL). Solved in 3 attempts (right first): full human histories 62.2% (31.6); answers 58.9% (28.8); **first attempts only 60.7% (27.8)**; **revisions only (steps after a failure) 60.0% (30.9)**; **attempt histories shuffled among the steps of the same puzzle 42.5% (15.8)**.
  Paired: full minus answers +3.3 (4.2) (+9.1, +1.3, -0.5); revisions minus first attempts -0.7 (1.8); first attempts minus answers +1.8 (5.2); revisions minus answers +1.1 (3.7); **shuffled minus full -19.7 (3.0)**.
  **G1 false** (revisions are not better than first attempts), **G2 true but within noise** (+1.8, sd 5.2), **G3 true, very strongly** (-19.7). Reading: neither the first attempts nor the revisions alone carry the effect (each recovers a bit more than half of it: 60.7 and 60.0 against 62.2 full and 58.9 answers; differences of 1-2 points inside the split sd of 2-5); the full histories are best. Training on histories that do not match the choices is catastrophic (42.5%, 16-19 points below even answers and RL from scratch): the learner leans heavily on its history channel, so the match between what was tried and what was chosen is exactly what the imitation phase teaches, and a mismatched history poisons it. This control shows the history channel is used; it does not by itself prove that human revision behaviour is the active ingredient.
  POOLED humans-then-RL minus answers-then-RL over all 17 splits (12 with augmentation: +4.0, +3.1, +1.6, +0.5, +1.0, +4.8, +5.8, +4.3, +5.9, +9.1, +1.3, -0.5; 5 without: +6.9, +3.6, -4.5, +6.6, +2.4): positive in 15 of 17 splits; with augmentation mean +3.4 (sd 2.6, standard error 0.75), without +3.0 (sd 4.1). The effect is real on average, small to moderate, and unreliable in any single split.

- **Q34: goal 3 of the brief, the HYBRID learner (graph + the 12 hand features of baseline_v1) in the augmentation + imitation + RL pipelines** (`harc_augmentation_pipeline.py --family hybrid --rl-history`, `2026-10-18_102809_harc_augmentation_pipeline_hybrid`, splits 0-2; the report title still says 'pure'). Solved in 3 attempts on unseen puzzles (right first): humans-then-RL **65.3% (35.6)**; answers-then-RL 64.6% (33.1); RL-agent-history-then-RL 65.1% (34.0); RL from scratch (augmented) **65.7% (35.3)**. Paired: S_aug minus SA_aug +0.8 (0.9); minus R_aug -0.3 (1.8); minus SRL_aug +0.3 (2.5); SA_aug minus R_aug -1.1 (0.9).
  **H-a false** (humans-then-RL >= 68%: 65.3), **H-b false** (lead over answers >= +2: +0.8), **H-c true** (RL from scratch >= 62%: 65.7). With the hand features every pipeline lands at 65-66%, the level of the 12-feature linear baseline (65.8-67.1%): **a learner that can perceive the puzzles shows no benefit from human learning histories** (+0.8 over answers, -0.3 over RL from scratch, all inside the noise), and the imitation phase adds nothing over RL. Combined with Q31 (pure learner: +5.3 over answers, +9.8 over RL from scratch) the reading is: human histories help a WEAK perceiver by giving the RL stage a better starting point; once perception is supplied by hand-made features, they do not. This answers goal 3 of the brief for the ARC domain: H1 is supported only for a learner whose perception is the bottleneck. Whether a learner that perceives through learned features (not hand-made ones) keeps the advantage is open (self-supervised pretraining of the object-graph encoder would test it).

- **Q35: does re-masking rescue binary feedback?** (`experiments/logic_remask_feedback_type.py`, `2026-10-18_120452_logic_remask_feedback_type`, 3 seeds, 300 worlds per level x 1 or 10 draws, hidden-premise episodes; binary = whether the history-free judge's guess for the revealed claim was right). Accuracy at claims 7-10, 1 draw / 10 draws: static 65.2 / 68.6; **binary 68.1 / 76.6**; **verdict 78.2 / 84.4**.
  Paired: verdict minus binary **+10.1 (2.4) -> +7.8 (0.4)**; binary minus static **+2.9 (0.9) -> +8.0 (0.8)**; verdict minus static +13.0 -> +15.7. **Y1 narrowly false** (gap at 10 draws +7.8 vs the predicted >= +8; the falsifier, < +5, is not met), **Y2 false** (binary gains MORE from the extra diversity, +8.5, than the verdict, +6.2).
  Reading: the 'binary teaches a fifth of what the verdict teaches' result of Q8 was partly a data limit: with 10 re-masked draws the right/wrong signal delivers 51% of the verdict's gain (8.0 of 15.7) instead of 22% (2.9 of 13.0). The verdict still wins by 8-10 points, so the human-study contrast (diagnostic vs scalar feedback) keeps a machine prediction of roughly +8 points on late claims in a data-rich learner, +10 in a data-poor one; for human learners the value of binary feedback may likewise depend on how much they have learned about the domain.

- **Q36: learned perception (self-supervised masked-node pretraining of the object-graph encoder)** (`harc_augmentation_pipeline.py --pretrain`, `stml/model/pretrain.py`, `2026-10-18_122514_harc_augmentation_pipeline_pretrain`, pure learner, splits 0-2; encoder trained 6 epochs to reconstruct masked object attributes on the 26,944 grids of the split's training puzzles and their augmented copies, masked loss 0.41 -> 0.21; then used to initialise each pipeline). Solved in 3 attempts on unseen puzzles, with pretraining (Q29 without, same splits): humans-then-RL **62.0%** (62.5); answers-then-RL **58.3%** (59.6); RL from scratch (augmented) **52.2%** (54.0).
  Paired differences from pretraining: -0.5, -1.3, -1.8 points; humans-then-RL minus answers-then-RL **+3.7 (1.7)** (without pretraining +2.9); minus RL from scratch +9.8.
  **P-a false** (no pipeline gains >= 2; all are 0.5-1.8 lower), **P-b false** (the human-history lead does not shrink; +3.7). This particular self-supervised objective gives the encoder no perception that matters for choosing answers: the hand features of Q34 help because they encode puzzle-level invariants (the output-shape rule, palette, symmetry, density), not because the object attributes were hard to read. The human-history advantage of the pure learner is therefore not simply a weak-perception effect that any better encoding removes; it disappears with the hand features (Q34) but not with masked-attribute pretraining (Q36). An objective that learns puzzle-level regularities (e.g. predict the output grid's shape or palette from the demonstration pairs) is untested.

- **Q37: puzzle-level pretext** (`harc_augmentation_pipeline.py --pretrain-task`, `2026-10-18_132510_harc_augmentation_pipeline_pretask`, pure learner, splits 0-2; before the pipelines the encoder, the pair encoder and the task encoder are trained to predict an 8-number descriptor of the held-out OUTPUT (shape, palette size, density, background share, two symmetries, same shape as the input) from the OTHER demonstration pairs and the held-out input; loss 0.058 -> 0.022; only demonstration pairs, never test outputs or candidate labels). Solved in 3 attempts on unseen puzzles, with the pretext (Q29 without, same splits): humans-then-RL **62.4%** (62.5); answers-then-RL **61.9%** (59.6); RL from scratch (augmented) **58.5%** (54.0).
  Changes from the pretext: humans-then-RL -0.1, answers-then-RL **+2.3**, RL from scratch **+4.5**. Humans-then-RL minus answers-then-RL **+0.4 (1.9)** (+2.9 without the pretext); minus RL from scratch +3.8.
  **P-c true** (RL from scratch +4.5, answers-then-RL +2.3, both >= +2), **P-d true** (the human-history lead falls to +0.4). Per split, humans-then-RL minus answers-then-RL: -1.3, -0.5, +3.0.
  **Synthesis of the ARC H1 line (Q27-Q37):** imitating human learning histories before RL helps a pure learner by 3-5 points over answers and 5-10 over RL from scratch (Q29-Q33), but the advantage vanishes when the same kind of knowledge is supplied another way: the 12 hand features (Q34: +0.8) or a puzzle-level pretext that teaches the regularities of outputs (Q37: +0.4); masked-attribute pretraining, which does not teach them, leaves it intact (Q36: +3.7). So what the human histories supply in this pipeline is best described as a PRIOR ABOUT WHAT SOLUTIONS LOOK LIKE (and how a failed attempt relates to the next one: shuffled histories collapse, Q33), not an advantage that survives once the learner has a good solution prior. For the project's thesis this is a qualified support: process imitation is a valid way to obtain that prior, but on ARC it is not necessary once the prior comes from elsewhere, and its additional value over a strong prior is not detectable here.

- **Q38 (part 1, 25% of the human steps): how much human data does the imitation phase need?** (`harc_augmentation_pipeline.py --human-frac 0.25`, `2026-10-18_142434_harc_augmentation_pipeline_frac0.25`, pure learner, splits 0-2, with augmentation; the answers arm uses the same number of steps as the human arm; RL phase unchanged). Solved in 3 attempts on unseen puzzles: humans-then-RL **59.6%** (right first 28.7); answers-then-RL **59.5%** (28.4); RL from scratch (augmented) 53.0% (24.7). Reference with ALL human steps (Q29): 62.5 / 59.6 / 54.0.
  Paired: S_aug minus SA_aug **+0.1 (4.7)** (per split +1.0, +5.4, -6.1; with all human steps +4.0, +3.1, +1.6); S_aug minus R_aug +6.5 (3.3); SA_aug minus R_aug +6.4 (3.4).
  **Q1 false** (predicted a lead >= +1.5 with 25% of the data): the human-history lead needs the volume of human steps (+2.9 with all of them, +0.1 with a quarter, erratic). **Q2 first half:** an imitation phase on a quarter of the steps already gives about +6.4 over RL from scratch whatever its source (humans or answers). So most of the benefit of the imitation phase over RL from scratch is cheap (any solution-bearing data); the extra benefit of HUMAN histories over answers is the part that needs lots of human steps. The 10% run follows.
  **Q38 part 2 (10% of the human steps), COMPLETED on a rerun** (`2026-10-18_203339_harc_augmentation_pipeline_frac0.1`, same splits 0-2): humans-then-RL **58.0%** (right first 28.4); answers-then-RL **59.1%** (27.5); RL from scratch (augmented) 53.2% (24.3). Humans-then-RL minus answers-then-RL **-1.1 (1.5)**; minus RL from scratch **+4.8 (2.6)**; answers-then-RL minus RL from scratch +5.9 (3.4). **Q2 true** (lead over answers <= +1.5, and humans-then-RL >= 3 above RL from scratch).
  **Data-fraction summary (pure learner, with augmentation, splits 0-2; human steps used in the imitation phase):** 100% (Q29): humans minus answers +2.9, humans minus RL +8.5, answers minus RL +5.7; 25%: +0.1, +6.5, +6.4; 10%: -1.1, +4.8, +5.9. The imitation phase's benefit over RL from scratch is about +6 whatever its source and whatever the amount (10% to 100% of the steps), while the extra benefit of HUMAN histories over answers appears only with the full volume (about 6,500 steps here). For a human study this says: a few hundred steps of any solution data already help; the human-process-specific gain needs thousands.
  (An earlier attempt at the 10% run, `2026-10-18_151741_harc_augmentation_pipeline_frac0.1`, was STOPPED unfinished (`2026-10-18_151741_harc_augmentation_pipeline_frac0.1`, kept as is): the Mac was on battery in Low Power Mode and the run advanced only ~30 s of CPU in 30 minutes. It completed only the first two conditions of split 0: humans-then-RL 56.9%, answers-then-RL 58.4% (lead -1.5, one split, no conclusion). The second half of Q2 (humans-then-RL stays >= 3 above RL from scratch with 10% of the steps) is UNTESTED; rerun when the machine is plugged in.

- **Q39: lemma precision via ensemble agreement** (`experiments/logic_lemma_ensemble.py`, `2026-10-18_221401_logic_lemma_ensemble`, 3 seeds, 3 independently trained lemma judges per seed, a lemma accepted only if all three give P(follows) > tau, 5 rounds, 12 fresh test worlds per cell). Accuracy after 5 rounds, plain judge A / single-model lemmas / ensemble tau 0.9 / ensemble tau 0.97 / true-lemma oracle (30%): chain-24 78.8 / 81.2 / 82.8 / 82.1 / 85.3; **chain-32 74.9 / 74.9 / 74.5 / 74.3 / 82.4**; tree-24 95.2 / 93.8 / 94.3 / 93.5 / 95.1; tree-32 93.3 / 92.2 / 90.7 / 89.9 / 93.7.
  Lemma precision, single / ensemble 0.9 / ensemble 0.97: chain-24 80 / 89 / 91%; **chain-32 73 / 82 / 85%**; tree-24 83 / 94 / 96%; tree-32 80 / 94 / 96%. Ensemble 0.9 minus plain judge: chain-24 **+4.0 (3.2)**, chain-32 **-0.4 (4.4)**, tree-24 -0.9, tree-32 -2.5; ensemble minus single: +1.6 (2.6), -0.4 (0.9), +0.5, -1.5.
  **M1 true** (precision +9..+12 points at 32-kind chains), **M2 false** (gain at 32 kinds -0.4, not >= +3; at 24 kinds +4.0, not >= +5; the falsifier is met: the ensemble gain is within the seed sd of the single-model gain).
  Reading: making the self-written lemmas much more precise (73 -> 85%) did NOT make them more useful at 32 kinds, while true lemmas still give +7.5. What the oracle supplies is hard lemmas (deep, non-obvious conclusions); what a judge is confident about is easy ones (shallow, already within its reach), so self-bootstrapping adds little new at the sizes where the judge is weak. The scratchpad idea is closed: +2..+5 at 12-24 kinds, none at 32, no cure through verification by agreement. Lemma training also costs 1-3 points on trees.

- **Q40: more human data, ARCTraj added** (`harc_augmentation_pipeline.py --arctraj`, `2026-10-18_232448_harc_augmentation_pipeline_arctraj`, pure learner, splits 0-2, augmentation; the ARCTraj first-attempt steps of the training puzzles (e.g. 3,822 on split 0) added to the 6,504 H-ARC steps; the answers arm has the same total number of steps). Solved in 3 attempts on unseen puzzles (right first): humans-then-RL **61.1%** (28.9); answers-then-RL 57.8% (26.0); RL from scratch (augmented) 53.2% (23.4). Reference Q29 (H-ARC only): 62.5 / 59.6 / 54.0.
  Humans-then-RL minus answers-then-RL **+3.3 (6.1)** (per split +11.1, -3.6, +2.3); minus RL from scratch +7.9 (3.9). **R1 false** (a lead >= +4 was predicted; +3.3 with a spread of 6.1), **R2 false** (humans-then-RL 61.1 vs 62.5 in Q29: -1.4). More human steps from a second dataset do not raise the human-history advantage: ARCTraj contributes first attempts only (no failed attempt -> revision sequences) and comes from a different interface, consistent with Q33 (first attempts alone carry only part of the effect) and Q38 (the extra human benefit needs the volume of H-ARC-style histories).

**Meta-analysis of the programme (2026-10-20):** `docs/META_ANALYSIS.md` (narrative: the five findings worth keeping, what did not work, noise and prediction hit rate), `docs/META_ANALYSIS_numbers.md` and `docs/figures/forest_logic.png`, `docs/figures/forest_arc.png` (made by `scripts/meta_analysis.py`).

- **Pilot-01 readout (2026-10-20; the owner's own run, worlds L1-L3 finished, L4-1 in progress; usability pilot, NOT data: one person who knows the hypotheses)** (`logs/pilot-01/`, 10 files, `python scripts/analyze_log.py logs/pilot-01`, `python scripts/analyze_game.py logs/` -> `runs/2026-10-20_201947_game_analysis`). The log pipeline works end to end (stml-log/1, one file per world). 72 proposed claims (36 targets, 17 contradicted, 10 weak, 5 undetermined, 4 premises, 13 repeats), 36 test answers.
  Test rounds: 61% correct (22/36); by truth: 'follows' 9/12, 'contradicted' 5/6, **'open' 8/18 (the participant said 'must be false' for 9 of the 18 open claims, 'can't tell' 8 times, 'I don't know yet' once)**. 'I don't know yet' was used 1 time in 36 (3%; the synthetic population assumed about 20%). Confidence as a correctness signal: 62% correct at 3, 58% at 4, 75% at 5 (n = 8 / 19 / 8; AUROC 0.55, simulation assumed 0.64); when PROPOSING claims confidence tracked success better (share of targets: 30% at 3, 69% at 4, 100% at 5).
  Errors (analyze_log): contradiction 17, overgeneralisation 12, illicit process 9, undistributed middle 7, some-chaining 3, two negatives 1, mostly in L3 worlds. Two L3 worlds were skipped (3/4 and 4/5 targets found). Timing: about 26 s per proposed claim and 28 s per test answer. The optional 'which premises did you use' marks and 'why' text were never used (0 of 72 claims).
  Readout for the study: the dominant human failure on 'open' claims is asserting 'false' where there is no certificate (the opposite of abstaining), the 'I don't know yet' option is nearly unused here, the optional marks / why fields yield no data unless prompted, and L3 is where people give up. All of this is one person and must be re-checked in the real sample.

- **Pilot sensitivity of the study sizing (Q41, 2026-10-20; N = 1, a sensitivity check NOT an estimate)** (`stml/synthetic.py: Population.naf_prob`, `experiments/sim_pilot_sensitivity.py`, `2026-10-20_202349_sim_pilot_sensitivity`; 3000 simulated participants per arm and scenario, first 9 worlds). The pilot judged 9 of 18 open claims 'must be false' (negation as failure) and used 'I don't know yet' on 3%; the population had neither. Scenarios (accuracy / 'don't know yet' share / accuracy on open claims / share of open claims judged false; pilot-01: 0.61 / 0.03 / 0.44 / 0.50):
  baseline 0.53 / 0.22 / 0.61 / 0.00; naf 0.5 0.38 / 0.23 / 0.31 / 0.30; naf 0.5 with tau 0.02 0.40 / 0.15 / 0.34 / 0.34. E3 (diagnostic vs scalar, gain 0.7): effect on the final test round +0.070 -> +0.054 / +0.053, completers per arm for 80% power **75 -> 100 / 100**.
  Predictions: **S1 half** (baseline judges no open claim false, true; naf 0.5 gives 30%, not ~50%, because the behaviour is applied only to claims that were not lured to 'true' or answered 'don't know'), **S2 true** (open-claim accuracy 0.31 < 0.45; overall accuracy falls 15 points, at the edge of the 5-15 band), **S3 false** (the needed completers change by +33%, above 25%), **S4 false** (tau 0.02 only reaches 15% 'don't know yet', not 3-8%: abstention also comes from weak evidence on lured claims).
  Reading: if people treat 'not provable' as 'false' as often as the pilot did, the sizing for E3 grows by about a third (75 -> 100 completers per arm for the +0.07 effect), because the behaviour adds noise that is not tied to the arm. Do not use it for planning until a sample (not one participant) shows how common it is; the parametrisation also does not reach the pilot's 50% yet.

### 7.9 What is still open

Human data (the game logs) for H1 proper; exploration-aware recalibration (done offline in Q13; not yet with a live learner);
the cognitive learner (Pillar 1); a proof/certificate check that does not rely on the learner's own network.

## 8. Goals, in order

1. Get the environment working on the Mac with Metal; all tests pass.
2. Convert H-ARC; run `baseline_v1` locally to reproduce section 7.1 with local logs.
3. Run `harc_h1_stml.py --quick`, then the full run. Compare with baseline_v1. The question:
   does a learner that can actually perceive the puzzles benefit from human learning histories?
4. Download and convert ARCTraj; add it to Stage A pretraining (cross-dataset). **Done** (7.3; gain not
   significant).
5. Add cross-task context (the transformer sees earlier puzzles of the same person) to test
   learning to learn. **Done**, null result (7.6).
6. Move the main H1/H2/H3 tests to the logic domain, where hypotheses are represented exactly (concept
   graphs) and the human data comes from Wugs and Daxes. **H2 and H3 done on synthetic learners (7.4, 7.5);
   H1 waits for the human game data.**
7. Build the cognitive learner (Pillar 1) for synthetic human-like trajectories.
8. Human study: IRB at UCLA, consent screen in the game, 30–50 then 200–300 participants.

## 9. Code map

```
stml/logic.py, proof.py      exact semantics; named-rule proofs (sound + complete)
stml/world.py                hidden worlds and the 5-level curriculum
stml/critic.py               Socratic critic
stml/referee.py              SAT referee (never used by the learner)
stml/metacog.py              four epistemic answers, honesty reward, calibration metrics, probes
stml/env.py                  agent environment (propose claims; judge with four answers)
stml/analysis.py             classify human reasoning errors
stml/episodes.py             the 15 game worlds; trajectories.py: game logs → trajectories
stml/datasets/               stml-traj/1 format; H-ARC and game converters
stml/model/                  THE STML LEARNER: graphs, network, H-ARC data, training
  harc_data, arctraj_data    ARC puzzles/pools/steps; ARCTraj steps + progress grids (training puzzles only)
  hand_features              the 12 hand features of baseline_v1 as an optional input
  train, train_seq           Stage A/B per puzzle; sequences across puzzles (learning to learn)
  logic_data, logic_judge    logic worlds as judging problems; judge, why head, certificate checks
stml/runlog.py               run folders: config, environment, log, metrics, results, report
experiments/harc_h1_stml.py  ARC, pure STML learner (7.2); harc_h1_stml_v2.py: + ARCTraj, pure/hybrid (7.3);
                             dev_stml_v2.py: dev ablation; harc_learning_to_learn.py (7.6)
experiments/logic_*.py       logic domain: h3_judge, h3_scaling, h3_certificate, h3_thresholds, h2_feedback
scripts/compare_runs.py      side-by-side tables of finished runs -> runs/COMPARISON.md
stml/synthetic.py            synthetic human-like participants for the game; experiments/sim_study_power.py sizes the human study
docs/FINAL_ARCHITECTURE_DRAFT.md  evidence-based draft of the final architecture (what is supported / not supported / what the human logs must decide); revise after the game logs
docs/ARCHITECTURE_V2.md      design proposal: literature -> components and pretraining stages (not implemented unless marked)
literature/                  papers.json (source of truth), INDEX.md, SUMMARY.md, pdfs/<key>/<key>.pdf; scripts/fetch_literature.py
experiments/baseline_v1/     first experiment (linear choice model), kept as reference
game/                        Wugs and Daxes
scripts/                     setup_mac.sh, build_game, play_local, analyze_log,
                             convert_datasets, download_arctraj
tests/                       core logic, datasets, model
data/raw, data/processed     datasets (not code); logs/ game logs; runs/ experiment runs
```

## 10. How to run (macOS, Apple Silicon, Metal)

```bash
cd ~/Research_Projects/STML
bash scripts/setup_mac.sh                 # creates .venv, installs requirements, checks Metal
source .venv/bin/activate
python -m pytest -q                       # all tests must pass
python scripts/convert_datasets.py harc --data data/raw/harc/data --arc data/raw/harc/arc_data
python experiments/baseline_v1/run.py --seeds 5          # ≈8 min, reproduces 7.1
python experiments/harc_h1_stml.py --quick               # smoke test
python experiments/harc_h1_stml.py --seeds 5             # the real run
python scripts/download_arctraj.py                       # ARCTraj from Hugging Face
python scripts/convert_datasets.py arctraj --arc data/raw/harc/arc_data
python experiments/harc_h1_stml_v2.py --family pure      # ~1.6 h each; run pure and hybrid in parallel
python experiments/logic_h3_certificate.py --seeds 5     # ~16 min; also logic_h3_thresholds, logic_h2_feedback
python scripts/compare_runs.py                           # tables for all finished runs
python scripts/play_local.py                             # play the game locally, logs to logs/
```

If an operation is not implemented on Metal, run with `PYTORCH_ENABLE_MPS_FALLBACK=1`, or pass
`--device cpu` to compare.

## 11. Rules for Claude Code in this folder

- Run everything here, on this Mac. Use Metal (`mps`) for training.
- Every experiment goes through `stml.runlog.Run`: a new folder in `runs/`, never overwrite.
  Write a short `report.md` for every run that matters.
- Never delete or modify anything in `data/raw/` or `logs/`; they are original data.
- Inspect a dataset's real fields before writing a converter. Validate converted data against
  the paper's published numbers when possible.
- Be honest about results: same model and data budget across conditions; fair baselines (RL
  restarts chosen on training data only); proper scores (log-likelihood, Brier); report CIs and
  per-split numbers; a null result is a result. Check suspicious numbers before believing them.
- Keep the referee away from the learner.
- No git for now.

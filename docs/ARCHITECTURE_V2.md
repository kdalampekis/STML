# STML-2: turning the neuroscience into architecture and pretraining

Status: **design proposal, not implemented**, except where a row says "tested" or "built". Written 2026-10-15 after the
literature review (`literature/SUMMARY.md`). Every component names the failure in our own results that it targets, and a
prediction that would falsify it. Where a claim rests on my reading of an abstract or on general knowledge rather than
on the paper's text, it is marked *(unverified)*.

## 0. Where we are, in one paragraph

On ARC (real human data) the graph learner barely beats chance on its own (50-55% solved vs 46% random, 22-25% right
first try), the 12 hand features of the first experiment do better, H1 is reversed, learning to learn is null. On logic
(exact structure, unlimited data) the same encoder learns exact reasoning (99-100%), is overconfident when the problems get
bigger than its training range, and a *second-order* self-check plus a shift-aware threshold fixes much of that. The
literature explains the split: learning is fast when new information fits an existing structure and slow when it does
not (Kumaran et al. 2016, text read), and "learning to learn" needs many related tasks (Wang et al. 2018, text read).

## 1. Components

| # | Component | Brain evidence | Our failure it targets | Falsifiable prediction |
|---|---|---|---|---|
| C1 | **Surprise channel**: the learner predicts its own correctness before feedback; the signed error delta = outcome - p enters the model and the loss | Dopamine = reward prediction error (Schultz 1997); third factor in plasticity (Gerstner 2018); separate gain for better-than-expected vs worse (Dabney 2020, Frank 2004) | Stage A copies choices and ignores how wrong the expectation was; calibration is poor (ECE up to 0.77) | Surprise-weighted imitation lowers ECE and the revision log-likelihood gap vs plain imitation, at equal data |
| C2 | **Fast episodic store + prioritized, interleaved replay** next to the slow graph learner | Complementary learning systems; replay biased to surprising/informative events; replay order by expected utility (Kumaran 2016, text; Mattar & Daw 2018) | Overfitting on 400 puzzles; mistakes are "inconsistent" with an empty schema and are learned slowly | Prioritized replay of high-surprise episodes beats uniform replay at equal compute |
| C3 | **Structure/content split** in the encoder: relation-transition weights shared across tasks (structure), per-task content in a fast memory | Tolman-Eichenbaum Machine: structure learned slowly and reused, content bound in one visit (Whittington 2020) | ARC: no prior structure, so each puzzle is learned from scratch; logic: it worked because structure was built in | A model pretrained on structure in domain A needs fewer examples in domain B than one trained from scratch |
| C4 | **Meta-learning over a task family** (in-context learner) | Prefrontal activations implement a second, fast learner trained by slow dopamine RL across *related* tasks (Wang 2018, text) | Learning-to-learn null: five unrelated puzzles, 6,000 episodes | Gain across positions appears when tasks share structure (logic worlds) and not when they do not (random ARC) |
| C5 | **Second-order metacognition**: confidence computed from a re-evaluation of the learner's own evidence, plus an explicit abstain action | Confidence = P(choice correct) is a separate computation (Pouget 2016, text; Kepecs 2008) | Softmax confidence collapsed under shift; sufficiency check ranked errors better (tested, 5/5 seeds) | Training the confidence head on the re-judgment features beats training on the first-order state, under shift |
| C6 | **Hypercorrection weighting**: up-weight steps where the learner (or the person) was confident and wrong | High-confidence errors are corrected more (Metcalfe 2017, abstract only) | Wrong human answers are noisy teachers (imitating answers beat imitating human histories) | Weighting by confidence x error beats uniform weighting on human trajectories *that carry confidence* (the game) |
| C7 (partly tested) | **Hypothesis sampling and local edits** as the action space | People hold a few sampled hypotheses (Dasgupta 2017); learning = repeated small program edits (Rule 2020, partly read); H-ARC: after a mistake people keep the grid size 86% of the time and a median 93% of cells | We only model the choice among final answers, which discards the process that H1 is about | Edit-level imitation predicts people's next action better than frequency baselines, and the gain grows with trajectory length |
| C8 | Predictive-coding layer with local updates | Whittington & Bogacz 2017, Millidge 2020, Lillicrap 2020 | none directly | Deferred: no evidence it improves sample efficiency |

**Test result (2026-10-16, logic domain, `experiments/logic_surprise.py`):** weighting the loss by surprise (C1) or by confident error (C6)
gave no accuracy gain over plain training, only a modest, shift-specific calibration gain that the standard focal loss matches or beats; prioritised
replay by surprise hurt at small N. As a loss reweighting, C1/C6 is NOT supported. The remaining untested form is the signed surprise as an
INPUT to a sequential learner (the feedback token), and surprise-gated memory writes (C2), which are different mechanisms.
The signed-surprise-as-input form was then tested too (`experiments/logic_surprise_input.py`, hidden-premise logic episodes with revealed verdicts): the
feedback itself helps a lot (+9 to +12 accuracy points at late positions) but adding the signed surprise, as edge types or as node features, adds nothing beyond
the verdict, and a shuffled-flag control does as well. So C1 is not supported in either form in logic. C2 (surprise-GATED memory writes) is still untested.

### C1 in code (smallest first step)
- `STMLLearner.logits` already returns a confidence logit per candidate. Add `delta = outcome - sigmoid(conf)` as one more
  scalar added to the feedback token embedding in `state()`, so the transformer sees how surprising each feedback was.
- In `fit_imitation`, weight each step's CE by `w = 1 + beta * |delta|.detach()` (clipped). Report ECE and Brier.
- In `train_rl`, keep the reward-minus-baseline advantage (already RPE-shaped) and test separate scales for positive and
  negative advantages (an asymmetric learning rate, cf. Frank 2004).

### C2 in code
- Episodic store: keys = task token + candidate embedding, values = (outcome, delta). Retrieve the k nearest episodes by
  cosine and attend over them as extra tokens in `state()`. Write only when `|delta| > tau` or on errors.
- Replay: sample past episodes with probability proportional to `|delta|^alpha`, interleaved 1:1 with schema-consistent
  (low-surprise) episodes, which is the remedy Kumaran 2016 reports for learning inconsistent information without
  forgetting.

### C3 in code
- `RelationalGNN` already shares relation weights `W_r` across puzzles (structure). Make the split explicit: keep `W_r`
  slow, and add a per-task fast content memory that is reset per task (TEM-style). The pretext task (below) trains `W_r`.

### C5 in code (largely built)
- `stml/model/logic_judge.py: certificate_scores` (sufficiency) and `experiments/logic_h3_thresholds.py`. Next: feed
  `suff` and the why-head probabilities into the confidence head as inputs, and add `unknown` as a fourth policy action
  scored by the honesty reward.

### C7 in code
- New data path: `data.csv` (H-ARC actions with the full grid and distance-to-target `d`) and `ARCTraj.csv` (actions
  including Undo/Redo, which are error-correction events). Model: encode the current grid graph and the task token, predict
  the next (operation, position, colour). Baselines: repeat the last op; most common colour; frequency by position.

## 2. Pretraining curriculum (what to pretrain on, in which order)

The central principle from Kumaran 2016: **establish structure first, then learn the inconsistent and surprising material
by interleaving it**, not the other way round.

| Stage | Objective | Data | Targets which failure |
|---|---|---|---|
| S0 structure | Self-supervised relational prediction: (a) logic worlds: judge + why head (done, unlimited data); (b) **ARC leave-one-pair-out**: from the other demonstration pairs of a puzzle, predict the held-out output graph | All 800 ARC puzzles' demonstration pairs (inputs, not test answers, so also legitimate on test puzzles) plus synthetic logic worlds | No prior structure; perception. (b) is new and is exactly the missing skill: judging whether a candidate fits the rule shown by the demonstrations |
| S1 process imitation | Next-edit prediction with surprise weighting (C1) and hypercorrection weighting (C6) | H-ARC actions, ARCTraj actions incl. Undo/Redo, game logs when available | H1 on the actual process, not final answers |
| S2 interleaved replay | Prioritized replay of high-surprise episodes mixed with low-surprise ones (C2) | Episodes from S1 | Overfitting / catastrophic interference |
| S3 meta-RL | RL² on a *family* with shared structure (logic worlds via `stml/env.py`, the game's own loop), honest reward with an abstain action (C4, C5) | Generated worlds, unlimited | Learning-to-learn null; "I don't know yet" |
| S4 calibration | Shift-aware threshold from a few answered problems in the new setting | A handful of labelled problems per new domain | Abstention failing under shift (tested) |

Evaluation, always: unseen puzzles/worlds; same data budget across conditions; sample-efficiency curves; transfer to a second
domain (adaptation speed), not only in-domain accuracy.

## 3. How each stage maps to what we already measured

- *Extended training on bigger worlds gave +25 points on 24-kind chains*: that is S0/S3 widening the range of structure,
  consistent with "learning is prior-knowledge dependent".
- *Imitating answers beat imitating human histories on ARC solve rate*: noisy teacher; C1/C6 aim to extract the useful part
  (the surprising, confident errors) instead of copying every step. Untested.
- *Sufficiency self-check beat softmax confidence*: C5 makes this the confidence head's input rather than a post-hoc gate.

## 4. Cognitive architectures (gap search; bibliographic details verified, content from general knowledge *(unverified)*)

- **ACT-R** (Anderson et al. 2004): declarative memory with activation that decays and strengthens with use, procedural rules
  selected by learned utility, and a goal module. Learning curves are power-law. Relevant as a source of *quantitative*
  human learning-curve shapes for the synthetic learner and for expected effect sizes.
- **Soar** (Laird, Newell & Rosenbloom 1987): when the system cannot decide (an *impasse*), it creates a subgoal, resolves it, and
  stores the result as a new rule ("chunking"). Impasse-driven learning is the machine form of the Socratic question and of
  productive failure.
- **Standard Model of the Mind** (Laird, Lebiere & Rosenbloom 2017): a shared component list (working memory, long-term
  declarative and procedural memory, perception/motor, learning from experience). A checklist: STML has no working-memory
  buffer, no declarative/procedural split, and no impasse mechanism.
- Action: use ACT-R's learning-curve forms to set priors in the sizing simulation, and consider an explicit *impasse event*
  (the learner cannot choose with confidence) as the trigger for the fast store (C2).

## 5. Language learning (gap search) and the artificial-language game

What the sources say *(abstract-level, not read in full)*:
- Infants segment words from a continuous stream by transition statistics alone, after about two minutes, with no feedback
  or reward (Saffran, Aslin & Newport 1996). Grammar can be learned implicitly without stating the rules (Reber 1967).
- Early learning combines statistics with social interaction (Kuhl 2004, 2007); later exposure gives poorer outcomes
  (Newport 1990). Comprehension and production run on coupled streams (Hickok & Poeppel 2007); syntax and semantics use
  partly different networks (Friederici 2011). A connectionist model learns syntax from prediction error (Chang, Dell & Bock
  2006, *abstract not confirmed*).
- Duolingo data (Settles & Meeder 2016; SLAM 2018) models retention and word-level errors but has one attempt per
  exercise, so it does not contain the revision loop.

Implications for a language version of the game (if you build one):
1. **Two phases**: a feedback-free exposure phase (statistical learning) and a hypothesis-testing phase with feedback.
   Compare what people learn implicitly vs explicitly.
2. Log, for every response, **confidence before the answer** and a free-text **explanation**; randomise the feedback type
   (right/wrong vs diagnostic) so H2 can be tested between participants.
3. Use an **invented grammar** (the Wugs-and-Daxes logic again): no prior knowledge, exact truth from a referee, and the same
   graph/rule representation as the logic domain, so a model pretrained on logic structure can be evaluated on it.
4. Probe production as well as judgment (dual stream).

## 6. Risks and what not to claim
- Nothing here is shown to improve results yet. The literature suggests mechanisms; our data decide.
- Several brain findings come from rodents or small human samples; their transfer to a transformer-graph model is an
  analogy, not a derivation.
- C1/C6 need confidence labels that H-ARC does not have. They can only be tested on the game logs (pending) or on synthetic
  learners.
- The ARC leave-one-pair-out pretext is a bet: it trains rule inference from demonstrations, but ARC rules are diverse and
  800 puzzles are few. Treat a null as informative, not as a refutation of the idea.

## 7. Order of work (what is being started)
1. Study-sizing simulation (`experiments/sim_study_power.py`): built first because it needs no new model.
2. Process-level imitation on H-ARC (C7): data pipeline and baselines first.
3. Cross-domain transfer (C3, S0): pretrain on logic structure, adapt to a second domain, compare with human vs RL trajectories.
4. C1/C2/C5 as ablations on top of the existing learner.

## 8. Evidence update (2026-10-17): what the experiments said about each component

Run folders and details: CLAUDE.md section 7 (7.8 has Q-ids); the evidence summary for the final architecture is `docs/FINAL_ARCHITECTURE_DRAFT.md`.

| component | tests so far | verdict |
| --- | --- | --- |
| C1 surprise channel | surprise as loss weight (surprise, confident-error, replay): no gain over plain/focal; signed surprise as an input: nothing beyond the verdict, shuffled control equal | NOT supported in logic with exact labels; untested with human confidence |
| C2 fast episodic store + replay | prioritised replay by surprise hurt at small N (-7..-11); surprise-GATED store (Q6, Q6b): a third of the memory keeps two thirds of the gain, +2.2 over a capacity-matched random store, -4.8 vs keep-all; advantage gone with 20% wrong verdicts (Q12) | small and conditional: worth it as a capacity argument and only with trustworthy feedback |
| C3 structure/content split | hypernetwork core matches the per-relation encoder; ARC -> logic transfer null from every source (fine-tuning within 0.6 points) | encoder part supported, transfer not shown |
| C4 meta-learning over a task family | learning-to-learn on ARC null (small budget); Q10: the REINFORCE asking policy equals the entropy rule; full meta-RL (Q7) not run | not shown |
| C5 second-order metacognition | proof-availability head: AUROC 0.99-1.00 under shift where confidence is 0.74-0.81; reward advantage +0.05 +- 0.03 across trainings (Q9c); sufficiency check beats confidence in AUROC/AURC; abstention threshold repaired from committed answers only with hysteresis (Q13, Q13b) | supported (ranking and robustness); reward gain modest |
| C6 hypercorrection weighting | same as C1 | not supported |
| C7 hypothesis sampling / local edits | cell-level edit model: top-1 34-35% vs 3% adjacency, but the failed attempt adds nothing beyond the current grid | partly: the process is predictable, feedback from failure is not detectable |
| C8 predictive coding | not built | deferred |
| H1 itself (stage A on human learning histories) | ARC, pure learner, augmentation + imitation + RL: humans-then-RL beats RL-agent-history-then-RL by 4.6 and answers-then-RL by 5.3 (Q31; pooled 9 splits +3.4 over answers); null without RL and in all earlier runs | first support; the pipeline matters (Stage A then Stage B) |
| new: lemmas as scratchpad (Q15, Q15b) | +3.7 / +4.8 at 12 / 16 kinds, +4.1 at 24, +1.1 at 32 over the plain judge; false lemmas cost 13-20 points | small gain, limited by lemma precision |
| new: auxiliary world model (Q16) | predicting hidden premises as a second task: +0.3 / -0.3 late | null |
| new: uncertainty-driven questions | entropy asking +5 over random, a learned policy matches it, structure-only = random; survives 20% noisy verdicts (+3.5) | supported (the Socratic question = the learner's own uncertainty) |
| new: noisy teacher | 20% wrong verdicts remove ~60% of the feedback gain at 600 worlds, ~45% at 2000; the learner learns to discount only with more data | a real risk for any non-exact feedback source |

Pretraining stages S0-S4 (section 2): none of the data-source ablations (human vs RL vs answers histories) produced a benefit in logic; the one pretraining-like lever that worked is training on the target size range (Q4b, +19 points on 8-kind chains) and more worlds (Q5).

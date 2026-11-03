# How brains learn, and what it means for STML

Résumé of the literature in this folder (62 entries in `papers.json`, 27 open-access PDFs in `pdfs/`, the rest cited by
identifier). Written 2026-10-15 to answer one question: *what does the science of learning tell us about how to build a
machine that learns from its mistakes and knows what it does not know?*

**How much I read.** `[text]` = I read the relevant passages in the PDF. `[abstract]` = I only saw the abstract or a search
summary, so treat the claim as unconfirmed until you read the paper. Author lists, venues and DOIs that I could not
confirm are marked "not verified" in `INDEX.md`. Four papers could not be downloaded (publisher or rate-limit blocks).

---

## 1. Eight findings from the science, and what each means for STML

### F1. Learning is driven by *prediction error* (surprise), not by outcomes alone
- Dopamine neurons signal the gap between expected and received reward, the quantity temporal-difference learning
  uses (Schultz, Dayan & Montague 1997 `[abstract]`). In the anterior cingulate, a negative prediction error produces the
  error-related negativity, the brain's "that was wrong" alarm (Holroyd & Coles 2002 `[abstract]`).
- At the synapse, learning is a Hebbian trace gated by a "third factor", typically "reward minus expected reward" or the
  saliency of an unexpected event (Gerstner et al. 2018 `[abstract]`).
- Success and failure appear to be learned through partly separate pathways (Frank, Seeberger & O'Reilly 2004 `[abstract]`).
- The brain does not track one expected value but a *distribution* over outcomes (Dabney et al. 2020 `[abstract]`).

**For STML.** Our Stage A copies *what was chosen*. The brain learns from *how wrong the expectation was*. The
accommodation rule in CLAUDE.md 4.4 (accumulated surprise S_t crosses a threshold) is consistent with this, but nothing
in the current learner is surprise-gated. A concrete step: weight training steps and memory writes by prediction error.

### F2. Two learning systems with different speeds, and the speed depends on prior knowledge
- Fast hippocampal episodic learning plus slow neocortical structure learning, linked by replay (Kumaran, Hassabis &
  McClelland 2016 `[text]`).
- **Key amendment `[text]`:** the neocortex is not slow per se. New information that is *consistent* with existing
  structure is learned fast without damaging old knowledge, while *inconsistent* information is learned slowly and needs
  interleaved rehearsal to avoid catastrophic interference. In their network simulation the weight changes were large for
  consistent items and small for inconsistent ones. Rats showed one-shot learning of new flavour-place pairs only in an
  arena where they already had a schema.
- Replay is not a faithful tape. It is biased toward surprising, rewarding or informative events (Kumaran 2016 `[text]`), and
  a normative account says the order follows the expected utility of updating a memory (Mattar & Daw 2018 `[abstract]`).
- Schema-congruent items are integrated through medial prefrontal cortex. Strongly incongruent items (prediction error)
  are captured by the medial temporal lobe (van Kesteren et al. 2012, SLIMM framework `[text]`).

**For STML.** This is the best single explanation of our ARC result. A network trained from scratch on about 400 puzzles
has *no schema*, so every puzzle is "inconsistent" and learning is slow and overfits (62-68% train vs 52-56% test). The
logic judge had its structure built in (concept graph, exact labels) and learned fast. Design implication: add a fast
episodic store plus *prioritized* replay of surprising/erroneous episodes next to the slow graph learner.

### F3. Structure is learned separately from content and reused
- The Tolman-Eichenbaum Machine keeps abstract structure (transition rules, learned slowly, shared across worlds) apart
  from content (fast bindings reset in each world). In a new world with known structure one visit per node suffices, and it
  solves transitive inference and family-tree questions on names it has never seen (Whittington et al. 2020 `[abstract]`
  plus a detailed summary of the PMC text I fetched).
- Map-like relational codes may organise knowledge of any kind, not just space (Behrens et al. 2018 `[abstract]`).
- People learn from little data by probabilistic inference over structured generative models, i.e. "intuitive theories"
  (Tenenbaum et al. 2011 `[abstract]`; Lake et al. 2017 `[abstract]`).

**For STML.** Strong support for graph state representations, and a concrete missing experiment: pretrain *structure*
in one domain (ARC objects or logic worlds) and measure how few examples a second domain needs. This is the
cross-domain transfer test you approved.

### F4. "Learning to learn" lives in activations and needs a *family of related tasks*
- Slow dopamine-driven weight learning shapes the prefrontal network into a second, fast learner that runs in its
  activations. The authors state three conditions: a recurrent network, slow RL on its weights, and **a series of
  interrelated tasks** (Wang et al. 2018 `[text]`). The Harlow monkeys reached one-trial learning only "after substantial
  practice".
- The same idea in AI: RL² and "Learning to reinforce learn" (Duan et al. 2016, Wang et al. 2016 `[abstract]`), and the review
  by Botvinick et al. 2019 (fast RL = episodic memory + meta-learned inductive biases `[abstract]`). Algorithm Distillation
  (Laskin et al. 2022) distils RL learning histories into a transformer that improves in context `[abstract]`.

**For STML.** Our learning-to-learn null is *expected* under this theory: our sequences were five unrelated ARC puzzles,
6,000 episodes, no shared structure to exploit. The meta-RL condition is "related tasks, many episodes". The right place
to test it is a task family with shared structure (logic worlds), not random ARC puzzles. The human 50%-to-58% gain
may reflect interface familiarity; that is a hypothesis, not a finding.

### F5. Metacognition is a separate computation
- Confidence is the probability that *your choice* is correct; certainty is posterior precision about a variable, and they
  are different quantities. Miscalibration of behavioural confidence is common and does not mean the brain lacks the
  information (Pouget, Drugowitsch & Kepecs 2016 `[text from PMC]`; Meyniel, Sigman & Mainen 2015 `[abstract]`).
- Confidence is represented in orbitofrontal cortex and drives giving up on a trial, i.e. abstention (Kepecs et al. 2008
  `[abstract]`). Metacognitive accuracy is separable from task accuracy and tied to anterior and lateral prefrontal cortex
  (Fleming & Dolan 2012 `[abstract]`).
- In the AI literature, large language models are fairly well calibrated on well-posed questions (Kadavath et al. 2022
  `[abstract]`), which is the prior art for H3.

**For STML.** Our H3 result matches this picture. The judge's *softmax* confidence collapsed under distribution shift, but a
second-order check, re-judging the claim from the premises it named (sufficiency), ranked its own errors better. In
brain terms: confidence is a computation *about* a first-order decision, not a by-product of it. Design implication: train
the confidence head on such second-order evidence, and calibrate its threshold on shift-like data.

### F6. Mistakes teach most when they were confident and surprising
- Errors made with high confidence are corrected *more* easily (hypercorrection), via surprise and attention to feedback
  (Metcalfe 2017 `[abstract]`). Errorful learning followed by feedback beats error-free study.
- Attempting problems before instruction can improve later learning (Kapur 2008, productive failure `[abstract]`; the effect
  is debated). Self-explanation improves understanding (Chi et al. 1994, N=24, and 1989 correlational `[abstract]`).
  Retrieval practice beats restudy on delayed tests although restudy *feels* better (Roediger & Karpicke 2006 `[abstract]`).

**For STML.** (a) Weight errors by confidence x surprise in training and in replay. (b) The human study should log
**confidence before the answer** and an **explanation**, because the hypercorrection prediction is directly testable
with your game. (c) A Socratic question works by creating an impasse (a prediction error) plus a request to explain, so the
critic's value may sit in *when* it fires, not just in what it says.

### F7. Humans sample a few hypotheses, edit them locally, and ask informative questions
- People generate a small sample of hypotheses (Monte Carlo) and deviate from rational inference accordingly (Dasgupta,
  Schulz & Gershman 2017 `[abstract]`). Learning is open-ended "hacking" of a mental program through many small
  improvements (Rule, Tenenbaum & Piantadosi 2020 `[text]`: they contrast it with theories that rely on local search and
  simple metrics, and list debugging, refactoring and testing as the operations).
- Choosing which test to run is choosing the expected-information-gain maximiser (Oaksford & Chater 1994 `[abstract]`);
  curiosity is information-seeking driven by learning progress and uncertainty (Gottlieb et al. 2013 `[abstract]`).

**For STML.** This is the specification of the not-yet-built cognitive learner (Pillar 1): sample few hypotheses, edit
locally, accumulate surprise, pick the next question by information gain. It also matches what H-ARC shows: after a
wrong answer people keep the grid size 86% of the time and keep a median 93% of cells.

### F8. Brain-compatible learning rules exist, but are not a shortcut
- Predictive-coding networks with local Hebbian updates can approximate backprop (Whittington & Bogacz 2017; Millidge,
  Tschantz & Buckley 2020 `[abstract]`); backprop-like credit assignment is plausible in cortex (Lillicrap et al. 2020
  `[abstract]`); Rao & Ballard 1999 is the origin of "feedback carries predictions, feedforward carries errors".

**For STML.** These show that error-driven, local learning can match backprop; they do not show it is more sample-efficient
on our tasks. I would treat a predictive-coding layer as a research track, not as the fix for the ARC result.

---

## 2. The Socratic / reward side

- **Process feedback beats outcome feedback for *reasoning*.** Scoring every step reached 78% on a MATH subset where outcome
  supervision did less (Lightman et al. 2023 `[abstract]`). On GSM8K, outcome supervision gave similar final-answer error,
  but step-level feedback cut the reasoning error of already-correct answers from 14.0% to 3.4% (Uesato et al. 2022
  `[abstract]`). That is the machine analogue of a Socratic critic. It mirrors what we found in H2: the critique helped when
  the signal carried more information than a right/wrong bit, but the "which premises" part added only about 1 point.
- **Self-questioning and verbal feedback.** Recursive Socratic questioning improves LLM reasoning (Qi et al. 2023
  `[abstract]`); agents that store written reflections instead of updating weights improve across trials (Shinn et al.
  2023, Reflexion `[abstract]`). Both are *language* routes to learning from mistakes; STML tries a *graph* route.
- **Reward as prediction error, not raw outcome.** If the neural teaching signal is reward minus expectation (F1), the
  RL reward in Stage B should be shaped the same way and combined with an honesty term (CLAUDE.md 4.5) so that "I don't
  know yet" has its own prediction-error-like signal.

## 3. What I would change in the plan (ordered)
1. **Study-sizing simulation** with a synthetic learner built from F7 (few sampled hypotheses, local edits, surprise
   threshold). Output: how many participants the game needs. Uses the model of human errors, so its assumptions are explicit.
2. **Cross-domain transfer test** built on F3: pretrain the graph learner on human vs RL trajectories, measure examples
   needed in logic.
3. **Meta-RL on a task family** (logic worlds sharing structure) to test F4 where the theory predicts it should work.
4. **Fast episodic store + prioritized replay** (F2) and **surprise-weighted** training (F1, F6).
5. **Second-order confidence** trained on re-judgments (F5), with thresholds fitted on shifted data (already shown to matter).
6. **Game logging** (F6): confidence before answering, explanations, and the order of tasks.
7. **Process-level imitation on H-ARC** (predict the next edit), once there is a bigger model; F7 says edits are the unit.

## 4. Gap topics searched on 2026-10-15 (cognitive architectures, development, sleep, language)
Bibliographic details were confirmed by search; the content below is `[abstract]` or general knowledge, not read in the papers.
- **Cognitive architectures.** ACT-R (Anderson et al. 2004): memory activation that strengthens with use and decays, rules chosen by
  learned utility. Soar (Laird, Newell & Rosenbloom 1987): an *impasse* creates a subgoal, and the result is stored as a new rule
  ("chunking") - impasse-driven learning is the machine form of productive failure and of the Socratic question. Standard Model
  of the Mind (Laird, Lebiere & Rosenbloom 2017): the common component list. STML has no working-memory buffer, no
  declarative/procedural split and no impasse mechanism.
- **Development.** Children revise intuitive theories by causal Bayesian learning, play-as-experiment and imitation (Gopnik &
  Wellman 2012); one-shot concept learning by program induction (Lake, Salakhutdinov & Tenenbaum 2015; Lake & Piantadosi 2020);
  systematic generalisation by meta-learning on a stream of compositional tasks (Lake & Baroni 2023, = F4 on a task family).
- **Sleep.** Offline reactivation consolidates memory (Diekelmann & Born 2010): the biological analogue of offline replay (F2).
- **Language.** Infants segment words from a continuous stream using transition statistics only, with no feedback (Saffran, Aslin &
  Newport 1996); grammar can be learned implicitly (Reber 1967); learning combines statistics with social interaction (Kuhl
  2004) and changes with age (Newport 1990); comprehension and production use coupled streams (Hickok & Poeppel 2007);
  syntax and semantics use partly different networks (Friederici 2011). A connectionist syntax-learning model driven by
  prediction error (Chang, Dell & Bock 2006) is in the index but its abstract was not confirmed. Duolingo data (Settles & Meeder
  2016, half-life regression; SLAM 2018, word-level errors) models retention and error prediction, with one attempt per exercise.
  Design consequences for a language game are in `docs/ARCHITECTURE_V2.md` section 5.

## 5. Caveats and gaps
- Most mechanisms above were established in animals or in small human studies; transfer to code is an inference, not a result.
- Several education findings are contested (productive failure) or based on small samples (self-explanation, N=24).
- **Still not covered:** attention, emotion/motivation and the neuroscience of *second*-language learning with feedback, and
  the literature on artificial-grammar learning with error feedback (my last search for it returned nothing usable).
- Papers I could not download: Dabney 2020, Behrens 2018, Fleming & Dolan 2012, Gottlieb 2013 (open-access pages are linked
  in `INDEX.md`). Many classic papers (Schultz, Holroyd, Kepecs, Metcalfe, Kapur, Chi, Roediger) are paywalled and are
  cited by identifier only.

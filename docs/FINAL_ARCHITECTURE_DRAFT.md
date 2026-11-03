# STML: draft of the final architecture, from the evidence so far

Status: **draft, written 2026-10-16 and updated 2026-10-18 (Q15-Q26) before any human game data.** Every line says what the evidence is (run folders in `runs/`, details in CLAUDE.md section 7). To be revised after
`python scripts/analyze_game.py logs/`. "Supported" means a pre-registered prediction held on 3 seeds or more; numbers are accuracy points unless stated, and differences under about 2 points are
within run-to-run noise.

## 0. The three hypotheses, in one paragraph each (state on 2026-10-18)

**H1, process imitation.** On ARC with the pure learner, imitating human learning histories before RL beats imitating answers (+3.4 over 17 splits) and the RL agent's own history (+4.6), and is far better than RL from scratch (+8-10); the shuffled-history control collapses (-20). But the advantage disappears when a solution prior is supplied by other means (hand features: +0.8; a puzzle-level pretext: +0.4). So process imitation is a valid way to obtain the prior, not an irreplaceable one, on this task. The logic domain, where the thesis lives, awaits the human game data.
**H2, diagnostic feedback.** In the sequential hidden-premise regime the true verdict beats a right/wrong bit by +8 to +10 points on late claims (+7.8 with plentiful re-masked data, where binary feedback delivers half of the verdict's gain, 22% with little data); extra structure on the feedback ('which premises were used', signed surprise) adds nothing; an unreliable teacher is the largest risk (20% wrong verdicts cost 40-60% of the gain; detection works (AUROC 0.90), repair does not).
**H3, knowing what it does not know.** A proof-availability head ranks 'a proof exists now' with AUROC 0.99-1.00 under shift and on larger worlds where confidence is 0.74-0.81; with plentiful training the verdict model's own confidence is the better in-distribution gate; abstention thresholds can be repaired online from committed answers with hysteresis; the head's reward gain out of range is about +0.05 +- 0.03.

## 1. What the learner is

**A graph model over an explicit state.** Kinds are nodes; premises and everything learned are typed edges; the claim being judged is marked on the nodes. Encoder: the **hypernetwork core**
(per-relation matrices generated from relation embeddings by a shared network) when cross-domain sharing is wanted, otherwise the original per-relation GNN; both are equal within noise
(78.5% vs 75.6% at N = 250, 96.7% vs 97.0% at N = 2000). A shallow net is enough (2, 4 and 6 layers equal; width 64; default hyperparameters sit on a plateau, tuning gave +0.5/+0.7, inside the seed sd).

**Feedback enters the state as edges, not as a loss weight.** A revealed verdict is added to the graph as a typed edge; the next judgement reasons over it. This is where the learning comes from:
+9 to +21 points at late positions, depending on how many claims have been revealed (10 vs 20 per episode: +12 vs +22); binary right/wrong feedback gives about a fifth of that (+2.8 vs +14.4).

## 2. What was tested and is supported

| decision | evidence | status |
| --- | --- | --- |
| Show the learner the **verdict**, not only right/wrong | verdict +81.3% vs binary +69.8% vs none +67.0% at the last claims (Q8) | supported, large |
| Choose what to ask by **uncertainty** (entropy or margin), not at random | +5.3 points over random at 6 asks of 12; +5.0 at 3; robust across model sizes/depths, pool 8-20; replicated on fresh worlds (+4.7, all 15 cells +3.5..+5.5) (Q1, H2, Q9) | supported, robust |
| What makes asking informative is asking claims **not yet provable** | entropy restricted to unprovable +5.1 of +5.2; provable-only asking -1.1 (Q1b) | supported |
| A **proof-availability head** ("a proof exists from what I have seen") as the abstention gate | AUROC 0.997-1.000 on every shifted set and 0.99 on 8-10 kind worlds where the verdict model's confidence is 0.74-0.81 (Q3, Q3b) | supported for ranking |
| ... as a **reward** advantage over plain confidence | in-dist +0.014 (Q9) / +0.026 (Q9b); 8-kind chains +0.014 with 100-episode sets (Q9, underpowered, 1 of 5 sets >= 0.03) but +0.092 with 400-episode sets (Q9b, 5 of 5 sets >= 0.03, all 15 cells >= +0.051) | sign positive on average; across 8 trainings on one fixed chain set the gap is +0.053 with sd 0.034 (1 of 8 trainings negative); in-distribution +0.016 (sd 0.011) (Q9c). Use the head as a gate for its AUROC and shift-robustness, not for a guaranteed reward gain |
| **Train on the target size range** | 8-kind chain accuracy 59.4% -> 78.5% after adding 8-10 kind worlds (Q4b); +25 points for 24-kind chains earlier | supported, large |
| **Re-mask the same worlds as free data** (new random visible subset and asked claims per draw) | 300 worlds x 3 draws +3.4 late (88% of the gain of 3x distinct worlds); x10 +6.0 overall over an equal-steps baseline; x30 87.1% at claims 7-10; 100 worlds x30 (84.0%) ~ 1000 distinct worlds (83.5%); more steps without diversity do nothing (Q17, Q17b, Q18) | supported, the strongest cheap lever; shrinks with size generalisation (no help on unseen sizes) |
| **Rule-preserving augmentation on ARC** (dihedral + colour permutation, candidates included) | +5.9 solved for human-history imitation, +6.2 for answers, 4x longer training -1.1 (Q27); human histories still not better than answers | supported on ARC; perception remains the bottleneck |
| **Imitate HUMAN learning histories before RL (H1, ARC)** | with augmentation and equal data/RL budget: humans-then-RL 64.5% vs RL-agent-history-then-RL 59.9% vs answers-then-RL 59.1% vs RL from scratch 54.7% (Q31); +3.4 over answers pooled over 9 splits, positive in 9/9; synthetic failed attempts do not reproduce it (Q30); no human advantage without RL (Q27) | how much human data: the imitation phase beats RL from scratch by ~+6 with any source from 10% to 100% of the steps, but humans beat answers only with the full ~6,500 steps (+2.9; +0.1 at 25%, -1.1 at 10%) (Q38) | supported on ARC for the PURE learner only: with the 12 hand features (a learner that perceives) all pipelines reach ~65% and humans-then-RL is +0.8 over answers, -0.3 over RL from scratch (Q34); needs the human game data for the logic domain |
| **More training data** | late-claim accuracy 81.3% -> 85.1% from 600 to 2000 worlds, still rising (Q5); sweep: gap closed 22% -> 35% -> 45% for 100/300/1000 worlds | supported; the main limit |
| **Asking policy = the learner's own uncertainty**; RL finds nothing better | learned REINFORCE policy 88.1% vs entropy rule 88.2% vs random 83.1%; structure-only features 83.4% = random (Q10) | supported |
| **Trustworthy feedback matters**: gain falls to 74% / 38% / 25% of the clean gain at 10 / 20 / 30% wrong verdicts (600 worlds); 55% / 32% at 20 / 30% with 2000 worlds, where noise-trained beats clean-trained by +2.1 / +3.1 | Q11, Q11b | supported; discounting is learnable but data-hungry |
| **Limited memory: store the surprising verdicts** | a third of the memory gives two thirds of the gain; +2.2 over a capacity-matched random store (CI +1.0..+3.3), -4.8 vs keep-all; advantage gone (-0.9) with 20% wrong verdicts (Q6, Q6b, Q12) | small, real only with trustworthy feedback |
| **Deployable abstention recalibration** from committed answers only | start at the Bayes threshold of the calibrated confidence, re-fit every 25 items from committed answers, adopt a new threshold only if it beats the current by a margin 0.1-0.2: recovers 94% of the full-label gain in overconfident cells, removes 91-97% of the ratchet loss elsewhere, 89% of the full-label gain overall (Q13, Q13b; offline replay, margin chosen on the same replay) | supported offline; not tested with a live learner |
| **More feedback per episode** | 5 / 10 / 20 claims: gain +5.2 / +11.9 / +22.0 (feedback sweep) | supported, large |

## 3. What was tested and is NOT supported (do not build on it)

- **Surprise as a loss weight** (surprise, confident-error, prioritised replay): no accuracy gain over plain training; only a calibration trade-off that the standard focal loss matches or beats; prioritised replay hurt at small N (-7..-11).
- **Signed surprise as an input** (as edge types or node features): no gain over the verdict, and a shuffled control does as well.
- **"Which premises were used" marks** on top of the verdict: adds nothing (range -0.9..+0.4 across runs); not a data-dilution effect (Q5).
- **Imitating human learning histories on ARC** (H-ARC): worse than imitating answers or RL for solving; no transfer to logic from any kind of ARC history (fine-tuning within 0.6 points; frozen core within 0-2 of random); a person's own failed attempt carries no
  detectable information about their next edit beyond a regulariser effect. These were tests on data without confidence ratings or explanations.
- **Depth, width and hyperparameter tuning** of the feedback learner, also with plentiful re-masked data (Q20: width 128 +0.1, 6 layers -0.2).
- **Self-supervised masked-node pretraining of the ARC object-graph encoder** (Q36: -0.5..-1.8 points; the hand features of Q34 help, this does not).
- **Test-time training on the revealed verdicts** (Q26, Q26b: every learning rate hurts or is neutral), **randomising the visible fraction** (Q19: -0.3..-0.7), **auxiliary hidden-premise prediction** (Q16).
- **Repairing a noisy teacher** by deletion, suspect flags or re-masking: leave-one-out detection works (AUROC 0.90) but recovers about 2 of the ~8 points lost at 20% wrong verdicts (Q21-Q23). The feedback source must be reliable.
- **Auxiliary hidden-premise prediction** as a data-efficiency trick (Q16: +0.3 / -0.3 points late at 300 / 1000 worlds; the verdict loss already supervises the same structure). **Lemma scratchpad as a general fix** (Q15, Q15b: +2..+5 points at 16-24 kinds, fades at 32 as the judge's own lemmas get less precise).
- **A learned asking policy beyond the entropy rule** (Q10), **structure-only asking** (= random), and **surprise-gated memory with noisy feedback** (Q12: advantage gone).

## 4. The pipeline to carry forward (numbers = in-distribution baseline for the human comparison)

ask 6 of 12 candidate claims by entropy -> verdicts revealed and added to the graph -> judge held-out claims -> abstain ("I don't know yet") when no proof is available.
Training: levels 2-5 plus 8- and 10-kind worlds, **each world re-masked 5 times** (Q25). Numbers with entropy asks (3 seeds): **in-distribution 89.4%** (87.2% without re-masking, 81.5-84.2% with random asks),
**8-kind chains 81.3%** (78.5%), 10-kind trees 77.0% (76.7%; an apparent informational ceiling). Luck-adjusted reward at lam = 4 (abstain -0.2), confidence / head / oracle gate: in-dist +0.254 / +0.224 / +0.226;
8-kind chains +0.193 / +0.178 / +0.188; 10-kind trees -0.014 / -0.021 / -0.012. The oracle gate is not an upper bound (a confident right answer without a proof earns 0, better than abstaining).
Gate: with plentiful training the verdict model's own confidence is the better in-distribution gate (+0.254 vs +0.224, Q24, Q25); the proof-availability head remains the shift-robust ranker
(AUROC 0.99-1.00 on shifted and larger worlds, Q3, Q3b) with a reward advantage of +0.05 +- 0.03 across trainings on out-of-range chains (Q9c). Threshold: start at the Bayes threshold of the calibrated
confidence and re-fit from committed answers with a hysteresis margin of 0.1-0.2 (Q13, Q13b; offline). Scratchpad lemmas add +2..+5 at 16-24 kinds but fade at 32 (Q15, Q15b); ensemble agreement raises their precision but not their usefulness (Q39), so the idea is closed.

## 5. What the human logs must decide (analysis already in `scripts/analyze_game.py`)

1. **Is a person's confidence informative** (E1 AUROC > 0.5) and is their "don't know yet" **selective** (by truth class and proof size)? Compare with the machine's gates (AUROC 0.92 confidence / 1.000 head for 'a proof exists').
2. **Does feedback type matter for people as it does for the learner** (verdict vs right/wrong)? Needs an `arm` in the session; the game as built gives everyone the diagnostic critic. The machine predicts about +8 points overall, +11 late.
3. **Hypercorrection and revision** (E2): not detectable in the current design at N <= 300 unless large (power 0.22-0.5); consider a direct re-test of failed claims.
4. **Do people ask the right questions?** The game asks players to propose statements that follow, i.e. claims they believe provable, which the machine results show is the low-information kind. Logging which claims people try (and their confidence) allows the same
   provable/unprovable split to be measured in humans.
5. Recalibrate `stml/synthetic.py: Population` and the sizing from the observed spread (printed by the analysis).

## 6. Open

- Human data for H1 proper (process imitation) with confidence and explanations: the ARC tests were null/reversed but lacked those channels.
- Hidden-premise episodes are a synthetic stand-in for 'learning an unknown world'; the real game shows all premises. A hidden-information game variant would make the machine/human comparison direct.
- No test beyond 10-kind worlds for the hidden-premise regime (exact labels need <= 10 kinds). Noisy feedback is tested only as random verdict flips, not as systematic human error. The asking policy was trained on low-dimensional features, not on the graph embedding. Meta-RL (Q7) was not run; Q10 is its reduced form.

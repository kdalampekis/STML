# What the experiments taught us (2026-10-14 to 2026-10-20, before any human game data)

Numbers: `docs/META_ANALYSIS_numbers.md` (made by `scripts/meta_analysis.py`, run folder `runs/*_meta_analysis`); figures: `docs/figures/forest_logic.png`, `docs/figures/forest_arc.png`.
All effects are machine-only, on synthetic logic worlds or on H-ARC puzzles; the human game data are still missing, so everything about people is still a prediction.

## The five findings worth keeping

1. **What matters is the variety of the training instances, not the number of gradient steps.** Re-drawing which premises are hidden and which claims are asked (re-masking) is as good as new worlds: 100 worlds re-masked 30 times (84.0%) match 1000 distinct worlds (83.5%); 10 draws beat an equal-steps baseline by +7.6 points while 10x more steps on the same episodes loses 1.3. The same holds on ARC: rule-preserving augmentation +5.9, 4x longer training -1.1. It is the cheapest lever in the programme and works in both domains.
2. **The content of the feedback matters; extra structure does not.** The true verdict beats a right/wrong bit by +8 to +12 points on late claims (the right/wrong bit is worth about half of the verdict's gain when data are plentiful, about a fifth when they are scarce). Telling the learner which premises were used, or a signed surprise, adds nothing. A wrong teacher is the main danger: 20% wrong verdicts cost 40-60% of the gain, a leave-one-out check detects them well (AUROC 0.90) but repairs only about a quarter of the loss.
3. **The Socratic question is the learner's own uncertainty.** Asking the claim the model is least sure about beats random asking by about +5 (replicated, robust to 20% wrong verdicts); a policy trained by RL discovers nothing better, and a policy without the uncertainty signal is no better than random.
4. **Knowing what it does not know works as a ranking, and the threshold can be repaired online.** A 'proof exists now' head ranks perfectly (AUROC 0.99-1.00) even where verdict confidence is 0.74-0.81; once the verdict model has plentiful training its own confidence is the better in-distribution gate; an abstention threshold can be recovered from committed answers only (94% of the full-label gain, with hysteresis).
5. **Process imitation (H1) has first, narrow support, and the mechanism looks like a prior.** On ARC, imitating human learning histories before RL beats imitating answers (+3.4 pooled over 12 splits with augmentation, 15 of 17 positive overall) and the RL agent's own history (+4.6); histories shuffled among steps collapse the pipeline (-19.7). But the advantage vanishes when the learner gets a solution prior another way (hand features +0.8, a puzzle-level pretext +0.4), it needs the volume of human steps (+0.1 at a quarter of them), and it is absent in the logic experiments we could run. Read it as: human histories are a way to obtain a prior about what solutions look like and how failures lead to the next attempt, not as a stand-alone advantage.

## What did not work (and should not be tried again as is)

Surprise as a loss weight, as an input, or as a replay priority (harmful at small data); the auxiliary hidden-premise loss; test-time training (hurts at every learning rate); randomising the visible fraction; wider or deeper networks (not capacity-limited, even with plentiful data); the lemma scratchpad at 32 kinds even with ensemble verification; masked-node pretraining of the ARC encoder; ARC -> logic transfer from any source; ARCTraj first attempts added to the human steps; a learned asking policy beyond the entropy rule.

## About the experiments themselves

- Predictions written before running held about 60% of the time (68% in the first half of the programme, 51% in the second): our intuition about which tricks help is only slightly better than a coin once the easy wins are taken.
- Typical noise: about 1 point between runs of the same condition in logic, about 3 points per split on ARC. Differences under 2 points are mostly noise; several early 'effects' shrank or changed size on replication (head-vs-confidence gate reward +0.014 / +0.092 / +0.053; human-history lead on ARC +2.9 / +2.1 / +5.3).
- Hardware: slowdowns when the Mac was on battery were paging, not compute; they cost days of wall-clock but did not change any result.

## What the human data must decide

Whether people's confidence and 'I don't know yet' are informative and selective; whether a diagnostic critic beats a right/wrong critic for people as it does for the learner (about +8 points predicted for a data-rich learner); whether people ask the low-information kind of question the game invites; and, for H1 proper, whether human histories in the logic domain carry the same prior as on ARC. See `docs/FINAL_ARCHITECTURE_DRAFT.md` section 5.

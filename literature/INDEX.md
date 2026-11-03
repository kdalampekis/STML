# Literature for STML

Neuroscience and cognitive science of learning, learning from errors, metacognition, and the AI work closest to STML. Source of truth: `papers.json`; rebuild with `python scripts/fetch_literature.py`. A PDF is stored only for open-access / author-hosted / preprint copies; the rest are cited by identifier. Details marked 'not verified' came from memory or a single search snippet.

## dopamine / prediction error

- **Schultz, Dayan, Montague (1997)**. A neural substrate of prediction and reward. *Science 275:1593-1599*. PMID 9054347 - [page](https://www.gatsby.ucl.ac.uk/~dayan/papers/sdm97.html)  
  Dopamine neurons signal reward prediction errors = TD error. The neural root of 'learn from the gap between expected and actual outcome'.
- **Dabney, Kurth-Nelson, Uchida, Starkweather, Hassabis, Munos, Botvinick (2020)**. A distributional code for value in dopamine-based reinforcement learning. *Nature 577:671-675*. doi:10.1038/s41586-019-1924-6 - [page](https://pmc.ncbi.nlm.nih.gov/articles/PMC7476215)  
  Dopamine neurons encode a distribution over outcomes, not a mean: the brain carries uncertainty in its value code. Relevant to calibration / 'knowing what it does not know'.
- **Frank, Seeberger, O'Reilly (2004)**. By carrot or by stick: cognitive reinforcement learning in parkinsonism. *Science 306:1940-1943*. doi:10.1126/science.1102941 - citation only  
  Separate pathways learn from positive vs negative outcomes (dopamine bursts vs dips). Supports separate treatment of success and error signals.

## error monitoring

- **Holroyd, Coles (2002)**. The neural basis of human error processing: reinforcement learning, dopamine, and the error-related negativity. *Psychological Review 109(4):679-709*. PsycNet - citation only  
  Anterior cingulate error signal (ERN) is generated when dopamine conveys a negative prediction error; the brain's 'that was wrong' alarm.
- **Botvinick, Braver, Barch, Carter, Cohen (2001)**. Conflict monitoring and cognitive control. *Psychological Review 108(3):624-652*. doi:10.1037/0033-295X.108.3.624 - citation only  
  Anterior cingulate monitors conflict between competing responses and recruits more control. A mechanism for 'noticing a contradiction'.

## predictive coding

- **Rao, Ballard (1999)**. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience 2(1):79-87*. doi:10.1038/4580 - citation only  
  Feedback carries predictions, feedforward carries residual errors. Origin of the predictive-coding view of cortex.

## brain-inspired learning rules

- **Whittington, Bogacz (2017)**. An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. *Neural Computation 29(5):1229-1262*. doi:10.1162/NECO_a_00949 - [pdf](pdfs/whittington2017/whittington2017.pdf)  
  A network using only local Hebbian updates and prediction errors can approximate backprop.
- **Millidge, Tschantz, Buckley (2020)**. Predictive coding approximates backprop along arbitrary computation graphs. *arXiv 2006.04182*. arXiv:2006.04182 - [pdf](pdfs/millidge2020/millidge2020.pdf)  
  Predictive coding with local updates matches backprop gradients on arbitrary graphs (CNNs, RNNs, LSTMs).
- **Lillicrap, Santoro, Marris, Akerman, Hinton (2020)**. Backpropagation and the brain. *Nature Reviews Neuroscience*. doi:10.1038/s41583-020-0277-3 - citation only  
  NGRAD framework: error signals approximated locally by activity differences; backprop-like learning may occur in cortex.
- **Gerstner, Lehmann, Liakoni, Corneil, Brea (2018)**. Eligibility traces and plasticity on behavioral time scales: experimental support of neoHebbian three-factor learning rules. *Frontiers in Neural Circuits 12:53*. doi:10.3389/fncir.2018.00053 - [pdf](pdfs/gerstner2018/gerstner2018.pdf)  
  Learning = Hebbian eligibility trace x a third gating factor (reward minus expected, or surprise). The cellular form of 'learn when surprised'.

## memory systems

- **Kumaran, Hassabis, McClelland (2016)**. What learning systems do intelligent agents need? Complementary learning systems theory updated. *Trends in Cognitive Sciences*. Trends in Cognitive Sciences 2016 (DOI not verified) - [pdf](pdfs/kumaran2016/kumaran2016.pdf)  
  Fast hippocampal episodic learning + slow neocortical structure learning, linked by replay. Blueprint for fast/slow learners.
- **van Kesteren, Ruiter, Fernandez, Henson (2012)**. How schema and novelty augment memory formation. *Trends in Neurosciences 35(4):211-219*. doi:10.1016/j.tins.2012.02.001 - [pdf](pdfs/vankesteren2012/vankesteren2012.pdf)  
  Schema-congruent info is integrated via prefrontal cortex; schema-incongruent (surprising) info drives hippocampal encoding. How contradictions are stored.
- **Mattar, Daw (2018)**. Prioritized memory access explains planning and hippocampal replay. *Nature Neuroscience 21(11):1609-1617*. PMC6203620 - [pdf](pdfs/mattar2018/mattar2018.pdf)  
  Replay order follows expected utility of updating a memory. A principle for which past experiences a learner should rehearse.
- **Diekelmann, Born (2010)**. The memory function of sleep. *Nature Reviews Neuroscience 11(2):114-126*. doi:10.1038/nrn2762 - citation only  
  Sleep supports consolidation (offline reactivation). Biological analogue of offline replay in Stage A.

## structure / cognitive maps

- **Whittington et al. (2020)**. The Tolman-Eichenbaum Machine: unifying space and relational memory through generalization in the hippocampal formation. *Cell 183*. doi:10.1016/j.cell.2020.10.024 - [pdf](pdfs/whittington2020/whittington2020.pdf)  
  Separates abstract structure (transition rules, learned slowly, shared across worlds) from content (fast Hebbian bindings). One visit per node suffices in a new world with known structure. Direct support for graph learners and logic transfer.
- **Behrens, Muller, Whittington, Mark, Baram, Stachenfeld, Kurth-Nelson (2018)**. What is a cognitive map? Organizing knowledge for flexible behavior. *Neuron 100:490-509*. doi:10.1016/j.neuron.2018.10.002 - [page](https://www.biorxiv.org/content/10.1101/365593)  
  Map-like relational codes may organise knowledge of all kinds, not just space. Justifies graph state representations.

## metacognition / confidence

- **Fleming, Dolan (2012)**. The neural basis of metacognitive ability. *Phil. Trans. R. Soc. B 367:1338-1349*. doi:10.1098/rstb.2011.0417 - [page](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3318765/)  
  Metacognitive accuracy is separable from task performance and tied to anterior/lateral prefrontal cortex. Knowing how well you know is a distinct skill.
- **Pouget, Drugowitsch, Kepecs (2016)**. Confidence and certainty: distinct probabilistic quantities for different goals. *Nature Neuroscience 19(3):366-374*. doi:10.1038/nn.4240 - [page](https://pmc.ncbi.nlm.nih.gov/articles/PMC5378479)  
  Confidence = probability that a choice is correct (choice-dependent); certainty = posterior precision about a variable. Clean definitions for the H3 confidence head.
- **Kepecs, Uchida, Zariwala, Mainen (2008)**. Neural correlates, computation and behavioural impact of decision confidence. *Nature 455:227-231*. doi:10.1038/nature07200 - citation only  
  Orbitofrontal neurons in rats track decision confidence; confidence drives giving up on a trial (abstention).
- **(authors not verified; Neuron 2015) (2015)**. Confidence as Bayesian probability: from neural origins to behavior. *Neuron (PII S0896-6273(15)00828-4)*. PII S0896-6273(15)00828-4 - [pdf](pdfs/meyniel2015/meyniel2015.pdf)  
  Review of confidence as the probability of being correct, its neural origins and behavioural uses. Companion to Pouget et al. 2016.

## meta-RL

- **Botvinick, Ritter, Wang, Kurth-Nelson, Blundell, Hassabis (2019)**. Reinforcement learning, fast and slow. *Trends in Cognitive Sciences 23(5):408-422*. doi:10.1016/j.tics.2019.02.006 - citation only  
  Fast RL = episodic memory + meta-learned inductive biases from a slow outer loop. Explains why a from-scratch learner on 400 puzzles is slow.
- **Wang, Kurth-Nelson, Kumaran, Tirumala, Soyer, Leibo, Hassabis, Botvinick (2018)**. Prefrontal cortex as a meta-reinforcement learning system. *bioRxiv 295964; Nature Neuroscience 2018 (venue not confirmed in my search)*. doi:10.1101/295964 - [pdf](pdfs/wang2018/wang2018.pdf)  
  Slow dopamine-driven learning sculpts prefrontal dynamics into a second, fast RL algorithm that runs in activations. Direct model for our learning-to-learn experiment.
- **Wang, Kurth-Nelson, Tirumala, Soyer, Leibo, Munos, Blundell, Kumaran, Botvinick (2016)**. Learning to reinforcement learn. *arXiv 1611.05763*. arXiv:1611.05763 - [pdf](pdfs/wang2016/wang2016.pdf)  
  Deep meta-RL: a recurrent agent trained by one RL algorithm implements a second, faster one.
- **Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel (2016)**. RL^2: fast reinforcement learning via slow reinforcement learning. *arXiv 1611.02779*. arXiv:1611.02779 - [pdf](pdfs/duan2016/duan2016.pdf)  
  Fast RL represented as an RNN whose weights are learned by slow RL; the method our sequence RL follows.
- **Laskin, Wang, Oh, Parisotto, Spencer, Steigerwald, Strouse, Hansen, Filos, Brooks, Gazeau, Sahni, Singh, Mnih (2022)**. In-context reinforcement learning with algorithm distillation. *arXiv 2210.14215 (ICLR 2023 per project notes)*. arXiv:2210.14215 - [pdf](pdfs/laskin2022/laskin2022.pdf)  
  Causal transformer on RL learning histories improves in context. The 'imitate learning histories' control in H1.

## human concept learning

- **Tenenbaum, Kemp, Griffiths, Goodman (2011)**. How to grow a mind: statistics, structure, and abstraction. *Science 331:1279-1285*. doi:10.1126/science.1192788 - [pdf](pdfs/tenenbaum2011/tenenbaum2011.pdf)  
  People learn from little data by probabilistic inference over structured, abstract generative models (theories, schemas).
- **Lake, Ullman, Tenenbaum, Gershman (2017)**. Building machines that learn and think like people. *Behavioral and Brain Sciences*. arXiv:1604.00289 - [pdf](pdfs/lake2017/lake2017.pdf)  
  Machines need causal models, intuitive theories and compositionality with learning-to-learn. Frames what STML should add.
- **Dasgupta, Schulz, Gershman (2017)**. Where do hypotheses come from?. *CBMM Memo 056 (journal version not confirmed in my search)*. hdl:1721.1/105158 - [page](https://dspace.mit.edu/handle/1721.1/105158)  
  People sample a few self-generated hypotheses (Monte Carlo) and deviate from rational inference accordingly. A model of human hypothesis revision for Pillar 1.
- **Rule, Tenenbaum, Piantadosi (2020)**. The child as hacker. *Trends in Cognitive Sciences 24(11):900-915*. doi:10.1016/j.tics.2020.07.005 - [pdf](pdfs/rule2020/rule2020.pdf)  
  Learning as program improvement: repeated small edits to a symbolic program. Matches the 'local repair' we see in H-ARC revisions.
- **Gopnik, Wellman (2012)**. Reconstructing constructivism: causal models, Bayesian learning mechanisms, and the theory theory. *Psychological Bulletin 138(6):1085-1108*. doi:10.1037/a0028044 - [page](https://pmc.ncbi.nlm.nih.gov/articles/PMC3422420)  
  Children revise intuitive theories through causal Bayesian learning, informal experiments in play, and imitation; theory change from 16 months to 4 years. The developmental version of assimilation / accommodation.
- **Lake, Salakhutdinov, Tenenbaum (2015)**. Human-level concept learning through probabilistic program induction. *Science 350(6266):1332-1338*. doi:10.1126/science.aab3050 (from memory) - citation only  
  Concepts as small programs inferred from one example reach human-level one-shot classification and beat deep nets. Evidence for structured priors.
- **Lake, Piantadosi (2020)**. People infer recursive visual concepts from just a few examples. *Computational Brain & Behavior 3:54-65 (arXiv 1904.08034)*. arXiv:1904.08034 - [pdf](pdfs/lake2020/lake2020.pdf)  
  People infer latent generating programs from 1-3 examples; a Bayesian program-learning model fits, a pretrained deep net does not.
- **Lake, Baroni (2023)**. Human-like systematic generalization through a meta-learning neural network. *Nature 623(7985):115-121*. doi:10.1038/s41586-023-06668-3 - [page](https://pmc.ncbi.nlm.nih.gov/articles/PMC10620072)  
  Meta-learning for compositionality (MLC): train on a changing stream of compositional tasks; only MLC combined systematicity and flexibility among seven models. A meta-learning recipe on a task FAMILY.

## information seeking

- **Oaksford, Chater (1994)**. A rational analysis of the selection task as optimal data selection. *Psychological Review 101(4):608-631*. PsycNet - citation only  
  Choosing which test to run = maximizing expected information gain. Basis of the Socratic question q* = argmax information gain.
- **Gottlieb, Oudeyer, Lopes, Baranes (2013)**. Information-seeking, curiosity, and attention: computational and neural mechanisms. *Trends in Cognitive Sciences 17(11):585-593*. doi:10.1016/j.tics.2013.09.001 - [page](https://pmc.ncbi.nlm.nih.gov/articles/PMC4193662)  
  Curiosity as information-seeking, computed from learning progress and uncertainty; neural basis.

## learning from errors (education)

- **Metcalfe (2017)**. Learning from errors. *Annual Review of Psychology 68:465-489*. doi:10.1146/annurev-psych-010416-044022 - citation only  
  Errorful learning plus corrective feedback helps; high-confidence errors are corrected more (hypercorrection), via surprise and attention to feedback. Core evidence that mistakes teach.
- **Kapur (2008)**. Productive failure. *Cognition and Instruction 26(3):379-424*. doi:10.1080/07370000802212669 (from memory, verify) - citation only  
  Attempting problems before instruction, and failing, can improve later learning/transfer (impasse-driven learning). Effect size debated.
- **Chi, de Leeuw, Chiu, LaVancher (1994)**. Eliciting self-explanations improves understanding. *Cognitive Science 18(3)*. Cog Sci 18(3) - citation only  
  Prompting learners to explain themselves improves understanding. Basis for the explanation part of the human study. Small sample (N=24).
- **Chi, Bassok, Lewis, Reimann, Glaser (1989)**. Self-explanations: how students study and use examples in learning to solve problems. *Cognitive Science 13(2)*. Cog Sci 13(2) - citation only  
  Good learners generate explanations and monitor understanding better. Correlational.
- **Roediger, Karpicke (2006)**. Test-enhanced learning: taking memory tests improves long-term retention. *Psychological Science 17(3):249-255*. Psych Sci 17(3) - citation only  
  Retrieval practice beats restudy on delayed tests even though restudy feels better. Learning signals and feelings of learning can diverge.

## AI: process feedback / Socratic

- **Lightman et al. (2023)**. Let's verify step by step. *arXiv 2305.20050; ICLR 2024*. arXiv:2305.20050 - [pdf](pdfs/lightman2023/lightman2023.pdf)  
  Process supervision (scoring each reasoning step) beats outcome supervision (final answer only): 78% on a MATH subset. The AI analogue of a Socratic critic that says where the reasoning broke.
- **Uesato et al. (2022)**. Solving math word problems with process- and outcome-based feedback. *arXiv 2211.14275*. arXiv:2211.14275 - [pdf](pdfs/uesato2022/uesato2022.pdf)  
  Outcome supervision gets similar final-answer error; process-based feedback is needed for correct reasoning (reasoning error 14.0% to 3.4%).
- **Qi, Xu, Shen, Liu, Jin, Wang, Huang (2023)**. The art of SOCRATIC QUESTIONING: recursive thinking with large language models. *EMNLP 2023 / arXiv 2305.14999*. arXiv:2305.14999 - [pdf](pdfs/qi2023/qi2023.pdf)  
  Recursive self-questioning (ask and answer sub-questions) improves LLM reasoning. Socratic questioning used by the learner itself.
- **Shinn et al. (2023)**. Reflexion: language agents with verbal reinforcement learning. *NeurIPS 2023 / arXiv 2303.11366*. arXiv:2303.11366 - [pdf](pdfs/shinn2023/shinn2023.pdf)  
  Agents turn feedback into written reflections stored in memory instead of updating weights (91% pass@1 HumanEval). Learning from mistakes via verbal feedback.

## AI: knowing what you know

- **Kadavath et al. (Anthropic) (2022)**. Language models (mostly) know what they know. *arXiv 2207.05221*. arXiv:2207.05221 - [pdf](pdfs/kadavath2022/kadavath2022.pdf)  
  Large LMs are well calibrated on multiple-choice/true-false when prompted well and can predict if they know an answer. Prior art for H3 (STML defines 'don't know' as absence of a certificate).

## AI: brain-inspired architecture

- **Goyal, Bengio (2020)**. Inductive biases for deep learning of higher-level cognition. *arXiv 2011.15091*. arXiv:2011.15091 - [pdf](pdfs/goyal2020/goyal2020.pdf)  
  Inductive biases from conscious, sequential, System-2 processing (sparse factors, attention, modularity) for out-of-distribution generalization.
- **Hassabis, Kumaran, Summerfield, Botvinick (2017)**. Neuroscience-inspired artificial intelligence. *Neuron 95(2):245-258*. doi:10.1016/j.neuron.2017.06.011 - citation only  
  Survey of how neuroscience has inspired and can keep inspiring AI (memory, attention, planning, transfer).

## data / learning logs

- **Piech, Bassen, Huang, Ganguli, Sahami, Guibas, Sohl-Dickstein (2015)**. Deep knowledge tracing. *NeurIPS 2015 / arXiv 1506.05908*. arXiv:1506.05908 - [pdf](pdfs/piech2015/piech2015.pdf)  
  RNN models of student knowledge over exercise sequences. Framework for the Duolingo-type log data (single-attempt).
- **Settles, Meeder (2016)**. A trainable spaced repetition model for language learning. *ACL 2016, pp. 1848-1858*. ACL P16-1174 - [pdf](pdfs/settles2016/settles2016.pdf)  
  Half-life regression on Duolingo data: models forgetting (recall probability decays with time since practice). Retention, not error revision.
- **Settles, Brust, Gustafson, Hagiwara, Madnani (2018)**. Second language acquisition modeling. *BEA workshop 2018, pp. 56-65*. ACL W18-0506 - [pdf](pdfs/settles2018/settles2018.pdf)  
  Duolingo SLAM shared task: predict future errors of learners from their history (7M+ words, 6k learners in the paper; the call described a smaller set). Word-level errors, one attempt per exercise.

## data / benchmarks

- **Chollet (2019)**. On the measure of intelligence. *arXiv 1911.01547*. arXiv:1911.01547 - [pdf](pdfs/chollet2019/chollet2019.pdf)  
  Intelligence = skill-acquisition efficiency; introduces ARC. The standard we are measured against.
- **LeGris, Vong, Lake, Gureckis (2024)**. H-ARC: a robust estimate of human performance on the Abstraction and Reasoning Corpus benchmark. *arXiv 2409.01374; Scientific Data 12:1380 (2025)*. arXiv:2409.01374 - [pdf](pdfs/legris2024/legris2024.pdf)  
  Our main human dataset: 1,729 people, 800 tasks, all actions and attempts. 76.2% on training tasks.

## cognitive architectures

- **Anderson, Bothell, Byrne, Douglass, Lebiere, Qin (2004)**. An integrated theory of the mind. *Psychological Review 111(4):1036-1060*. doi:10.1037/0033-295X.111.4.1036 - citation only  
  ACT-R: modules (declarative memory, procedural rules, goal, perception, motor) coordinated by production rules; learning = chunk strengthening with activation + utility learning from reward. A tested human-learning-curve model; a candidate source for the synthetic human learner.
- **Laird, Lebiere, Rosenbloom (2017)**. A standard model of the mind. *AI Magazine 38(4):13-26 (open access, CC BY)*. doi:10.1609/aimag.v38i4.2744 - [page](https://ojs.aaai.org/index.php/aimagazine/article/view/2744)  
  Proposes a common architecture shared by ACT-R, Soar and others. Useful checklist of components STML lacks (working memory, long-term declarative + procedural memory, learning from reward).
- **Laird, Newell, Rosenbloom (1987)**. SOAR: an architecture for general intelligence. *Artificial Intelligence 33:1-64*. AIJ 33 - citation only  
  Soar: problem solving by search in problem spaces; impasses trigger subgoals; 'chunking' turns resolved impasses into new rules. Impasse-driven learning is the machine version of productive failure / the Socratic question.

## language learning

- **Saffran, Aslin, Newport (1996)**. Statistical learning by 8-month-old infants. *Science 274(5294):1926-1928*. PMID 8943209 - [page](https://pubmed.ncbi.nlm.nih.gov/8943209/)  
  Infants segment words from a continuous nonsense-syllable stream after ~2 minutes using only transition statistics, with no instruction, reward or feedback. Language structure is learned without explicit error feedback.
- **Kuhl (2004)**. Early language acquisition: cracking the speech code. *Nature Reviews Neuroscience 5:831-843*. doi:10.1038/nrn1533 - citation only  
  Infants combine statistical pattern detection with social interaction; social context gates speech learning (Kuhl 2007). Learning signals in language are partly social.
- **Newport (1990)**. Maturational constraints on language learning. *Cognitive Science 14:11-28*. Cog Sci 14 - citation only  
  Learning capacity changes with age: later exposure gives poorer outcomes. A reason to treat 'human learning' as depending on the learner's stage.
- **Hickok, Poeppel (2007)**. The cortical organization of speech processing. *Nature Reviews Neuroscience 8:393*. NRN 8:393 - citation only  
  Dual-stream model: a ventral stream maps sound to meaning, a dorsal stream maps sound to articulation. Two coupled pathways for comprehension and production.
- **Friederici (2011)**. The brain basis of language processing: from structure to function. *Physiological Reviews 91(4):1357-1392*. doi:10.1152/physrev.00006.2011 - citation only  
  Syntactic processing relies on left-lateralised temporal and inferior frontal networks; semantic processing on less lateralised temporo-frontal networks.
- **Chang, Dell, Bock (2006)**. Becoming syntactic. *Psychological Review 113(2):234-272*. doi:10.1037/0033-295X.113.2.234 - citation only  
  A connectionist model that learns syntax by predicting the next word and adjusting on prediction error (my recollection; the search did not confirm the abstract - verify before relying on it).
- **Reber (1967)**. Implicit learning of artificial grammars. *Journal of Verbal Learning and Verbal Behavior 6:855-863*. JVLVB 6 - citation only  
  People classify strings from an artificial finite-state grammar above chance without being able to state the rules. The classic paradigm for an artificial-language version of the game.

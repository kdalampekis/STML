# STML: Phase 1 core

**Start with `CLAUDE.md`**: the full project brief (idea, hypotheses, architecture, datasets,
results so far, goals, how to run on the Mac with Metal). Setup: `bash scripts/setup_mac.sh`.

Code for the STML research proposal: learning how humans learn, by logging
how people discover logical rules from mistakes and Socratic feedback, then
training machines on those learning histories.

## The domain

Classical categorical logic (syllogisms / basic set theory) over invented
kinds ("wugs", "daxes"). Each world has a hidden population; the learner sees
a few true premises and proposes statements that **must** follow. The four
statement forms are the classical A/E/I/O:

| Form | Example | Set-theory reading |
| --- | --- | --- |
| all | All wugs are daxes | W ⊆ D |
| no | No wugs are daxes | W ∩ D = ∅ |
| some | Some wugs are daxes | W ∩ D ≠ ∅ |
| some_not | Some wugs are not daxes | W \ D ≠ ∅ |

Every kind is non-empty (existential import). Why this domain: it has exact
answers, it is what logic courses teach, people make well-documented errors in
it (conversion, undistributed middle, ...), and nonsense names remove prior
knowledge, so success has to come from reasoning.

## What is in here

```
stml/logic.py         claims, exact region semantics, entailment, small models
stml/proof.py         named-rule proofs (Barbara, Celarent, Darii, Ferio, Baroco, Bocardo)
stml/world.py         hidden-world generator and the 5-level curriculum
stml/critic.py        Socratic critic: status, minimal premises, proof or countermodel, question
stml/referee.py       SAT referee (PySAT + CaDiCaL): certified verdicts, unsat cores, countermodels
stml/metacog.py       four epistemic answers, honesty reward, calibration metrics, test-round probes
stml/env.py           Gym-style env: propose claims (H1, H2) or judge them with four answers (H3)
stml/analysis.py      classify wrong claims into classical reasoning errors
stml/episodes.py      the fixed 15 worlds shared by people and agents
stml/trajectories.py  load game logs into learning trajectories + learning-curve summary
stml/agents.py        random and oracle reference agents
stml/datasets/        one format (stml-traj/1) for human learning data from any source:
                      H-ARC converter, Wugs and Daxes converter, domain-free learning profile
game/template.html    the human pilot game (Wugs and Daxes)
scripts/build_game.py builds game/index.html with every critique precomputed
scripts/play_local.py  play on your own computer; every move is saved to logs/
scripts/analyze_log.py learning curve, error types and test-round metrics from logs
scripts/convert_datasets.py convert H-ARC or game logs to stml-traj/1 and print a learning profile
stml/model/           the STML learner: object/concept graphs, relational GNN, history transformer, training
stml/runlog.py        every run gets runs/<time>_<name>/ with config, log, metrics, results, report
experiments/          harc_h1_stml.py (our architecture), baseline_v1/ (first experiment)
scripts/setup_mac.sh  one-time macOS setup (venv, Metal check)
scripts/download_arctraj.py  ARCTraj from Hugging Face (run on the Mac)
tests/                core logic, datasets, model
```

Key guarantee: `logic.py` decides truth exactly, and the named rules in
`proof.py` are tested to be sound and complete against it, so every correct
answer comes with a step-by-step proof and every non-consequence with a
concrete countermodel.

## Use

```bash
bash scripts/setup_mac.sh          # macOS: venv + requirements + Metal check
python -m pytest -q               # run tests
python scripts/build_game.py      # rebuild the game after changing levels or critic text
python scripts/play_local.py      # play at http://127.0.0.1:8000, logs saved to logs/<code>/
python scripts/analyze_log.py logs/pilot-01   # learning curve + test-round metrics
```

Agents:

```python
from stml.env import STMLEnv
from stml.agents import RandomAgent, run_episode
env = STMLEnv(level=3, feedback="diagnostic")   # or "scalar"
print(run_episode(env, RandomAgent(0), seed=1000 * 3))
```

## Logs: where your play ends up

| How you play | Where the log goes |
| --- | --- |
| The published page on claude.ai | The page's private research store, `runs/<participant>/logs/` (only the owner reads all) |
| `python scripts/play_local.py` | Automatically in `logs/<participant code>/`, one JSON file per world |
| `game/index.html` opened directly | Nowhere automatically: press **Download log** (top right) and put the file in `logs/` |

All three use the `stml-log/1` schema described in `stml/trajectories.py`.
Worlds are rebuilt from `(level, seed)`, so a log is all you need to replay and
analyse a session, or to train on it.

## The SAT referee

The referee re-checks every verdict with an independent encoding (individuals
x kinds, one witness per existential statement) and returns a certificate:
an unsat core (which premises a verdict needs) or a countermodel. It shares no
code with the critic, and the learner never calls it.

## Public human-learning data

`stml-traj/1` treats every source as the same loop: attempt, commit, feedback,
revision, explanation. To bring in H-ARC (1,729 people solving ARC puzzles with
up to 3 attempts and right/wrong feedback):

1. Clone https://github.com/Le-Gris/h-arc (task files in `arc_data/`).
2. Download the data archive from https://osf.io/bh8yq and unzip it there
   (`data.csv`, `summary_data.csv`).
3. `python scripts/convert_datasets.py harc --data h-arc/data --arc h-arc/arc_data`

Game logs: `python scripts/convert_datasets.py game logs/pilot-01`.
ARCTraj and Sudoku-Bench converters come next, once their files are in hand.

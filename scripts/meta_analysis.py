"""Meta-analysis of the STML experiment programme (2026-10-14 .. 2026-10-20).

Reads nothing from the run folders: the effects below were transcribed by hand from CLAUDE.md section 7 (each row names its source experiment), so this script documents and summarises
the programme; it does not recompute anything. Outputs (in a new run folder via stml.runlog.Run): key_effects.csv, forest.png, prediction_scorecard.csv, report.md.
Units: accuracy points at the late claims (logic) or points solved-in-3 (ARC), differences between conditions; 'sd' = sd of the paired difference over seeds / splits (blank when not recorded).
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stml.runlog import Run  # noqa: E402

# (domain, source, label, effect, sd, n, theme) ; theme: lever / small / null / harm
EFFECTS = [
    ("logic", "Q8", "verdict feedback vs none (late claims)", 14.4, 0.7, 3, "lever"),
    ("logic", "Q8", "verdict vs binary right/wrong (600 worlds)", 11.5, 0.3, 3, "lever"),
    ("logic", "Q35", "verdict vs binary, 10 re-masked draws", 7.8, 0.4, 3, "lever"),
    ("logic", "Q8", "binary vs no feedback (600 worlds)", 2.8, 1.1, 3, "small"),
    ("logic", "Q35", "binary vs no feedback, 10 re-masked draws", 8.0, 0.8, 3, "lever"),
    ("logic", "Q1", "ask by uncertainty vs random", 5.3, 0.6, 3, "lever"),
    ("logic", "Q9", "...replicated on 5 fresh test sets", 4.7, 0.2, 3, "lever"),
    ("logic", "Q14", "...with 20% wrong verdicts", 3.5, 0.7, 3, "lever"),
    ("logic", "Q10", "learned (RL) asking policy vs hand rule", -0.2, 0.2, 3, "null"),
    ("logic", "Q10", "structure-only asking vs random", 0.3, 0.1, 3, "null"),
    ("logic", "Q17", "re-mask x3 vs one episode per world", 3.4, 0.9, 3, "lever"),
    ("logic", "Q17b", "re-mask x10 vs equal-steps baseline", 7.6, 0.7, 3, "lever"),
    ("logic", "Q17b", "10x more steps, no diversity", -1.3, 0.5, 3, "harm"),
    ("logic", "Q18", "300 worlds x30 re-masks vs 1000 distinct worlds", 3.6, 1.0, 3, "lever"),
    ("logic", "Q4b", "train on target size (8-kind chains, accuracy)", 19.1, None, 3, "lever"),
    ("logic", "Q5", "3.3x more training worlds", 3.8, None, 3, "lever"),
    ("logic", "Q16", "auxiliary hidden-premise loss", 0.3, 0.8, 3, "null"),
    ("logic", "Q19", "randomised visible fraction (test at 50%)", -0.7, 0.5, 3, "null"),
    ("logic", "Q20", "wider network (128 vs 64)", 0.1, 0.1, 3, "null"),
    ("logic", "Q20", "deeper network (6 vs 4 layers)", -0.2, 0.5, 3, "null"),
    ("logic", "Q26", "test-time training, lr 3e-4", -2.5, 0.7, 3, "harm"),
    ("logic", "Q26b", "test-time training, lr 1e-5", -0.1, 0.3, 3, "null"),
    ("logic", "Q6b", "surprise-gated memory vs random memory", 2.2, 1.4, 8, "small"),
    ("logic", "Q12", "...with 20% wrong verdicts", -0.9, 1.8, 4, "null"),
    ("logic", "Q11b", "noise-trained vs clean-trained learner (20% noise)", 2.1, 0.3, 3, "small"),
    ("logic", "Q22", "leave-one-out filter on noisy feedback", 2.3, 0.4, 3, "small"),
    ("logic", "Q23", "'suspect' flag on noisy feedback", 0.6, 0.4, 3, "null"),
    ("logic", "Q15", "lemma scratchpad, 16-kind chains (vs round 0)", 4.8, 2.2, 3, "small"),
    ("logic", "Q15b", "lemma scratchpad, 32-kind chains (vs plain)", 1.1, 0.8, 3, "null"),
    ("logic", "Q39", "ensemble-agreement lemmas, 32-kind chains", -0.4, 4.4, 3, "null"),
    ("logic", "Q2/Q3", "signed surprise as input", 0.4, 1.1, 3, "null"),
    ("logic", "Q5/Q8b", "'which premises were used' marks", 0.2, 1.2, 3, "null"),
    ("logic", "P1", "surprise loss-weighting vs plain", 0.0, 1.5, 3, "null"),
    ("logic", "P1", "prioritised replay by surprise (small N)", -9.0, 2.0, 3, "harm"),
    ("arc", "v2", "human histories vs answers (imitation only, pure)", -4.3, None, 5, "harm"),
    ("arc", "Q27", "augmentation, human-history imitation", 5.9, 1.7, 3, "lever"),
    ("arc", "Q27", "4x longer training, no diversity", -1.1, 1.5, 3, "null"),
    ("arc", "Q28", "augmentation, RL from scratch", -2.1, 1.4, 3, "harm"),
    ("arc", "Q31", "humans-then-RL vs RL from scratch", 9.8, 2.6, 3, "lever"),
    ("arc", "Q29-33", "humans-then-RL vs answers-then-RL (12 aug. splits)", 3.4, 2.6, 12, "small"),
    ("arc", "Q32", "...without augmentation (5 splits)", 3.0, 4.1, 5, "small"),
    ("arc", "Q31", "humans-then-RL vs RL-agent-history-then-RL", 4.6, 1.6, 3, "small"),
    ("arc", "Q33", "histories shuffled vs full", -19.7, 3.0, 3, "harm"),
    ("arc", "Q33", "revisions only vs first attempts only", -0.7, 1.8, 3, "null"),
    ("arc", "Q34", "hybrid learner: humans-then-RL vs answers-then-RL", 0.8, 0.9, 3, "null"),
    ("arc", "Q36", "masked-node pretraining (humans-then-RL)", -1.2, 0.5, 3, "null"),
    ("arc", "Q37", "puzzle-level pretext, RL from scratch", 4.5, 2.6, 3, "lever"),
    ("arc", "Q37", "...humans-then-RL vs answers-then-RL after pretext", 0.4, 1.9, 3, "null"),
    ("arc", "Q38", "25% of human steps: humans vs answers", 0.1, 4.7, 3, "null"),
    ("arc", "Q38", "10% of human steps: humans vs answers", -1.1, 1.5, 3, "null"),
    ("arc", "Q40", "+ARCTraj steps: humans vs answers", 3.3, 6.1, 3, "null"),
]

# (experiment, predictions held, predictions made), scored by hand from the queue / CLAUDE.md; 'mostly/half' counted as 0.5
SCORE = [("Q1", 2, 3), ("Q1b", 2, 2), ("Q3", 2.5, 3), ("Q3b", 2, 3), ("Q4", 3, 4), ("Q4b", 1, 3), ("Q5", 1, 2), ("Q6", 2.5, 3), ("Q6b", 1, 1), ("Q8", 3, 3), ("Q8b", 1, 2),
         ("Q9", 2.5, 4), ("Q9c", 0, 1), ("Q10", 2, 2), ("Q11", 0.5, 2), ("Q11b", 2, 2), ("Q12", 1, 1), ("Q13", 2, 2), ("Q13b", 1, 1), ("Q14", 2, 2), ("Q15", 1.5, 4),
         ("Q15b", 2, 3), ("Q16", 1, 3), ("Q17", 1, 3), ("Q17b", 2, 2), ("Q18", 3, 3), ("Q19", 1, 2), ("Q20", 1, 2), ("Q21", 0, 2), ("Q22", 3, 3), ("Q23", 0, 2),
         ("Q24", 2, 3), ("Q25", 0.5, 3), ("Q26", 1, 2), ("Q26b", 1, 1), ("Q27", 3, 3), ("Q28", 0, 2), ("Q29", 1, 2), ("Q30", 1, 2), ("Q31", 2, 2), ("Q32", 1, 2),
         ("Q33", 2, 3), ("Q34", 1, 3), ("Q35", 0, 2), ("Q36", 0, 2), ("Q37", 2, 2), ("Q38", 1.5, 3), ("Q39", 1, 2), ("Q40", 0, 2)]

COLORS = {"lever": "#1b7837", "small": "#7fbf7b", "null": "#9e9e9e", "harm": "#c0392b"}


def forest(rows, title, path):
    rows = sorted(rows, key=lambda r: r[3])
    fig, ax = plt.subplots(figsize=(9.5, 0.33 * len(rows) + 1.6))
    for i, (_, src, label, eff, sd, n, theme) in enumerate(rows):
        ax.errorbar(eff, i, xerr=None if sd is None else 1.96 * sd / max(1, np.sqrt(n)) if False else (None if sd is None else sd), fmt="o", color=COLORS[theme], capsize=3, ms=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[2]} [{r[1]}]" for r in rows], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvspan(-1, 1, color="#eeeeee", zorder=0)
    ax.set_xlabel("difference in points (bar = sd over seeds / splits; grey band = run-to-run noise)")
    ax.set_title(title, fontsize=11)
    for k, v in COLORS.items():
        ax.plot([], [], "o", color=v, label={"lever": "clear gain", "small": "small / conditional gain", "null": "no effect", "harm": "harmful"}[k])
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    with Run("meta_analysis", {"n_effects": len(EFFECTS), "n_experiments_scored": len(SCORE)}) as run:
        with open(run.dir / "key_effects.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["domain", "source", "comparison", "effect_points", "sd", "n", "theme"])
            w.writerows(EFFECTS)
        for dom, title in (("logic", "Logic domain (hidden-premise learner): effects at the late claims"), ("arc", "ARC domain (pure STML learner, solved in 3 attempts)")):
            forest([r for r in EFFECTS if r[0] == dom], title, run.dir / f"forest_{dom}.png")
            shutil.copy(run.dir / f"forest_{dom}.png", ROOT / "docs/figures" / f"forest_{dom}.png")
        held = sum(h for _, h, _ in SCORE)
        made = sum(m for _, _, m in SCORE)
        with open(run.dir / "prediction_scorecard.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["experiment", "held", "made"])
            w.writerows(SCORE)
        themes = {t: sum(1 for r in EFFECTS if r[6] == t) for t in COLORS}
        noise = [r[4] for r in EFFECTS if r[4] is not None]
        big = [r for r in EFFECTS if r[6] == "lever"]
        L = ["# Meta-analysis of the experiment programme", "",
             f"{len(SCORE)} experiments with written predictions; {len(EFFECTS)} headline comparisons transcribed from CLAUDE.md section 7 (see the script for the source of each row).", "",
             "## 1. How good were the predictions?", "",
             f"Across {len(SCORE)} experiments, {made:.0f} predictions were written before running; {held:.1f} held ('mostly' and 'half' counted 0.5): **{100 * held / made:.0f}%**.",
             "By half: " + ", ".join(f"{a}: {100 * sum(h for e, h, _ in part) / sum(m for _, _, m in part):.0f}%" for a, part in (("first 25 experiments", SCORE[:25]), ("last 25", SCORE[25:]))) + ".",
             "Interpretation: the programme was mostly exploratory, so predicted effects often failed (a failed prediction is information, not a failed experiment). The hit rate is far from perfect, which is also a calibration warning: the analyst's intuition about which tricks help was wrong about half of the time.", "",
             "## 2. What kind of effects did we find?", "",
             f"Of {len(EFFECTS)} headline comparisons: {themes['lever']} clear gains, {themes['small']} small or conditional gains, {themes['null']} no effect (inside noise), {themes['harm']} harmful.",
             f"Median sd of a paired difference over seeds / splits: {np.median(noise):.1f} points (iqr {np.percentile(noise, 25):.1f}-{np.percentile(noise, 75):.1f}); single runs of the same condition differ by about 1 point in logic and about 3 points on ARC.",
             f"Median size of a clear gain: {np.median([abs(r[3]) for r in big]):.1f} points; largest: " + ", ".join(f"{r[2]} ({r[3]:+.1f})" for r in sorted(big, key=lambda r: -r[3])[:3]) + ".", "",
             "## 3. Figures", "", "![logic](../docs/figures/forest_logic.png)", "", "![arc](../docs/figures/forest_arc.png)", ""]
        run.finish({"held": held, "made": made, "themes": themes}, "\n".join(L) + "\n")
        shutil.copy(run.dir / "report.md", ROOT / "docs" / "META_ANALYSIS_numbers.md")


if __name__ == "__main__":
    main()

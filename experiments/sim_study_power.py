"""Sizing the human study with synthetic human-like participants (stml/synthetic.py).

Question: how many participants does the Wugs-and-Daxes study need to detect the effects we care about?
  E3  feedback arm       diagnostic vs scalar feedback: accuracy on the final (Transfer) test round, 12 statements
                          Welch t-test between arms, completers only
  E2  hypercorrection    after an error made with high confidence (4-5) vs low confidence (1-2): is the next error in the
                          same world of a DIFFERENT type more often?  (as the game is now: consecutive errors, ~5 per bin)
  E2b hypercorrection, direct re-test (a DESIGN OPTION: each error type is re-presented later; ~7 per bin)
  E1  metacognition       participant-level AUROC of confidence for correctness on the test rounds, vs 0.5

Everything here rests on ASSUMED effect sizes: there is no human data yet. The experiment is therefore a sensitivity
analysis over a grid of effects (diag_gain: how much better the critique teaches; hc: how strongly confident errors are
corrected). Read it as "N needed IF the effect is this big", and recalibrate once the first real logs exist.

Method: simulate a large pool of participants per scenario, then draw studies of N participants by resampling the pool
(2000 draws per cell). Completion rate 80.6% (1393 of 1729 H-ARC participants finished): enrolled = completers / 0.806.

    python experiments/sim_study_power.py --quick
    python experiments/sim_study_power.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.synthetic import Population, endpoints, game_worlds, simulate_participant  # noqa: E402
from stml.runlog import Run  # noqa: E402

COMPLETION = 1393 / 1729
NS = (10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300)
HCS = (0.0, 1.0, 2.0)
GAINS = (0.35, 0.5, 0.7, 1.0)          # 0.35 = no benefit of diagnostic feedback (null)
ALPHAS = (0.05, 0.05 / 3)


def pool(arm: str, gain: float, hc: float, m: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pop = Population(hc=hc)
    rows = [endpoints(simulate_participant(rng, arm, pop, gain)) for _ in range(m)]
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


def welch_p(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two-sided Welch p-values for each row (draws x N)."""
    na, nb = a.shape[1], b.shape[1]
    va, vb = a.var(1, ddof=1) / na, b.var(1, ddof=1) / nb
    t = (b.mean(1) - a.mean(1)) / np.sqrt(va + vb + 1e-12)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1) + 1e-18)
    return 2 * stats.t.sf(np.abs(t), df)


def onesample_p(x: np.ndarray, mu: float = 0.0) -> np.ndarray:
    """Two-sided one-sample t p-values per row, ignoring NaN (participants without data drop out)."""
    n = np.sum(~np.isnan(x), 1)
    mean = np.nansum(x, 1) / np.maximum(n, 1)
    sd = np.sqrt(np.nansum((x - mean[:, None]) ** 2, 1) / np.maximum(n - 1, 1))
    t = (mean - mu) / (sd / np.sqrt(np.maximum(n, 1)) + 1e-12)
    p = 2 * stats.t.sf(np.abs(t), np.maximum(n - 1, 1))
    return np.where(n >= 5, p, 1.0)


def power_two_arm(sc: np.ndarray, dg: np.ndarray, n: int, draws: int, rng) -> tuple[float, float]:
    ia = rng.integers(len(sc), size=(draws, n))
    ib = rng.integers(len(dg), size=(draws, n))
    p = welch_p(sc[ia], dg[ib])
    return tuple(float((p < a).mean()) for a in ALPHAS)


def power_within(x: np.ndarray, n: int, draws: int, rng, mu: float = 0.0) -> tuple[float, float]:
    p = onesample_p(x[rng.integers(len(x), size=(draws, n))], mu)
    return tuple(float((p < a).mean()) for a in ALPHAS)


def first_n(curve: dict[int, float], target: float):
    for n in sorted(curve):
        if curve[n] >= target:
            return n
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=int, default=4000, help="simulated participants per scenario")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.pool, a.draws = 300, 200
    with Run("sim_study_power" + ("_quick" if a.quick else ""), {**vars(a), "ns": NS, "hcs": HCS, "gains": GAINS,
                                                               "completion": COMPLETION}) as run:
        worlds = game_worlds()
        run.log(f"game: {len(worlds)} episodes, {sum(len(w.probes) for w in worlds)} test statements, "
                f"{sum(len(w.targets) for w in worlds)} targets")
        rng = np.random.default_rng(a.seed + 1)
        pools: dict[tuple, dict] = {}
        for hc in HCS:
            pools[("scalar", None, hc)] = pool("scalar", 1.0, hc, a.pool, a.seed + int(10 * hc))
            for g in GAINS:
                pools[("diagnostic", g, hc)] = pool("diagnostic", g, hc, a.pool, a.seed + 100 + int(10 * hc) + int(100 * g))
            run.log(f"hc={hc}: pools simulated")

        def done(d, key):      # completers only for the final test
            return d[key][d["completed"] == 1]

        results: dict = {"scenarios": {}}
        for hc in HCS:
            for g in GAINS:
                S, D = pools[("scalar", None, hc)], pools[("diagnostic", g, hc)]
                sc, dg = done(S, "acc_final"), done(D, "acc_final")
                eff = float(dg.mean() - sc.mean())
                sd = math.sqrt((sc.var(ddof=1) + dg.var(ddof=1)) / 2)
                curve05, curve_b = {}, {}
                for n in NS:
                    curve05[n], curve_b[n] = power_two_arm(sc, dg, n, a.draws, rng)
                # hypercorrection and metacognition: participants from both arms (half each), all participants incl. dropouts
                both = {k: np.concatenate([S[k], D[k]]) for k in S}
                rep = both["repeat_low"] - both["repeat_high"]
                ret = both["retest_low"] - both["retest_high"]
                e2 = {n: power_within(rep, n, a.draws, rng)[0] for n in NS}
                e2b = {n: power_within(ret, n, a.draws, rng)[0] for n in NS}
                e1 = {n: power_within(both["auroc"], n, a.draws, rng, mu=0.5)[0] for n in NS}
                results["scenarios"][f"hc={hc}|gain={g}"] = {
                    "hc": hc, "gain": g, "e3_effect_points": 100 * eff, "e3_cohens_d": eff / sd,
                    "e3_power": curve05, "e3_power_bonf": curve_b,
                    "e2_effect_points": 100 * float(np.nanmean(rep)), "e2_power": e2,
                    "e2b_effect_points": 100 * float(np.nanmean(ret)), "e2b_power": e2b,
                    "e1_auroc": float(np.nanmean(both["auroc"])), "e1_power": e1,
                    "summary": {"acc_all_scalar": float(np.nanmean(S["acc_all"])), "acc_final_scalar": float(sc.mean()),
                                "acc_final_diag": float(dg.mean()), "abstain_scalar": float(np.nanmean(S["abstain"])),
                                "errors_per_person": float(np.nanmean(both["n_errors"])),
                                "pairs_low": float(np.nanmean(both["n_low"])), "pairs_high": float(np.nanmean(both["n_high"])),
                                "retests_low": float(np.nanmean(both["n_rt_low"])), "retests_high": float(np.nanmean(both["n_rt_high"]))}}
            run.log(f"hc={hc}: power computed")

        # ---------------------------------------------------------------- report
        L = ["# Sizing the human study with synthetic participants", "",
             "**Everything below is conditional on assumed effect sizes** (no human data yet). Synthetic participants play the real "
             f"game ({len(worlds)} episodes, {sum(len(w.probes) for w in worlds)} test statements, real critic and error taxonomy); "
             f"pool {a.pool} per scenario, {a.draws} resampled studies per cell, completion {COMPLETION:.1%} "
             "(share of H-ARC participants who finished). Alpha 0.05 unless stated.", ""]
        base = results["scenarios"]["hc=1.0|gain=0.35"]["summary"]
        L += ["## Population (calibration targets, scalar arm)", "",
              f"overall test accuracy {base['acc_all_scalar']:.1%}, final-test accuracy {base['acc_final_scalar']:.1%}, "
              f"abstention {base['abstain_scalar']:.1%}, errors per person {base['errors_per_person']:.1f}. "
              "These are my assumptions; compare with the first real logs and refit `stml/synthetic.py: Population`.", ""]
        L += ["## E3: diagnostic vs scalar feedback (final test, completers per arm)", "",
              "Effect = difference in accuracy on the 12 final test statements. `gain 0.35` = no benefit (type-I error check).", "",
              "| diag gain | effect (points) | Cohen d | " + " | ".join(f"n={n}" for n in NS) + " | n/arm for 80% | for 90% | enrolled/arm for 80% |",
              "| ---: | ---: | ---: | " + " | ".join("---:" for _ in NS) + " | ---: | ---: | ---: |"]
        for g in GAINS:
            r = results["scenarios"][f"hc=1.0|gain={g}"]
            n80, n90 = first_n(r["e3_power"], 0.8), first_n(r["e3_power"], 0.9)
            enr = "-" if n80 is None else str(math.ceil(n80 / COMPLETION))
            L.append(f"| {g} | {r['e3_effect_points']:+.1f} | {r['e3_cohens_d']:+.2f} | " + " | ".join(f"{r['e3_power'][n]:.2f}" for n in NS)
                     + f" | {n80 or '>300'} | {n90 or '>300'} | {enr if n80 else '>' + str(math.ceil(300 / COMPLETION))} |")
        L += ["", "(Power in the `gain 0.35` row should be about 0.05.)", ""]
        L += ["## E2: hypercorrection, as the game is now (consecutive errors in a world)", "",
              "Effect = how much more often a low-confidence error is repeated (same type next) than a high-confidence one. "
              "Participants from both arms, all participants (dropouts included); total N.", "",
              "| hc | effect (points) | pairs/person low / high | " + " | ".join(f"N={n}" for n in NS) + " | N for 80% |",
              "| ---: | ---: | ---: | " + " | ".join("---:" for _ in NS) + " | ---: |"]
        for hc in HCS:
            r = results["scenarios"][f"hc={hc}|gain=0.7"]
            n80 = first_n(r["e2_power"], 0.8)
            L.append(f"| {hc} | {r['e2_effect_points']:+.1f} | {r['summary']['pairs_low']:.1f} / {r['summary']['pairs_high']:.1f} | "
                     + " | ".join(f"{r['e2_power'][n]:.2f}" for n in NS) + f" | {n80 or '>300'} |")
        L += ["", "## E2b: hypercorrection with a direct re-test (design option)", "",
              "Each error type is re-presented at the end of the next episode; outcome = fell for it again.", "",
              "| hc | effect (points) | re-tests/person low / high | " + " | ".join(f"N={n}" for n in NS) + " | N for 80% |",
              "| ---: | ---: | ---: | " + " | ".join("---:" for _ in NS) + " | ---: |"]
        for hc in HCS:
            r = results["scenarios"][f"hc={hc}|gain=0.7"]
            n80 = first_n(r["e2b_power"], 0.8)
            L.append(f"| {hc} | {r['e2b_effect_points']:+.1f} | {r['summary']['retests_low']:.1f} / {r['summary']['retests_high']:.1f} | "
                     + " | ".join(f"{r['e2b_power'][n]:.2f}" for n in NS) + f" | {n80 or '>300'} |")
        r = results["scenarios"]["hc=1.0|gain=0.7"]
        L += ["", "## E1: is confidence informative about correctness (AUROC > 0.5)?", "",
              f"Mean participant AUROC in the default scenario: {r['e1_auroc']:.3f}.", "",
              "| " + " | ".join(f"N={n}" for n in NS) + " |", "| " + " | ".join("---:" for _ in NS) + " |",
              "| " + " | ".join(f"{r['e1_power'][n]:.2f}" for n in NS) + " |", "",
              "## Reading this", "",
              "- A planned 30-50 pilot only detects a large feedback effect; the 200-300 target is where small-to-medium ones become detectable. "
              "The table gives the number per arm for the assumed effect; halve the assumed effect and the number roughly quadruples.",
              "- If hypercorrection is detectable at all in the current design depends on how many errors each person makes and how strong the "
              "effect is; compare E2 with E2b to see what a direct re-test would buy.",
              "- The simulation cannot say whether the real effects are this big. Its use is to show which of the planned analyses are "
              "decided by N and which by design.", ""]
        _plots(results, run.dir)
        run.finish(results, "\n".join(L))


def _plots(results: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for g in GAINS:
        r = results["scenarios"][f"hc=1.0|gain={g}"]
        ax[0].plot(NS, [r["e3_power"][n] for n in NS], marker="o", label=f"gain {g}: {r['e3_effect_points']:+.1f} pts")
    ax[0].set_title("E3 feedback arm: power vs completers per arm")
    for hc in HCS:
        r = results["scenarios"][f"hc={hc}|gain=0.7"]
        ax[1].plot(NS, [r["e2_power"][n] for n in NS], marker="o", label=f"hc {hc}: {r['e2_effect_points']:+.1f} pts")
        ax[2].plot(NS, [r["e2b_power"][n] for n in NS], marker="o", label=f"hc {hc}: {r['e2b_effect_points']:+.1f} pts")
    ax[1].set_title("E2 hypercorrection (current design): total N")
    ax[2].set_title("E2b with a direct re-test: total N")
    for x in ax:
        x.axhline(0.8, color="grey", ls="--", lw=0.8)
        x.set_xscale("log")
        x.set_ylim(0, 1.02)
        x.set_xlabel("participants")
        x.set_ylabel("power")
        x.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "power_curves.png", dpi=130)


if __name__ == "__main__":
    main()

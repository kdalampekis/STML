"""Pilot sensitivity of the study-sizing simulation (stml/synthetic.py).

pilot-01 (ONE person, the owner, who knows the hypotheses; worlds L1-L3, 36 test statements, 18 of them 'open'): he answered 'must be false' on 9 of the 18 open claims ('negation as failure'),
'I don't know yet' on 1 of 36 (3%), and was right on 61% of the statements. The synthetic population assumed no such behaviour (open claims were answered 'open' or lured to 'true') and about 20%
'don't know yet'. This script asks how much the simulated quantities and the sizing change if the population behaved like the pilot. It is a SENSITIVITY CHECK, not an estimate: N = 1.

Scenarios: baseline (naf 0, tau 0.15); naf 0.5 (tau 0.15); naf 0.5 + low abstention (tau 0.02).
Observed in the first 9 worlds (L1-L3 equivalent) for each scenario: accuracy, share 'don't know yet', accuracy on open claims, share of open claims judged 'false'. Then the number of completers needed for 80% power on E3
(diagnostic vs scalar feedback, gain 0.7, hc 1) with the Welch test on the final Transfer round.
PREDICTIONS, written before running:
  S1  the baseline population judges almost no open claim 'false' (< 5%); naf 0.5 gives about 50% (it is the parameter).
  S2  naf 0.5 lowers the simulated accuracy on open claims below 45% (the pilot: 44%) and lowers overall accuracy by 5-15 points relative to the baseline.
  S3  the completers needed for 80% power on E3 change by less than 25% (negation-as-failure adds the same noise to both arms).
  S4  low abstention (tau 0.02) brings the simulated 'don't know yet' share near 3-8%.

    python experiments/sim_pilot_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from sim_study_power import NS, first_n, power_two_arm  # noqa: E402
from stml.runlog import Run  # noqa: E402
from stml.synthetic import Population, endpoints, simulate_participant  # noqa: E402

SCEN = {"baseline": Population(), "naf0.5": Population(naf_prob=0.5), "naf0.5_lowtau": Population(naf_prob=0.5, tau_mean=0.02)}
PILOT = {"accuracy": 22 / 36, "dont_know": 1 / 36, "acc_open": 8 / 18, "false_on_open": 9 / 18}


def stats_first9(parts) -> dict:
    acc, dk, o_acc, o_false = [], [], [], []
    for p in parts:
        pr = [x for x in p.probes if x[0] < 9]
        if len(pr) < 36:
            continue
        acc.append(np.mean([x[4] for x in pr]))
        dk.append(np.mean([x[2] == "unknown" for x in pr]))
        op = [x for x in pr if x[1] == "open"]
        o_acc.append(np.mean([x[4] for x in op]))
        o_false.append(np.mean([x[2] == "false" for x in op]))
    return {"accuracy": float(np.mean(acc)), "dont_know": float(np.mean(dk)), "acc_open": float(np.mean(o_acc)), "false_on_open": float(np.mean(o_false))}


def main() -> None:
    with Run("sim_pilot_sensitivity", {"scenarios": list(SCEN), "pilot": PILOT}) as run:
        rng = np.random.default_rng(0)
        res, power = {}, {}
        for name, pop in SCEN.items():
            sc = [simulate_participant(np.random.default_rng(10 + i), "scalar", pop) for i in range(3000)]
            dg = [simulate_participant(np.random.default_rng(20_000 + i), "diagnostic", pop, 0.7) for i in range(3000)]
            res[name] = stats_first9(sc)
            ea = np.array([endpoints(p)["acc_final"] for p in sc if p.completed])
            eb = np.array([endpoints(p)["acc_final"] for p in dg if p.completed])
            curve = {n: power_two_arm(ea, eb, n, 1500, rng)[0] for n in NS}
            power[name] = {"effect": float(eb.mean() - ea.mean()), "sd": float(np.sqrt((ea.var(ddof=1) + eb.var(ddof=1)) / 2)), "n80": first_n(curve, 0.8)}
            run.log(f"{name}: {res[name]} | E3 effect {power[name]['effect']:+.3f}, n80 {power[name]['n80']}")
        L = ["# Pilot sensitivity of the study sizing", "", "pilot-01 is ONE person who knows the hypotheses: this is a sensitivity check, not an estimate.", "",
             "| quantity (first 9 worlds) | pilot-01 | " + " | ".join(SCEN) + " |", "| --- | ---: | " + " | ".join("---:" for _ in SCEN) + " |"]
        for k in PILOT:
            L.append(f"| {k} | {PILOT[k]:.2f} | " + " | ".join(f"{res[n][k]:.2f}" for n in SCEN) + " |")
        L += ["", "| E3 (diagnostic vs scalar, gain 0.7) | " + " | ".join(SCEN) + " |", "| --- | " + " | ".join("---:" for _ in SCEN) + " |",
              "| effect on the final test round (accuracy) | " + " | ".join(f"{power[n]['effect']:+.3f}" for n in SCEN) + " |",
              "| sd of the endpoint | " + " | ".join(f"{power[n]['sd']:.3f}" for n in SCEN) + " |",
              "| completers per arm for 80% power (alpha 0.05) | " + " | ".join(str(power[n]['n80']) for n in SCEN) + " |"]
        run.finish({"stats": res, "power": power}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

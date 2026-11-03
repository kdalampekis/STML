"""Analysis of Wugs and Daxes game logs (stml-log/1), planned in advance of the real data.

Per participant -> the same endpoints the study-sizing simulation uses (stml/synthetic.py: endpoints), computed from the log events; then group analyses:
  E1  is confidence informative about correctness on the test rounds (participant-level AUROC vs 0.5)
  E2  hypercorrection: after a high-confidence (4-5) vs low-confidence (1-2) error, how often is the NEXT error in the same world of the same type
  E3  feedback arm: diagnostic vs scalar, accuracy on the final (Transfer) test round, completers  (only if the session records `arm`)
  curves: probe accuracy and abstention across the 15 worlds; abstention by truth class and by proof size
  sizing update: the N per arm for several effect sizes given the OBSERVED between-person sd of the final test accuracy
  versus the simulation's assumptions: observed summary statistics next to the values stml/synthetic.py was calibrated to produce
Nothing here uses the logs for training. Inputs: log documents or files, as accepted by stml.trajectories.load_events.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from .synthetic import auroc, game_worlds
from .trajectories import load_events, to_trajectories

TRUTH_ANSWER = {"follows": "true", "contradicted": "false", "open": "open"}


def load_docs(paths) -> list[dict]:
    docs = []
    for p in map(Path, paths):
        for f in (sorted(p.rglob("*.json")) if p.is_dir() else [p]):
            obj = json.loads(f.read_text())
            docs.extend(obj if isinstance(obj, list) else [obj])
    return docs


def by_participant(docs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        pid = (d.get("session") or {}).get("participant")
        if pid is None:
            pid = next((e.get("participant") for e in d.get("events", []) if e.get("type") == "session_start"), "unknown")
        out[pid].append(d)
    return dict(out)


def participant_endpoints(docs: list[dict]) -> dict:
    events = load_events(docs)
    arm = next((d["session"].get("arm") for d in docs if (d.get("session") or {}).get("arm")), None) \
        or next((e.get("arm") for e in events if e.get("type") == "session_start" and e.get("arm")), None)
    trajs = to_trajectories(events)
    worlds = {w.eid: i for i, w in enumerate(game_worlds())}
    support = {(w.eid, p.key): p.support for w in game_worlds() for p in w.probes}
    probes = []                                                   # (world index, eid, truth, answer, confidence, correct, support)
    for tr in trajs:
        for e in tr.probes:
            probes.append((worlds.get(tr.episode, -1), tr.episode, e["truth"], e["answer"], e.get("confidence"),
                           e["answer"] == TRUTH_ANSWER[e["truth"]], support.get((tr.episode, e["claim"]), 0)))
    answered = [x for x in probes if x[4] is not None and x[3] != "unknown"]
    conf = np.array([x[4] for x in answered], dtype=float)
    right = np.array([x[5] for x in answered], dtype=float)
    final = [x for x in probes if x[1].startswith("L5-")]
    rep = {"low": [], "high": []}
    n_err = n_claims = 0
    for tr in trajs:
        wrong = [s for s in tr.steps if not s.correct and s.confidence is not None]
        n_claims += len(tr.steps)
        n_err += sum(1 for s in tr.steps if not s.correct)
        for a, b in zip(wrong, wrong[1:]):
            if a.confidence <= 2:
                rep["low"].append(a.errors[:1] == b.errors[:1])
            elif a.confidence >= 4:
                rep["high"].append(a.errors[:1] == b.errors[:1])
    finished = {tr.episode for tr in trajs if tr.reason in ("complete", "skip")}
    return {
        "arm": arm, "n_episodes": len(trajs), "completed": len(finished) >= 15,
        "n_probes": len(probes), "acc_all": float(np.mean([x[5] for x in probes])) if probes else float("nan"),
        "abstain": float(np.mean([x[3] == "unknown" for x in probes])) if probes else float("nan"),
        "auroc": auroc(conf, right) if len(answered) >= 8 else float("nan"),
        "acc_final": float(np.mean([x[5] for x in final])) if len(final) == 12 else float("nan"),
        "n_errors": n_err, "n_claims": n_claims,
        "repeat_low": float(np.mean(rep["low"])) if rep["low"] else float("nan"),
        "repeat_high": float(np.mean(rep["high"])) if rep["high"] else float("nan"),
        "probes": probes,
    }


def _t_ci(x: np.ndarray, mu: float = 0.0):
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return None
    t, p = stats.ttest_1samp(x, mu)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "ci": [float(x.mean() - half), float(x.mean() + half)], "p": float(p)}


def analyze(participants: dict[str, list[dict]]) -> dict:
    E = {pid: participant_endpoints(d) for pid, d in participants.items()}
    col = lambda k, keep=lambda e: True: np.array([e[k] for e in E.values() if keep(e)], dtype=float)
    done = lambda e: e["completed"]
    out: dict = {"n_participants": len(E), "n_completed": int(sum(e["completed"] for e in E.values())),
                 "arms": {a: int(sum(e["arm"] == a for e in E.values())) for a in sorted({e["arm"] for e in E.values() if e["arm"]})}}
    out["summary"] = {k: {"mean": float(np.nanmean(col(k))), "sd": float(np.nanstd(col(k)))} for k in ("acc_all", "abstain", "auroc", "n_errors")
                      if len(E) and not np.all(np.isnan(col(k)))}
    out["E1_metacognition"] = _t_ci(col("auroc"), 0.5)
    out["E2_hypercorrection"] = _t_ci(col("repeat_low") - col("repeat_high"), 0.0)
    arms = {a: col("acc_final", lambda e, a=a: e["arm"] == a and e["completed"]) for a in out["arms"]}
    arms = {a: v[~np.isnan(v)] for a, v in arms.items()}
    if "scalar" in arms and "diagnostic" in arms and len(arms["scalar"]) >= 3 and len(arms["diagnostic"]) >= 3:
        s, d = arms["scalar"], arms["diagnostic"]
        t, p = stats.ttest_ind(d, s, equal_var=False)
        sd = math.sqrt((s.var(ddof=1) + d.var(ddof=1)) / 2)
        out["E3_feedback_arm"] = {"n_scalar": len(s), "n_diagnostic": len(d), "mean_scalar": float(s.mean()), "mean_diagnostic": float(d.mean()),
                                  "diff_points": float(100 * (d.mean() - s.mean())), "cohens_d": float((d.mean() - s.mean()) / sd), "p": float(p), "sd": float(sd)}
    # curves over the 15 worlds
    curve = defaultdict(list)
    ab = defaultdict(list)
    by_truth = defaultdict(list)
    by_support = defaultdict(list)
    for e in E.values():
        per = defaultdict(list)
        for (wi, eid, truth, ans, conf, ok, sup) in e["probes"]:
            per[wi].append((ok, ans == "unknown"))
            by_truth[truth].append(ans == "unknown")
            if truth != "open":
                by_support[min(sup, 4)].append(ans == "unknown")
        for wi, v in per.items():
            curve[wi].append(float(np.mean([x[0] for x in v])))
            ab[wi].append(float(np.mean([x[1] for x in v])))
    out["curve"] = {"accuracy": {int(k): float(np.mean(v)) for k, v in sorted(curve.items())}, "abstain": {int(k): float(np.mean(v)) for k, v in sorted(ab.items())}}
    out["abstain_by_truth"] = {k: float(np.mean(v)) for k, v in by_truth.items()}
    out["abstain_by_support"] = {int(k): float(np.mean(v)) for k, v in sorted(by_support.items())}
    # sizing update from the observed between-person sd of the final test accuracy
    allfinal = col("acc_final", done)
    allfinal = allfinal[~np.isnan(allfinal)]
    if len(allfinal) >= 5:
        sd = float(allfinal.std(ddof=1))
        z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
        out["sizing"] = {"observed_sd_final_accuracy": sd, "n_per_arm_80pct": {str(pts): int(math.ceil(2 * (z * sd / (pts / 100)) ** 2)) for pts in (4, 6, 8, 10)}}
    out["participants"] = {pid: {k: v for k, v in e.items() if k != "probes"} for pid, e in E.items()}
    return out


def render(res: dict, sim_summary: dict | None = None) -> str:
    L = ["# Game-log analysis", "", f"{res['n_participants']} participants ({res['n_completed']} finished all 15 worlds); arms: {res['arms'] or 'not recorded'}.", ""]
    s = res["summary"]
    if s:
        L += ["| measure | observed mean (sd across people) |" + (" simulation assumed |" if sim_summary else ""), "| --- | ---: |" + (" ---: |" if sim_summary else "")]
        names = {"acc_all": "test-round accuracy", "abstain": "share answered 'don't know yet'", "auroc": "confidence AUROC for correctness", "n_errors": "errors per person"}
        for k, v in s.items():
            sim = f" {sim_summary.get(k, '-')} |" if sim_summary else ""
            L.append(f"| {names[k]} | {v['mean']:.3f} ({v['sd']:.3f}) |{sim}")
        L.append("")

    def fmt(name, r, unit=""):
        if r is None:
            return f"- {name}: not enough data yet"
        return f"- {name}: mean {r['mean']:+.3f}{unit} (95% CI {r['ci'][0]:+.3f} to {r['ci'][1]:+.3f}), n={r['n']}, p={r['p']:.3g}"
    e1 = res["E1_metacognition"]
    L += ["## Planned analyses", "",
          (f"- E1 confidence AUROC for correctness: {e1['mean']:.3f} (95% CI {e1['ci'][0]:.3f} to {e1['ci'][1]:.3f}), n={e1['n']}, p={e1['p']:.3g} against 0.5" if e1
           else "- E1 confidence AUROC: not enough data yet"),
          fmt("E2 hypercorrection (repeat rate after a low-confidence error minus after a high-confidence one)", res["E2_hypercorrection"])]
    e3 = res.get("E3_feedback_arm")
    L.append(f"- E3 feedback arm: diagnostic {e3['mean_diagnostic']:.1%} (n={e3['n_diagnostic']}) vs scalar {e3['mean_scalar']:.1%} (n={e3['n_scalar']}), "
             f"difference {e3['diff_points']:+.1f} points, d={e3['cohens_d']:+.2f}, p={e3['p']:.3g}" if e3 else "- E3 feedback arm: needs sessions that record `arm` for both arms")
    L += ["", "## Learning across the 15 worlds", "", "| world | " + " | ".join(str(k + 1) for k in res["curve"]["accuracy"]) + " |",
          "| --- | " + " | ".join("---:" for _ in res["curve"]["accuracy"]) + " |",
          "| test accuracy | " + " | ".join(f"{v:.0%}" for v in res["curve"]["accuracy"].values()) + " |",
          "| answered 'don't know yet' | " + " | ".join(f"{v:.0%}" for v in res["curve"]["abstain"].values()) + " |", "",
          "Abstention by truth: " + ", ".join(f"{k} {v:.0%}" for k, v in res["abstain_by_truth"].items()) + "; by proof size (follows/contradicted): "
          + ", ".join(f"{k}{'+' if k == 4 else ''} premises {v:.0%}" for k, v in res["abstain_by_support"].items()), ""]
    if "sizing" in res:
        z = res["sizing"]
        L += ["## Sizing update from the observed spread", "", f"Observed between-person sd of the final-test accuracy: {z['observed_sd_final_accuracy']:.3f}. "
              "Completers per arm for 80% power (alpha 0.05, two-sided) to detect a difference of: " + ", ".join(f"{k} points: {v}" for k, v in z["n_per_arm_80pct"].items()) + ".", ""]
    return "\n".join(L) + "\n"

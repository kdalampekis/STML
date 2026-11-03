"""H3 in the hidden-premise regime: does the learner know WHICH claims it can answer now?

Here "I don't know yet" has an exact meaning. A claim is KNOWABLE at step t if a proof of its verdict exists from what the learner has seen:
the visible premises plus the revealed claims that turned out true (or the negation of those that turned out contradicted). Otherwise the right
answer is "I don't know yet", whatever the true verdict. (stml/model/logic_episodes.py: knowable)
Models (same graph model on the same feedback graph): M_fb predicts the verdict; M_cert predicts whether the claim is knowable (a trained
'is a proof available' head, with exact labels). Gates for abstaining: confidence of M_fb; P(knowable) of M_cert; oracle (abstain iff not knowable).
Honest reward: +1 right, -lam wrong, -0.2 abstain (as in the H3 work); thresholds chosen on validation episodes.

PREDICTIONS, written before running:
  K1  M_fb confidence does NOT separate knowable from unknowable claims well (AUROC < 0.75): it was trained to guess, not to track proofs.
  K2  M_cert separates them clearly (AUROC > 0.9), and gating by it beats gating by confidence on the honest score at lam = 2 and 4.
  K3  the share of knowable claims, and answered coverage under the oracle gate, rise with the step t as feedback accumulates.
  Falsifier: M_cert AUROC within 0.05 of the confidence AUROC.
RESULT of the first run: K1 false (confidence AUROC for 'knowable' = 0.92), K2 half (head AUROC 1.000 but its gate LOSES to the confidence gate under the standard reward: the reward
pays lucky guesses; unknowable claims are still answered right ~70% of the time), K3 true.
POST-HOC addition (motivated by CLAUDE.md 4.5, 'a correct answer without a certificate is a lucky guess'): a luck-adjusted reward, answered+knowable+right = +1,
  answered+unknowable+right = 0, answered+wrong = -lam, abstain = -0.2; thresholds fitted on validation episodes for that reward.
  K4  under the luck-adjusted reward the certificate gate beats the confidence gate by >= 0.05 at lam = 2 and 4, and the oracle gate is the best policy.

    python experiments/logic_knowable.py --quick
    python experiments/logic_knowable.py
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import stml.model.logic_data as ld  # noqa: E402
from logic_h3_certificate import K, auroc, best_threshold, honest  # noqa: E402
from stml.model.logic_episodes import N_TYPES, episode_items, knowable, make_episodes  # noqa: E402
from stml.model.logic_judge import LogicJudge, fit, logits_of  # noqa: E402
from stml.model.network import pick_device  # noqa: E402
from stml.runlog import Run  # noqa: E402

BANDS = {"t=0": (0, 0), "t=1-3": (1, 3), "t=4-6": (4, 6), "t=7-9": (7, 9)}
LAMS = (1.0, 2.0, 4.0)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--worlds", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ld.SIZE_FREE = True
    nw, epochs, k = (a.worlds, 50, 100), 10, 10
    if a.quick:
        a.seeds, nw, epochs, k = 1, (24, 8, 12), 2, 6
    device = pick_device(None)
    levels = [2, 3, 4, 5]
    with Run("logic_knowable" + ("_quick" if a.quick else ""), {**vars(a), "k": k}) as run:
        tr = make_episodes(levels, range(nw[0]), k, 0.5, rng_seed=1)
        va = make_episodes(levels, range(nw[0], nw[0] + nw[1]), k, 0.5, rng_seed=2)
        te = make_episodes(levels, range(1000, 1000 + nw[2]), k, 0.5, rng_seed=3)
        kn = {n: np.array([knowable(e, t) for e in eps for t in range(k)]) for n, eps in (("tr", tr), ("va", va), ("te", te))}
        run.log(f"knowable share: train {kn['tr'].mean():.1%}, test {kn['te'].mean():.1%}")
        it = {n: episode_items(eps, "fb") for n, eps in (("tr", tr), ("va", va), ("te", te))}
        cert = {n: [dataclasses.replace(x, label=int(f)) for x, f in zip(it[n], kn[n])] for n in it}
        tt = np.array([x.t for x in it["te"]])
        y = np.array([x.label for x in it["te"]])
        res = {b: {} for b in list(BANDS) + ["all"]}
        pol = {l: {p: [] for p in ("answer all", "confidence gate", "certificate gate", "oracle gate")} for l in LAMS}
        polc = {l: {p: [] for p in ("answer all", "confidence gate", "certificate gate", "oracle gate")} for l in LAMS}
        cov = {p: [] for p in ("confidence gate", "certificate gate", "oracle gate")}
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            mf = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
            fit(mf, it["tr"], it["va"], device, epochs=epochs, seed=seed, log=lambda s: None)
            torch.manual_seed(seed)
            mc = LogicJudge(64, 4, n_types=N_TYPES["fb"]).to(device)
            fit(mc, cert["tr"], cert["va"], device, epochs=epochs, seed=seed, log=lambda s: None)
            P = {n: softmax(logits_of(mf, it[n], device).cpu().numpy()) for n in ("va", "te")}
            C = {n: softmax(logits_of(mc, cert[n], device).cpu().numpy())[:, 1] for n in ("va", "te")}
            yv = np.array([x.label for x in it["va"]])
            right = {"va": (P["va"].argmax(1) == yv).astype(float), "te": (P["te"].argmax(1) == y).astype(float)}
            conf = {n: P[n].max(1) for n in P}
            for b, (lo, hi) in list(BANDS.items()) + [("all", (0, 99))]:
                m = (tt >= lo) & (tt <= hi)
                r = res[b]
                r.setdefault("knowable share", []).append(float(kn["te"][m].mean()))
                r.setdefault("acc (all)", []).append(float(right["te"][m].mean()))
                r.setdefault("acc knowable", []).append(float(right["te"][m & kn["te"]].mean()) if (m & kn["te"]).any() else np.nan)
                r.setdefault("acc unknowable", []).append(float(right["te"][m & ~kn["te"]].mean()))
                r.setdefault("AUROC confidence -> knowable", []).append(auroc(conf["te"][m], kn["te"][m].astype(float)))
                r.setdefault("AUROC P(knowable) -> knowable", []).append(auroc(C["te"][m], kn["te"][m].astype(float)))
                r.setdefault("AUROC confidence -> correct", []).append(auroc(conf["te"][m], right["te"][m]))
                r.setdefault("AUROC P(knowable) -> correct", []).append(auroc(C["te"][m], right["te"][m]))
            # abstention policies (+1 right, -lam wrong, -0.2 abstain), thresholds fixed on validation episodes
            for lam in LAMS:
                tc, tk = best_threshold(conf["va"], right["va"], lam), best_threshold(C["va"], right["va"], lam)
                answer = {"answer all": np.ones(len(y), bool), "confidence gate": conf["te"] > tc, "certificate gate": C["te"] > tk, "oracle gate": kn["te"]}
                for p, g in answer.items():
                    pol[lam][p].append(float(np.where(g, np.where(right["te"] == 1, 1.0, -lam), -K).mean()))
                    if lam == 2.0 and p in cov:
                        cov[p].append(float(g.mean()))
            # luck-adjusted reward: a right answer without a proof available earns nothing
            def best_thr(score, ans_reward):
                cands = np.unique(np.quantile(score, np.linspace(0, 1, 101)))
                return float(max(cands, key=lambda t: np.where(score > t, ans_reward, -K).mean()))
            for lam in LAMS:
                rv = np.where(right["va"] == 1, np.where(kn["va"], 1.0, 0.0), -lam)
                rt = np.where(right["te"] == 1, np.where(kn["te"], 1.0, 0.0), -lam)
                tc, tk = best_thr(conf["va"], rv), best_thr(C["va"], rv)
                for p, g in {"answer all": np.ones(len(y), bool), "confidence gate": conf["te"] > tc, "certificate gate": C["te"] > tk, "oracle gate": kn["te"]}.items():
                    polc[lam][p].append(float(np.where(g, rt, -K).mean()))
            run.log(f"seed {seed}: acc {right['te'].mean():.1%}; AUROC confidence->knowable {res['all']['AUROC confidence -> knowable'][-1]:.3f}, "
                    f"P(knowable)->knowable {res['all']['AUROC P(knowable) -> knowable'][-1]:.3f}")
        mean = lambda xs: float(np.nanmean(xs))
        rows = ["knowable share", "acc (all)", "acc knowable", "acc unknowable", "AUROC confidence -> knowable", "AUROC P(knowable) -> knowable",
                "AUROC confidence -> correct", "AUROC P(knowable) -> correct"]
        L = ["# H3 in the hidden-premise regime: does the learner know which claims it can answer now?", "",
             f"{a.seeds} seeds; {len(te)} test episodes x {k} claims; 50% of the premises visible. Predictions K1-K3 are in the script docstring.", "",
             "| measure | " + " | ".join(res) + " |", "| --- | " + " | ".join("---:" for _ in res) + " |"]
        for r in rows:
            f = "{:.1%}" if r.startswith(("knowable", "acc")) else "{:.3f}"
            L.append(f"| {r} | " + " | ".join(f.format(mean(res[b][r])) + f" ({np.nanstd(res[b][r]):.3f})" for b in res) + " |")
        L += ["", "## Honest score of the abstention policies (+1 right, -lam wrong, -0.2 abstain; thresholds from validation)", "",
              "| policy | " + " | ".join(f"lam={l}" for l in LAMS) + " | coverage (lam=2) |", "| --- | " + " | ".join("---:" for _ in LAMS) + " | ---: |"]
        for p in pol[2.0]:
            L.append(f"| {p} | " + " | ".join(f"{mean(pol[l][p]):+.3f} ({np.std(pol[l][p]):.3f})" for l in LAMS) + f" | {mean(cov[p]):.1%} |" if p in cov else f"| {p} | " + " | ".join(f"{mean(pol[l][p]):+.3f} ({np.std(pol[l][p]):.3f})" for l in LAMS) + " | 100% |")
        L.append(f"| always abstain | " + " | ".join(f"{-K:+.3f}" for _ in LAMS) + " | 0% |")
        L += ["", "## Luck-adjusted reward (post hoc; right + knowable +1, right but unknowable 0, wrong -lam, abstain -0.2; thresholds from validation)", "",
              "| policy | " + " | ".join(f"lam={l}" for l in LAMS) + " |", "| --- | " + " | ".join("---:" for _ in LAMS) + " |"]
        for p in polc[2.0]:
            L.append(f"| {p} | " + " | ".join(f"{mean(polc[l][p]):+.3f} ({np.std(polc[l][p]):.3f})" for l in LAMS) + " |")
        L.append("| always abstain | " + " | ".join(f"{-K:+.3f}" for _ in LAMS) + " |")
        run.finish({"bands": res, "policies": {str(l): v for l, v in pol.items()}, "policies_luck_adjusted": {str(l): v for l, v in polc.items()}}, "\n".join(L) + "\n")


if __name__ == "__main__":
    main()

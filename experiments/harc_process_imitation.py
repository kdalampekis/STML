"""Process-level imitation, step 1: how predictable is a person's NEXT ACTION, and what explains it?

H1 says the *process* of learning (mistakes, corrections) carries information that final answers do not. The previous
experiments only saw final answers. Here the unit is the single edit action of H-ARC (paint a cell, change colour,
select, flood-fill, copy from input, undo, reset, resize, submit).

Question 1: next-operation prediction on UNSEEN puzzles. Feature sets add one thing at a time:
    frequency | last op | last 3 ops | + attempt number and feedback (was the last submit wrong?) | + timing | + current-grid features
Question 2 (descriptive): after a wrong submit, what do people do? locality of consecutive edits (adjacent cell? same colour?)

This is a regularity analysis with a gradient-boosted classifier, not yet the neural process model.

    python experiments/harc_process_imitation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stml.runlog import Run, table  # noqa: E402

DESC = {"write_first_description", "write_last_description", "no_last_description", "write_description"}
OPS = ["edit", "change_color", "select_cells", "select_change_color", "submit", "undo", "reset_grid", "change_height",
       "change_width", "change_tool", "copy_from_input", "floodfill", "select_paste"]


def grid_feats(g, ti: np.ndarray | None) -> tuple[float, float, float, float]:
    """Agreement with the test input, same size as the input, number of colours, share of non-background cells."""
    if not isinstance(g, str) or "|" not in g:
        return (np.nan,) * 4
    rows = g.strip("|").split("|")
    a = np.array([[int(c) for c in r] for r in rows], dtype=np.int8)
    same = ti is not None and a.shape == ti.shape
    agree = float((a == ti).mean()) if same else 0.0
    bg = np.bincount(a.ravel()).argmax()
    return agree, float(same), float(len(np.unique(a))), float((a != bg).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-rows", type=int, default=250000, help="training rows per model (sampled)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.max_rows = 20000
    with Run("harc_process_imitation" + ("_quick" if a.quick else ""), vars(a)) as run:
        cols = ["joint_id_task", "task_name", "attempt_number", "action_id", "action", "action_x", "action_y",
                "selected_symbol", "test_output_grid", "solved", "complete", "time"]
        d = pd.read_csv(ROOT / "data/raw/harc/data/data.csv", usecols=cols)
        d = d[d["complete"].astype(bool) & ~d["action"].isin(DESC)].copy()
        d = d.sort_values(["joint_id_task", "attempt_number", "action_id"], kind="stable")
        d["op"] = d["action"].where(d["action"].isin(OPS), "other")
        d["t"] = pd.to_datetime(d["time"], errors="coerce", format="mixed")
        arc = pd.concat(pd.read_csv(f) for f in sorted((ROOT / "data/raw/harc/arc_data").glob("ARC_*_tasks.csv")))
        arc = arc[arc.example_type == "test"].sort_values("example_number").drop_duplicates("task_name")
        ti = {n: np.array([[int(c) for c in r] for r in g.strip("|").split("|")], dtype=np.int8)
              for n, g in zip(arc.task_name, arc.input_grid)}
        g = d.groupby("joint_id_task", sort=False)
        d["step"] = g.cumcount()
        for k in (1, 2, 3):
            d[f"lag{k}"] = g["op"].shift(k).fillna("START")
        sub_wrong = (d["op"] == "submit") & (~d["solved"].astype(bool))
        d["n_wrong_before"] = sub_wrong.groupby(d["joint_id_task"]).cumsum() - sub_wrong.astype(int)
        d["since_attempt"] = d.groupby(["joint_id_task", "attempt_number"], sort=False).cumcount()
        d["dt"] = np.log1p(g["t"].diff().dt.total_seconds().clip(lower=0, upper=600).fillna(0))
        # grid after the previous action (what the person is looking at when deciding)
        feats = [grid_feats(x, ti.get(n)) for x, n in zip(d["test_output_grid"].shift(1), d["task_name"].shift(1))]
        d[["g_agree", "g_same", "g_colors", "g_fg"]] = np.array(feats)
        d.loc[d["step"] == 0, ["g_agree", "g_same", "g_colors", "g_fg"]] = np.nan
        d["log_step"] = np.log1p(d["step"])
        d["log_since"] = np.log1p(d["since_attempt"])
        run.log(f"{len(d)} actions, {d.joint_id_task.nunique()} participant-task sequences, {d.task_name.nunique()} puzzles")

        names = sorted(d.task_name.unique())
        rng = np.random.default_rng(a.seed)
        test_tasks = set(rng.permutation(names)[: len(names) // 2])
        te = d[d.task_name.isin(test_tasks)]
        tr = d[~d.task_name.isin(test_tasks)]
        tr = tr.sample(min(a.max_rows, len(tr)), random_state=a.seed)
        te = te.sample(min(a.max_rows, len(te)), random_state=a.seed)
        classes = sorted(d["op"].unique())
        yi_tr, yi_te = tr["op"].map({c: i for i, c in enumerate(classes)}), te["op"].map({c: i for i, c in enumerate(classes)})

        def design(df, sets):
            X = pd.DataFrame(index=df.index)
            for s in sets:
                if s.startswith("lag"):
                    X = X.join(pd.get_dummies(df[s].astype(pd.CategoricalDtype(OPS + ["other", "START"])), prefix=s).astype(float))
                else:
                    X[s] = df[s]
            return X
        families = {
            "frequency only": [],
            "+ last op": ["lag1"],
            "+ last 3 ops": ["lag1", "lag2", "lag3"],
            "+ attempt number and feedback": ["lag1", "lag2", "lag3", "attempt_number", "n_wrong_before", "log_since", "log_step"],
            "+ timing": ["lag1", "lag2", "lag3", "attempt_number", "n_wrong_before", "log_since", "log_step", "dt"],
            "+ current grid": ["lag1", "lag2", "lag3", "attempt_number", "n_wrong_before", "log_since", "log_step", "dt",
                               "g_agree", "g_same", "g_colors", "g_fg"],
        }
        rows = {}
        prior = np.bincount(yi_tr, minlength=len(classes)) / len(yi_tr)
        for name, sets in families.items():
            if not sets:
                P = np.tile(prior, (len(te), 1))
            else:
                clf = HistGradientBoostingClassifier(max_iter=80 if not a.quick else 20, learning_rate=0.1, random_state=a.seed)
                clf.fit(design(tr, sets), yi_tr)
                P = clf.predict_proba(design(te, sets))
            ll = log_loss(yi_te, np.clip(P, 1e-9, 1), labels=list(range(len(classes))))
            acc = float((P.argmax(1) == yi_te.values).mean())
            after = (te["n_wrong_before"] > 0).values
            rows[name] = {"loss": ll, "acc": acc,
                          "acc_after_wrong": float((P.argmax(1) == yi_te.values)[after].mean())}
            run.log(f"{name}: test log-loss {ll:.3f}, accuracy {acc:.1%}")
            if name == "+ current grid":
                last = (clf, sets, P)
        # descriptive: what happens after a wrong submit; locality of edits
        wrong = d[(d["lag1"] == "submit") & (d["n_wrong_before"] > 0)]
        right_first = d[(d["lag1"] == "START")]
        after_wrong = wrong["op"].value_counts(normalize=True).head(6)
        base = d["op"].value_counts(normalize=True)
        e = d[d["op"] == "edit"].copy()
        e["px"], e["py"] = e.groupby("joint_id_task")["action_x"].shift(1), e.groupby("joint_id_task")["action_y"].shift(1)
        e["psym"] = e.groupby("joint_id_task")["selected_symbol"].shift(1)
        e = e.dropna(subset=["px", "py", "action_x"])
        adj = ((e.action_x - e.px).abs() <= 1) & ((e.action_y - e.py).abs() <= 1)
        sameline = (e.action_x == e.px) | (e.action_y == e.py)
        samecol = e.selected_symbol == e.psym
        cols_ = [("loss", "Test log-loss", "{:.3f}"), ("acc", "Accuracy", "{:.1%}"), ("acc_after_wrong", "Accuracy once an attempt has failed", "{:.1%}")]
        L = ["# Process-level imitation, step 1: predicting the next action", "",
             f"{len(d):,} actions, {d.joint_id_task.nunique():,} participant-task sequences; puzzles split 50/50 (test puzzles never seen); "
             f"{len(classes)} operation classes; gradient-boosted classifier. Higher accuracy / lower loss = more predictable.", "",
             table(rows, cols_), "",
             "## After a wrong submit, the next operation (share) vs overall", "",
             "| operation | after a wrong submit | overall |", "| --- | ---: | ---: |"]
        L += [f"| {k} | {v:.1%} | {base.get(k, 0):.1%} |" for k, v in after_wrong.items()]
        L += ["", "## Locality of consecutive edits", "",
              f"- the next painted cell is adjacent (8-neighbourhood) to the previous one: {adj.mean():.1%}",
              f"- same row or column: {sameline.mean():.1%}",
              f"- same colour as the previous edit: {samecol.mean():.1%}", ""]
        run.finish({"models": rows, "after_wrong": after_wrong.to_dict(), "locality": {
            "adjacent": float(adj.mean()), "same_line": float(sameline.mean()), "same_colour": float(samecol.mean())}}, "\n".join(L))


if __name__ == "__main__":
    main()

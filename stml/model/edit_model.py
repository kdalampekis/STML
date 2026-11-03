"""Cell-level edit prediction for H-ARC: which cell will the person paint next?

Input channels (30 x 30, padded): the grid before the edit, the test input, last two edit positions, the active colour,
the valid-cell mask, and (the "process" channels) the grid the person last SUBMITTED, its difference to the current grid, and a
"has failed before" flag. The history channels are the experiment: the same network with them zeroed is the control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

H = 30
DESC = {"write_first_description", "write_last_description", "no_last_description", "write_description"}


def parse(g) -> np.ndarray | None:
    if not isinstance(g, str) or "|" not in g:
        return None
    rows = g.strip("|").split("|")
    a = np.frombuffer("".join(rows).encode(), dtype=np.uint8).reshape(len(rows), -1) - 48
    return a if a.shape[0] <= H and a.shape[1] <= H else None


def pad(a: np.ndarray | None) -> np.ndarray:
    out = np.full((H, H), -1, dtype=np.int8)
    if a is not None:
        out[: a.shape[0], : a.shape[1]] = a
    return out


def build_rows(data_csv, arc_dir, max_train: int, max_test: int, seed: int = 0):
    """Edit events with the state before each edit. Puzzles are split 50/50 (test puzzles never seen)."""
    cols = ["joint_id_task", "task_name", "attempt_number", "action_id", "action", "action_x", "action_y",
            "selected_symbol", "test_output_grid", "solved", "complete"]
    d = pd.read_csv(data_csv, usecols=cols)
    d = d[d["complete"].astype(bool) & ~d["action"].isin(DESC)].sort_values(["joint_id_task", "attempt_number", "action_id"], kind="stable")
    g = d.groupby("joint_id_task", sort=False)
    d["grid_prev"] = g["test_output_grid"].shift(1)
    sub = d["test_output_grid"].where(d["action"] == "submit")
    d["sub_prev"] = sub.groupby(d["joint_id_task"]).ffill().groupby(d["joint_id_task"]).shift(1)
    wrong = (d["action"] == "submit") & (~d["solved"].astype(bool))
    d["n_wrong"] = wrong.groupby(d["joint_id_task"]).cumsum() - wrong.astype(int)
    e = d[(d["action"] == "edit") & d["grid_prev"].notna()].copy()
    ge = e.groupby("joint_id_task", sort=False)
    for k in (1, 2):
        e[f"px{k}"], e[f"py{k}"] = ge["action_x"].shift(k), ge["action_y"].shift(k)
    names = sorted(e.task_name.unique())
    rng = np.random.default_rng(seed)
    test_tasks = set(rng.permutation(names)[: len(names) // 2])
    arc = pd.concat(pd.read_csv(f) for f in sorted(__import__("pathlib").Path(arc_dir).glob("ARC_*_tasks.csv")))
    arc = arc[arc.example_type == "test"].sort_values("example_number").drop_duplicates("task_name")
    ti = {n: parse(x) for n, x in zip(arc.task_name, arc.input_grid)}
    out = {}
    for split, mask, cap in (("train", ~e.task_name.isin(test_tasks), max_train), ("test", e.task_name.isin(test_tasks), max_test)):
        s = e[mask]
        s = s.sample(min(cap, len(s)), random_state=seed)
        grid = np.stack([pad(parse(x)) for x in s.grid_prev])
        ax, ay = s.action_x.values, s.action_y.values
        inside = (ax >= 0) & (ay >= 0) & (ax < H) & (ay < H)
        keep = inside.copy()
        keep[inside] = grid[np.nonzero(inside)[0], ay[inside].astype(int), ax[inside].astype(int)] >= 0     # the edited cell must exist in the grid before the edit
        s, grid = s[keep], grid[keep]
        out[split] = {
            "grid": grid,
            "inp": np.stack([pad(ti.get(n)) for n in s.task_name]),
            "sub": np.stack([pad(parse(x)) for x in s.sub_prev]),
            "has_sub": s.sub_prev.notna().values,
            "pos1": s[["px1", "py1"]].fillna(-1).values.astype(np.int16), "pos2": s[["px2", "py2"]].fillna(-1).values.astype(np.int16),
            "colour": s.selected_symbol.fillna(0).values.astype(np.int64),
            "y": (s.action_y.values.astype(np.int64) * H + s.action_x.values.astype(np.int64)),   # row-major cell index
            "n_wrong": s.n_wrong.values.astype(np.int64),
            "task": pd.factorize(s.task_name)[0].astype(np.int64),
        }
    return out


def make_input(b: dict, device, history: bool, shuffle_history: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.as_tensor(b["grid"], device=device).long()
    inp = torch.as_tensor(b["inp"], device=device).long()
    valid = (g >= 0)
    B = g.shape[0]
    oh = lambda x: F.one_hot(x.clamp(min=0), 10).permute(0, 3, 1, 2).float() * (x >= 0).unsqueeze(1).float()
    ch = [oh(g), oh(inp)]
    same_size = ((g >= 0).sum((1, 2)) == (inp >= 0).sum((1, 2))).float().view(B, 1, 1, 1)
    ch.append(same_size.expand(B, 1, H, H))
    for key in ("pos1", "pos2"):
        m = torch.zeros(B, 1, H, H, device=device)
        p = torch.as_tensor(b[key], device=device).long()
        ok = (p[:, 0] >= 0) & (p[:, 1] >= 0)
        idx = torch.nonzero(ok).squeeze(1)
        m[idx, 0, p[idx, 1], p[idx, 0]] = 1.0
        ch.append(m)
    ch.append(F.one_hot(torch.as_tensor(b["colour"], device=device).clamp(0, 9), 10).float().view(B, 10, 1, 1).expand(B, 10, H, H))
    ch.append(valid.unsqueeze(1).float())
    sub = torch.as_tensor(b["sub"], device=device).long()
    flag = torch.as_tensor(b["n_wrong"] > 0, device=device).float().view(B, 1, 1, 1).expand(B, 1, H, H)
    has = torch.as_tensor(np.asarray(b["has_sub"]).copy(), device=device)
    if shuffle_history:                                        # control: history from a different sample, link to this edit broken
        perm = torch.randperm(B, device=device)
        sub, has, flag = sub[perm], has[perm], flag[perm]
    diff = ((sub != g) & (sub >= 0) & valid & has.view(B, 1, 1)).unsqueeze(1).float()
    if history:
        ch += [oh(sub) * has.view(B, 1, 1, 1).float(), diff, flag]
    else:
        ch += [torch.zeros(B, 10, H, H, device=device), torch.zeros(B, 1, H, H, device=device), torch.zeros(B, 1, H, H, device=device)]
    return torch.cat(ch, 1), valid.view(B, -1)


class EditNet(nn.Module):
    def __init__(self, c_in: int = 46, width: int = 64, dilations=(1, 2, 4, 1, 1)):
        super().__init__()
        self.inp = nn.Conv2d(c_in, width, 3, padding=1)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Conv2d(width, width, 3, padding=d, dilation=d), nn.GroupNorm(8, width), nn.ReLU(),
                                                  nn.Conv2d(width, width, 3, padding=1)) for d in dilations)
        self.out = nn.Conv2d(width, 1, 1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.inp(x))
        for blk in self.blocks:
            h = F.relu(h + blk(h))
        return self.out(h).flatten(1).masked_fill(~valid, -1e9)           # [B, 900] logits over valid cells

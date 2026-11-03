"""Self-supervised pretraining of the object-graph encoder ('learn to perceive'): masked node-attribute prediction.

A random fraction of the nodes of every grid graph has its input features zeroed (colour one-hot, size, bounding box, centroid, background flag, grid size); the encoder must reconstruct them from the
other objects and the typed edges (touch, same colour, same shape, aligned row / column). No labels, no human data, no candidate answers' correctness: only the grids themselves.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphs import Graph
from .network import GraphBatch, RelationalGNN


def pretrain_gnn(gnn: RelationalGNN, graphs: list[Graph], device: torch.device, epochs: int = 5, batch: int = 64,
                 mask_frac: float = 0.2, lr: float = 1e-3, seed: int = 0, log=lambda s: None) -> dict:
    """Trains `gnn` in place with a throw-away linear decoder; returns its state_dict."""
    torch.manual_seed(seed)
    rnd = random.Random(seed)
    node_dim = gnn.inp.in_features
    dec = nn.Linear(gnn.d, node_dim).to(device)
    gnn.to(device).train()
    opt = torch.optim.AdamW(list(gnn.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-4)
    order = list(range(len(graphs)))
    for ep in range(epochs):
        rnd.shuffle(order)
        tot = n = 0
        for i in range(0, len(order), batch):
            b = GraphBatch.build([graphs[j] for j in order[i:i + batch]], device)
            target = b.x.clone()
            mask = torch.rand(b.x.shape[0], device=device) < mask_frac
            if not mask.any():
                continue
            b.x = b.x.clone()
            b.x[mask] = 0.0
            pred = dec(gnn.nodes(b))[mask]
            t = target[mask]
            loss = F.cross_entropy(pred[:, :10], t[:, :10].argmax(1)) + F.mse_loss(pred[:, 10:], t[:, 10:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + float(loss), n + 1
        log(f"pretrain epoch {ep + 1}/{epochs}: loss {tot / max(n, 1):.3f}")
    return {k: v.detach().cpu().clone() for k, v in gnn.state_dict().items()}


# ----------------------------------------------------------------------------- puzzle-level pretext: predict properties of the held-out output

def output_descriptor(grid_str: str, input_str: str) -> list[float]:
    """8 properties of an output grid that a solver needs: shape, palette size, density, background share, symmetries, same shape as the input."""
    import numpy as np
    from .graphs import parse_grid
    g, i = parse_grid(grid_str), parse_grid(input_str)
    return [g.shape[0] / 30.0, g.shape[1] / 30.0, len(np.unique(g)) / 10.0, float((g != 0).mean()),
            float(np.bincount(g.ravel()).max() / g.size), float((g == g[:, ::-1]).all()), float((g == g[::-1, :]).all()),
            float(i is not None and g.shape == i.shape)]


def pretrain_task(model, puzzles: list, device: torch.device, epochs: int = 6, batch: int = 8, lr: float = 5e-4, seed: int = 0, log=lambda s: None) -> dict:
    """Leave-one-pair-out pretext on the demonstration pairs of the given (training) puzzles: the task vector built from the OTHER pairs and the held-out INPUT must predict
    the descriptor of the held-out OUTPUT. Trains model.gnn, model.pair, model.task_in and a throw-away head; returns their state dicts. Uses only the demonstration pairs, never the test output or candidates' labels."""
    torch.manual_seed(seed)
    rnd = random.Random(seed)
    head = nn.Linear(model.d, 8).to(device)
    mods = nn.ModuleDict({"gnn": model.gnn, "pair": model.pair, "task_in": model.task_in}).to(device).train()
    opt = torch.optim.AdamW(list(mods.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    usable = [p for p in puzzles if len(p.graph_train) >= 2]
    for ep in range(epochs):
        rnd.shuffle(usable)
        tot = n = 0
        for i in range(0, len(usable), batch):
            chunk = usable[i:i + batch]
            graphs, idx = [], []
            for p in chunk:
                ids = []
                for gi, go in p.graph_train:
                    ids.append((len(graphs), len(graphs) + 1))
                    graphs += [gi, go]
                idx.append(ids)
            emb = model.gnn(GraphBatch.build(graphs, device))
            preds, targets = [], []
            for p, ids in zip(chunk, idx):
                pin = emb[[a for a, _ in ids]]
                pout = emb[[b for _, b in ids]]
                pr = model.pair(torch.cat([pin, pout, pout - pin], 1))
                for k in range(len(ids)):
                    others = [j for j in range(len(ids)) if j != k]
                    task = model.task_in(torch.cat([pr[others].mean(0), pin[k]], 0))
                    preds.append(head(task))
                    targets.append(output_descriptor(p.train[k][1], p.train[k][0]))
            loss = F.mse_loss(torch.stack(preds), torch.tensor(targets, dtype=torch.float32, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + float(loss), n + 1
        log(f"task pretext epoch {ep + 1}/{epochs}: loss {tot / max(n, 1):.4f}")
    return {k: {kk: vv.detach().cpu().clone() for kk, vv in m.state_dict().items()} for k, m in mods.items()}

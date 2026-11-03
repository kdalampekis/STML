"""Turn states into typed graphs: the one input format of the STML learner.

Two domains, one representation:

  ARC grid      -> object graph. Each node is an object: a 4-connected region of
                   one colour. Edges: 'touch' (objects share a border),
                   'same_color', 'same_shape' (same size and bounding box),
                   'aligned_row', 'aligned_col' (overlapping rows / columns).
  Logic world   -> concept graph. Each node is a kind (wug, dax, ...). Edges are
                   the four relations All / No / Some / Some-not, each marked by
                   the learner's epistemic state: given, proven, refuted, open.

A Graph is plain numpy so it can be cached to disk and batched cheaply; the
encoder turns batches into torch tensors on the chosen device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------- common

@dataclass
class Graph:
    x: np.ndarray            # node features, float32 [n_nodes, node_dim]
    edges: np.ndarray        # int64 [n_edges, 3]: (source, target, edge type)
    n_types: int

    @property
    def n(self) -> int:
        return int(self.x.shape[0])


# ----------------------------------------------------------------------------- ARC grids

ARC_EDGE_TYPES = ("touch", "same_color", "same_shape", "aligned_row", "aligned_col")
ARC_NODE_DIM = 10 + 9   # colour one-hot + size, bbox, centroid, is-background, grid size


def parse_grid(s) -> np.ndarray | None:
    """'|077|777|' -> int8 array; None if it is not a grid."""
    if isinstance(s, np.ndarray):
        return s
    if not isinstance(s, str) or "|" not in s:
        return None
    rows = s.strip().strip("|").split("|")
    if not rows or rows == [""] or len({len(r) for r in rows}) != 1:
        return None
    return np.array([[int(c) for c in r] for r in rows], dtype=np.int8)


def _objects(g: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """4-connected components of equal colour: list of (colour, boolean mask)."""
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    out = []
    for y in range(h):
        for x in range(w):
            if seen[y, x]:
                continue
            c = g[y, x]
            stack, mask = [(y, x)], np.zeros_like(g, dtype=bool)
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                mask[cy, cx] = True
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] == c:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append((int(c), mask))
    return out


def grid_graph(grid, max_objects: int = 64) -> Graph:
    """Object graph of an ARC grid. Very fragmented grids keep their largest objects."""
    g = parse_grid(grid)
    if g is None:
        raise ValueError("not a grid")
    h, w = g.shape
    objs = _objects(g)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    objs.sort(key=lambda o: -int(o[1].sum()))
    objs = objs[:max_objects]
    feats, boxes, sizes = [], [], []
    for c, m in objs:
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        size = int(m.sum())
        onehot = np.zeros(10, dtype=np.float32)
        onehot[c] = 1.0
        feats.append(np.concatenate([onehot, np.array([
            np.log1p(size) / np.log1p(900), y0 / 30, x0 / 30, (y1 - y0 + 1) / 30, (x1 - x0 + 1) / 30,
            ys.mean() / max(h, 1), xs.mean() / max(w, 1), float(c == bg), (h * w) / 900], dtype=np.float32)]))
        boxes.append((y0, y1, x0, x1))
        sizes.append(size)
    # touch: dilate each mask by one cell (4-neighbourhood) and test overlap
    masks = [m for _, m in objs]
    dil = []
    for m in masks:
        d = m.copy()
        d[1:, :] |= m[:-1, :]
        d[:-1, :] |= m[1:, :]
        d[:, 1:] |= m[:, :-1]
        d[:, :-1] |= m[:, 1:]
        dil.append(d)
    edges = []
    n = len(objs)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (dil[i] & masks[j]).any():
                edges.append((i, j, 0))
            if objs[i][0] == objs[j][0]:
                edges.append((i, j, 1))
            bi, bj = boxes[i], boxes[j]
            if sizes[i] == sizes[j] and (bi[1] - bi[0], bi[3] - bi[2]) == (bj[1] - bj[0], bj[3] - bj[2]):
                edges.append((i, j, 2))
            if bi[0] <= bj[1] and bj[0] <= bi[1]:
                edges.append((i, j, 3))
            if bi[2] <= bj[3] and bj[2] <= bi[3]:
                edges.append((i, j, 4))
    e = np.array(edges, dtype=np.int64).reshape(-1, 3)
    return Graph(np.stack(feats).astype(np.float32), e, len(ARC_EDGE_TYPES))


# ----------------------------------------------------------------------------- logic worlds

LOGIC_FORMS = ("all", "no", "some", "some_not")
LOGIC_MARKS = ("given", "proven", "refuted", "open")
LOGIC_EDGE_TYPES = tuple(f"{f}:{m}" for f in LOGIC_FORMS for m in LOGIC_MARKS)
LOGIC_NODE_DIM = 4


def concept_graph(n_kinds: int, edges: list[tuple[str, int, int, str]]) -> Graph:
    """Concept graph of a logic world.

    edges: (form, a, b, mark) with form in LOGIC_FORMS and mark in LOGIC_MARKS,
    meaning e.g. ('all', 0, 2, 'proven'): the learner has proven All K0 are K2.
    Node features carry no names, so renaming kinds changes nothing.
    """
    deg_in = np.zeros(n_kinds)
    deg_out = np.zeros(n_kinds)
    out = []
    for form, a, b, mark in edges:
        t = LOGIC_EDGE_TYPES.index(f"{form}:{mark}")
        out.append((a, b, t))
        deg_out[a] += 1
        deg_in[b] += 1
    x = np.stack([np.ones(n_kinds), deg_in / max(1, n_kinds), deg_out / max(1, n_kinds),
                  np.full(n_kinds, n_kinds / 12)], axis=1).astype(np.float32)
    return Graph(x, np.array(out, dtype=np.int64).reshape(-1, 3), len(LOGIC_EDGE_TYPES))

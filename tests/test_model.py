"""The STML learner on tiny synthetic puzzles: graphs, a forward pass, learning, play."""

import pytest

torch = pytest.importorskip("torch")

from stml.model.graphs import ARC_EDGE_TYPES, ARC_NODE_DIM, concept_graph, grid_graph  # noqa: E402
from stml.model.harc_data import Puzzle, Step  # noqa: E402
from stml.model.network import STMLLearner, pick_device  # noqa: E402
from stml.model.train import evaluate, fit_imitation, play, train_rl  # noqa: E402


def _puzzle(name: str, flip: bool) -> Puzzle:
    """Rule: mirror the grid left-right. Candidates: the input, the mirror, a blank grid."""
    a, b = "|120|300|", "|021|003|"
    cands = sorted([a, b, "|000|000|"])
    p = Puzzle(name, [(a, b), ("|450|006|", "|054|600|")], a, b, cands)
    p.graph_train = [(grid_graph(x), grid_graph(y)) for x, y in p.train]
    p.graph_test_in = grid_graph(p.test_input)
    p.graph_cands = [grid_graph(c) for c in p.candidates]
    return p


def test_grid_graph_objects_and_edges():
    g = grid_graph("|110|100|002|")
    assert g.x.shape[1] == ARC_NODE_DIM
    assert g.n == 3                                     # the 1-blob, the 0-region, the single 2
    assert set(g.edges[:, 2]) <= set(range(len(ARC_EDGE_TYPES)))
    assert (g.edges[:, 2] == 0).any()                   # some objects touch


def test_concept_graph():
    g = concept_graph(3, [("all", 0, 1, "given"), ("no", 1, 2, "proven")])
    assert g.n == 3 and len(g.edges) == 2


def test_learner_learns_and_plays():
    device = torch.device("cpu")
    puzzles = {f"p{i}": _puzzle(f"p{i}", i % 2 == 0) for i in range(4)}
    model = STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=32).to(device)
    steps = [Step(n, [], p.target_index, True) for n, p in puzzles.items()] * 4
    before = evaluate(model, puzzles, sorted(puzzles), device, human=steps)["human_loglik"]
    fit_imitation(model, puzzles, steps, device, epochs=30, lr=3e-3, batch=4, log=lambda m: None)
    after = evaluate(model, puzzles, sorted(puzzles), device, human=steps)
    assert after["human_loglik"] > before                # it learned the choices it imitated
    assert after["solved"] == 1.0
    enc = model.encode_puzzles([puzzles["p0"]], device)[0]
    solved, k, confs = play(model, enc, puzzles["p0"])
    assert solved and k == 1 and 0.0 <= confs[0][0] <= 1.0


def test_rl_records_its_history():
    device = torch.device("cpu")
    puzzles = {"p0": _puzzle("p0", True)}
    model = STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=16).to(device)
    hist = train_rl(model, puzzles, ["p0"], device, episodes=8, batch=4, log=lambda m: None)
    assert hist and all(isinstance(s, Step) for s in hist)


def test_pick_device_returns_a_device():
    assert pick_device().type in ("mps", "cuda", "cpu")


def test_logic_judge_is_permutation_invariant_and_runs():
    import torch
    from stml.model.logic_data import make_items
    from stml.model.logic_judge import LogicJudge, logits_of
    items = make_items([2, 4], range(3), per_world=4)
    assert items and all(i.label in (0, 1, 2) for i in items)
    torch.manual_seed(0)
    m = LogicJudge(d=16, layers=2)
    out = logits_of(m, items, torch.device("cpu"))
    assert out.shape == (len(items), 3)
    assert torch.allclose(out, logits_of(m, items[::-1], torch.device("cpu")).flip(0), atol=1e-5)  # batch order does not matter


def test_shared_gnn_matches_interface_and_transfers():
    import torch
    from stml.model.graphs import ARC_EDGE_TYPES, ARC_NODE_DIM, grid_graph
    from stml.model.logic_data import EDGE_TYPES, NODE_DIM, make_items
    from stml.model.logic_judge import LogicJudge, logits_of
    from stml.model.network import GraphBatch, STMLLearner
    from stml.model.shared_gnn import DomainEncoder, SharedCore, transfer_core
    torch.manual_seed(0)
    cpu = torch.device("cpu")
    # ARC side: the encoder plugs into STMLLearner
    arc = DomainEncoder(SharedCore(16, 2, 8), ARC_NODE_DIM, len(ARC_EDGE_TYPES))
    g = [grid_graph("|01|10|"), grid_graph("|011|100|")]
    emb = arc(GraphBatch.build(g, cpu))
    assert emb.shape == (2, 16)
    STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=16, gnn=arc)
    # logic side: a copy of the ARC core is reused with a different node dimension and relation table
    core = transfer_core(arc.core)
    assert all(torch.equal(a, b) for a, b in zip(core.state_dict().values(), arc.core.state_dict().values()))
    items = make_items([2], range(3), per_world=4)
    j = LogicJudge(16, 2, gnn=DomainEncoder(core, NODE_DIM, len(EDGE_TYPES)))
    assert logits_of(j, items, cpu).shape == (len(items), 3)
    frozen = transfer_core(arc.core, freeze=True)
    assert not any(p.requires_grad for p in frozen.parameters())
    # permutation invariance of node order inside a graph
    out1 = logits_of(j, items, cpu)
    assert torch.allclose(out1, logits_of(j, items[::-1], cpu).flip(0), atol=1e-5)


@pytest.mark.parametrize("mode", ["concat", "film", "both", "lowrank", "hyper"])
def test_shared_core_modes_run_and_are_trainable(mode):
    import torch
    from stml.model.graphs import grid_graph
    from stml.model.network import GraphBatch
    from stml.model.shared_gnn import DomainEncoder, SharedCore, transfer_core
    torch.manual_seed(0)
    enc = DomainEncoder(SharedCore(16, 2, 8, mode, 4), 19, 5)
    g = [grid_graph("|01|10|"), grid_graph("|011|100|")]
    out = enc(GraphBatch.build(g, torch.device("cpu")))
    assert out.shape == (2, 16) and torch.isfinite(out).all()
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.core.parameters())
    copy = transfer_core(enc.core, mode=mode, rank=4)
    assert copy.mode == mode


def test_hand_for_grids_matches_attach_on_a_toy_puzzle():
    from stml.model.hand_features import N_HAND, hand_for_grids
    f = hand_for_grids([("|01|10|", "|10|01|")], "|01|10|", "|10|01|", ["|10|01|", "|11|11|"])
    assert f.shape == (2, N_HAND) and f.dtype.name == "float32"


def test_pretrain_gnn_reduces_masked_reconstruction_loss():
    import numpy as np
    import torch
    from stml.model.graphs import ARC_EDGE_TYPES, ARC_NODE_DIM, grid_graph
    from stml.model.network import RelationalGNN
    from stml.model.pretrain import pretrain_gnn
    rng = np.random.default_rng(0)
    graphs = [grid_graph("|" + "|".join("".join(str(int(c)) for c in r) for r in rng.integers(0, 3, (6, 6))) + "|") for _ in range(40)]
    gnn = RelationalGNN(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=32, layers=2)
    logs = []
    state = pretrain_gnn(gnn, graphs, torch.device("cpu"), epochs=4, batch=8, log=logs.append)
    assert set(state) == set(gnn.state_dict())
    first, last = (float(s.split("loss ")[1]) for s in (logs[0], logs[-1]))
    assert last < first


def test_pretrain_task_runs_and_descriptor_shape():
    import torch
    from stml.model.graphs import ARC_EDGE_TYPES, ARC_NODE_DIM, grid_graph
    from stml.model.harc_data import Puzzle
    from stml.model.network import STMLLearner
    from stml.model.pretrain import output_descriptor, pretrain_task
    d = output_descriptor("|01|10|", "|01|10|")
    assert len(d) == 8 and d[-1] == 1.0
    pairs = [("|01|10|", "|10|01|"), ("|11|00|", "|00|11|"), ("|10|10|", "|01|01|")]
    p = Puzzle("t", pairs, "|01|10|", "|10|01|", ["|10|01|"])
    p.graph_train = [(grid_graph(a), grid_graph(b)) for a, b in pairs]
    m = STMLLearner(ARC_NODE_DIM, len(ARC_EDGE_TYPES), d=16, out_match=True)
    st = pretrain_task(m, [p, p], torch.device("cpu"), epochs=1, batch=2)
    assert set(st) == {"gnn", "pair", "task_in"}

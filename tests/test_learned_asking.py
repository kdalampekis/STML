import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import logic_learned_asking as la  # noqa: E402
import stml.model.logic_data as ld  # noqa: E402
from stml.model.logic_episodes import N_TYPES, make_episodes  # noqa: E402
from stml.model.logic_judge import LogicJudge  # noqa: E402


def test_features_and_rollout():
    ld.SIZE_FREE = True
    eps = [e for e in make_episodes([2, 3], range(6), 20, 0.5) if len(e.queries) >= 20][:4]
    m = LogicJudge(16, 2, n_types=N_TYPES["fb"])
    f = la.features(m, eps, [[] for _ in eps], torch.device("cpu"), True)
    assert f.shape == (len(eps), la.POOL, la.NF + 2 + 5)
    net = nn.Sequential(nn.Linear(f.shape[2], 8), nn.ReLU(), nn.Linear(8, 1))
    asked, logps, _ = la.rollout(m, eps, torch.device("cpu"), "learned", True, True, net=net)
    assert all(len(set(a)) == la.BUDGET for a in asked) and len(logps) == la.BUDGET
    assert la.heldout_acc(m, eps, asked, torch.device("cpu")).shape == (len(eps),)

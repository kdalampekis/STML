import torch

import stml.model.logic_data as ld
from stml.model.logic_aux import attach_aux, aux_targets, fit_aux
from stml.model.logic_episodes import N_TYPES, episode_items, make_episodes
from stml.model.logic_judge import LogicJudge


def test_aux_targets_mask_visible_and_fit_runs():
    ld.SIZE_FREE = True
    eps = make_episodes([2, 3], range(6), k=4, vis_frac=0.5)
    t, m = aux_targets(eps[0])
    assert t.shape == m.shape and t.shape[2] == 4
    hidden = len(eps[0].full) - len(eps[0].visible)
    assert hidden > 0 and (t * m).sum() >= hidden          # every hidden premise is a masked-in positive (symmetric ones count twice)
    for it_eps in (None,):
        pass
    items = episode_items(eps, "fb")
    attach_aux(items, eps)
    sh = episode_items(eps, "fb")
    attach_aux(sh, eps, shuffle_seed=0)
    m1 = LogicJudge(16, 2, n_types=N_TYPES["fb"])
    fit_aux(m1, items, items[:8], torch.device("cpu"), epochs=1, aux_weight=0.5)
    assert hasattr(m1, "aux_head")

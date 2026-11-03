import random

import stml.model.logic_data as ld
from stml.model.logic_lemma import N_TYPES_LEMMA, build_world, item_for, train_items, with_lemmas


def test_lemma_edges_and_items():
    ld.SIZE_FREE = True
    w = build_world(5, 1, 0.9)
    assert len(w.pool) == len(w.labels) and 0 in w.labels
    ent = [i for i, l in enumerate(w.labels) if l == 0]
    base = item_for(w, 0, [])
    withl = item_for(w, 0, [w.pool[ent[0]]] if ent[0] != 0 else [w.pool[ent[-1]]])
    assert withl.graph.n_types == N_TYPES_LEMMA == base.graph.n_types
    assert len(withl.graph.edges) == len(base.graph.edges) + 2
    assert withl.graph.edges[:, 2].max() >= 8
    its = train_items([w], random.Random(0), True)
    assert len(its) >= 3 and all(it.graph.n_types == N_TYPES_LEMMA for it in its)

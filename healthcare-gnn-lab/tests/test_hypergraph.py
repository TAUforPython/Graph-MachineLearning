import numpy as np
from healthcare_gnn.graphs.hypergraph import incidence_from_memberships, normalized_hypergraph_operator

def test_transparent_hypergraph_operator():
    H=incidence_from_memberships(4,[[0,1,2],[2,3]])
    operator=normalized_hypergraph_operator(H)
    assert operator.shape==(4,4); assert np.allclose(operator,operator.T); assert np.isfinite(operator).all()

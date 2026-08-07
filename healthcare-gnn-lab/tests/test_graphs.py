import numpy as np
import pytest
from healthcare_gnn.graphs.build import build_graph
from healthcare_gnn.graphs.diagnostics import graph_diagnostics

def test_graph_is_deterministic_symmetric_and_inductive():
    X=np.arange(60,dtype=float).reshape(20,3); train=np.arange(12)
    a=build_graph(X,k=2,metric="gower",fit_indices=train)
    b=build_graph(X,k=2,metric="gower",fit_indices=train)
    assert np.array_equal(a.edge_index,b.edge_index)
    edges=set(map(tuple,a.edge_index.T)); assert all((v,u) in edges for u,v in edges)
    assert all(u!=v for u,v in edges)
    assert not any(u>=12 and v>=12 for u,v in edges)
    assert graph_diagnostics(a.edge_index,20)["isolated_nodes"]==0

def test_strict_mode_refuses_full_fit():
    with pytest.raises(ValueError,match="full cohort"): build_graph(np.eye(5),k=1,fit_indices=np.arange(5))

def test_preprocessing_is_fit_only_on_training_rows():
    X=np.array([[0.],[1.],[2.],[10000.]])
    result=build_graph(X,k=1,fit_indices=[0,1,2])
    assert np.isclose(result.transformed_X[:3].mean(),0.0)

from healthcare_gnn.graphs.ecg import lead_adjacency

def test_ecg_adjacencies_are_symmetric_and_label_free():
    signal=np.random.default_rng(1).normal(size=(12,100))
    for mode in ("physiological","functional","combined"):
        A=lead_adjacency(signal,mode,0.4)
        assert A.shape==(12,12) and np.allclose(A,A.T) and np.all(np.diag(A)==0)

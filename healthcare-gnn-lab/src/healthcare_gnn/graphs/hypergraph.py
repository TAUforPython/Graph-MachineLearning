"""Transparent incidence-matrix hypergraph propagation."""
from __future__ import annotations
import numpy as np

def incidence_from_memberships(n_nodes: int, hyperedges: list[list[int]]) -> np.ndarray:
    H=np.zeros((n_nodes,len(hyperedges)),dtype=np.float32)
    for column,members in enumerate(hyperedges):
        if not members: raise ValueError("Hyperedges cannot be empty")
        H[np.asarray(members,dtype=int),column]=1.0
    return H

def normalized_hypergraph_operator(H: np.ndarray) -> np.ndarray:
    H=np.asarray(H,dtype=float)
    dv=H.sum(axis=1); de=H.sum(axis=0)
    if np.any(dv==0) or np.any(de==0): raise ValueError("Incidence matrix contains an isolated node or empty hyperedge")
    return np.diag(dv**-0.5) @ H @ np.diag(de**-1) @ H.T @ np.diag(dv**-0.5)

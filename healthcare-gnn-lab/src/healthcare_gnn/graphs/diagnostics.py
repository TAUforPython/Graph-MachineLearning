"""Graph-value diagnostics and perturbations."""
from __future__ import annotations
import numpy as np


def graph_diagnostics(edge_index: np.ndarray, n_nodes: int, labels: np.ndarray | None = None) -> dict[str, float | int]:
    adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
    adjacency[edge_index[0], edge_index[1]] = True
    degree = adjacency.sum(axis=1)
    seen=set(); components=0
    for start in range(n_nodes):
        if start in seen: continue
        components += 1; stack=[start]
        while stack:
            node=stack.pop()
            if node in seen: continue
            seen.add(node); stack.extend(np.flatnonzero(adjacency[node]).tolist())
    result: dict[str,float|int] = {
        "density": float(adjacency.sum() / max(1, n_nodes*(n_nodes-1))),
        "isolated_nodes": int((degree == 0).sum()), "connected_components": components,
        "mean_degree": float(degree.mean()), "max_degree": int(degree.max(initial=0)),
    }
    if labels is not None and edge_index.shape[1]:
        y=np.asarray(labels); result["edge_homophily"] = float(np.mean(y[edge_index[0]] == y[edge_index[1]]))
        result["label_mixing"] = 1.0 - result["edge_homophily"]
    return result


def drop_edges(edge_index: np.ndarray, probability: float, seed: int) -> np.ndarray:
    if not 0 <= probability < 1: raise ValueError("probability must be in [0,1)")
    keep=np.random.default_rng(seed).random(edge_index.shape[1]) >= probability
    return edge_index[:,keep]


def shuffle_features(X: np.ndarray, seed: int) -> np.ndarray:
    return np.asarray(X)[np.random.default_rng(seed).permutation(len(X))]

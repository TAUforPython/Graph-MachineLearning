"""Leakage-aware graph construction."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

Method = Literal["knn", "radius", "physiological", "functional", "learned"]
Metric = Literal["cosine", "euclidean", "gower", "poincare"]

@dataclass(frozen=True)
class GraphBuildResult:
    edge_index: np.ndarray
    edge_weight: np.ndarray
    transformed_X: np.ndarray
    fit_indices: np.ndarray
    metadata: dict[str, object]


def _gower_like(X: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Range-normalized L1 distance for numeric mixed-scale features."""
    scaled = X / np.where(ranges > 0, ranges, 1.0)
    return pairwise_distances(scaled, metric="manhattan") / X.shape[1]


def _symmetrize(adjacency: np.ndarray, mode: Literal["union", "mutual"]) -> np.ndarray:
    return np.maximum(adjacency, adjacency.T) if mode == "union" else np.minimum(adjacency, adjacency.T)


def build_graph(
    X: np.ndarray,
    method: Method = "knn",
    metric: Metric = "cosine",
    k: int = 3,
    threshold: float | None = None,
    fit_indices: Sequence[int] | None = None,
    *,
    symmetrize: Literal["union", "mutual"] = "union",
    self_loops: bool = False,
    strict_inductive: bool = True,
) -> GraphBuildResult:
    """Fit preprocessing on ``fit_indices`` and construct a feature-only graph.

    Labels are intentionally not accepted. In strict mode, callers must provide a
    proper subset as fit indices, preventing an accidental full-cohort fit.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 2:
        raise ValueError("X must be a 2-D array with at least two rows")
    fit = np.arange(len(X)) if fit_indices is None else np.asarray(fit_indices, dtype=int)
    if fit.size == 0 or np.any((fit < 0) | (fit >= len(X))):
        raise ValueError("fit_indices must contain valid training rows")
    if strict_inductive and len(np.unique(fit)) == len(X):
        raise ValueError("Strict inductive mode refuses preprocessing fit on the full cohort")
    if not 1 <= k < len(X):
        raise ValueError("k must be between 1 and n_samples-1")
    if method in {"learned", "physiological"}:
        raise ValueError(f"{method!r} requires an explicit task-specific adjacency, not a patient-feature matrix")
    if metric == "poincare":
        raise ValueError("Poincaré KNN requires already validated ball embeddings; use the geometry module explicitly")

    imputer = SimpleImputer(strategy="median").fit(X[fit])
    imputed = imputer.transform(X)
    scaler = StandardScaler().fit(imputed[fit])
    transformed = scaler.transform(imputed)
    if metric == "gower":
        ranges = np.ptp(imputed[fit], axis=0)
        distances = _gower_like(imputed, ranges)
    else:
        distances = pairwise_distances(transformed, metric=metric)
    np.fill_diagonal(distances, np.inf)

    adjacency = np.zeros_like(distances)
    if method in {"knn", "functional"}:
        candidates = np.unique(fit) if strict_inductive else np.arange(len(X))
        if k >= len(candidates):
            raise ValueError("k must be smaller than the training reference set")
        for row in range(len(X)):
            allowed = candidates[candidates != row]
            if len(allowed) < k:
                raise ValueError("Too few training reference nodes for the requested k")
            neighbors = allowed[np.argpartition(distances[row, allowed], kth=k - 1)[:k]]
            adjacency[row, neighbors] = np.exp(-distances[row, neighbors])
    elif method == "radius":
        if threshold is None or threshold <= 0:
            raise ValueError("radius construction requires threshold > 0")
        mask = distances <= threshold
        adjacency[mask] = np.exp(-distances[mask])
    else:
        raise ValueError(f"Unsupported graph method: {method}")
    adjacency = _symmetrize(adjacency, symmetrize)
    if self_loops:
        np.fill_diagonal(adjacency, 1.0)
    else:
        np.fill_diagonal(adjacency, 0.0)
    sources, targets = np.nonzero(adjacency)
    return GraphBuildResult(
        edge_index=np.vstack([sources, targets]).astype(np.int64),
        edge_weight=adjacency[sources, targets].astype(np.float32),
        transformed_X=transformed.astype(np.float32), fit_indices=fit.copy(),
        metadata={"method": method, "metric": metric, "k": k, "symmetrize": symmetrize,
                  "self_loops": self_loops, "strict_inductive": strict_inductive},
    )

"""Deterministic non-clinical fixtures for CI and pipeline demonstrations."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SyntheticCohort:
    X: np.ndarray
    y: np.ndarray
    patient_ids: np.ndarray
    sex: np.ndarray
    age: np.ndarray


def make_synthetic_cohort(n_samples: int, n_features: int, seed: int) -> SyntheticCohort:
    """Create a binary task with no claim of clinical realism."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, 4))
    loadings = rng.normal(size=(4, n_features))
    X = latent @ loadings + rng.normal(scale=0.8, size=(n_samples, n_features))
    score = 0.9 * latent[:, 0] - 0.6 * latent[:, 1] + rng.normal(size=n_samples)
    y = (score > np.quantile(score, 0.68)).astype(np.int64)
    return SyntheticCohort(
        X=X.astype(np.float32), y=y, patient_ids=np.arange(n_samples),
        sex=rng.integers(0, 2, n_samples), age=rng.integers(30, 90, n_samples),
    )

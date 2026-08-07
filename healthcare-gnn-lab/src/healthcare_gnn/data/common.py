"""Shared data contracts and leakage guards."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class PatientSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def validate(self, patient_ids: np.ndarray) -> None:
        groups = [set(patient_ids[index].tolist()) for index in (self.train, self.validation, self.test)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("Patient leakage: a patient occurs in multiple splits")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def patient_split(patient_ids: Iterable[object], seed: int = 0) -> PatientSplit:
    """Create deterministic 60/20/20 group-separated indices."""
    ids = np.asarray(list(patient_ids))
    all_indices = np.arange(len(ids))
    outer = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=seed)
    train, remainder = next(outer.split(all_indices, groups=ids))
    inner = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    validation_local, test_local = next(inner.split(remainder, groups=ids[remainder]))
    split = PatientSplit(train, remainder[validation_local], remainder[test_local])
    split.validate(ids)
    return split


def assert_events_before_index(event_times: np.ndarray, index_times: np.ndarray) -> None:
    """Reject EHR features recorded at or after their prediction index."""
    if np.any(np.asarray(event_times) >= np.asarray(index_times)):
        raise ValueError("Temporal leakage: feature event is not before prediction index")

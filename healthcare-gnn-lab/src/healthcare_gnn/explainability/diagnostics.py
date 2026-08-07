"""Interpretability policy shared by reports."""
from __future__ import annotations

def explanation_warning(method: str) -> str:
    return f"{method} is a descriptive model diagnostic, not a causal or clinical explanation."

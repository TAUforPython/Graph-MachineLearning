"""Non-graph baselines."""
from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def make_baseline(name: str, seed: int):
    if name == "logistic": return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    if name == "random_forest": return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1)
    if name == "mlp": return MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=seed)
    raise ValueError(f"Unknown baseline: {name}")

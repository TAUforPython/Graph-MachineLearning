import numpy as np
from healthcare_gnn.evaluation.metrics import classification_metrics

def test_label_permutation_constant_predictor_is_chance_level():
    rng=np.random.default_rng(2); y=np.array([0]*70+[1]*30); rng.shuffle(y)
    probability=np.full(100,0.3); metrics=classification_metrics(y,probability,0.5)
    assert abs(metrics["auprc"]-0.3)<1e-12
    assert abs(metrics["balanced_accuracy"]-0.5)<1e-12

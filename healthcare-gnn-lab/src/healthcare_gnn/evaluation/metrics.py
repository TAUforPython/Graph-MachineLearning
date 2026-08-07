"""Leakage-safe classification metrics and patient-level uncertainty."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
 confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)


def select_f1_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates=np.linspace(0.05,0.95,91)
    scores=[f1_score(y,probability>=t,zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def classification_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str,float|list[list[int]]]:
    y=np.asarray(y); p=np.asarray(probability); pred=p>=threshold; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"auprc":float(average_precision_score(y,p)),"auroc":float(roc_auc_score(y,p)),
      "f1":float(f1_score(y,pred,zero_division=0)),"sensitivity":float(recall_score(y,pred,zero_division=0)),
      "specificity":float(tn/max(1,tn+fp)),"precision":float(precision_score(y,pred,zero_division=0)),
      "balanced_accuracy":float(balanced_accuracy_score(y,pred)),"brier":float(brier_score_loss(y,p)),
      "confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]}


def bootstrap_auprc(y: np.ndarray, probability: np.ndarray, patient_ids: np.ndarray, seed: int, repeats: int=200) -> list[float]:
    """Patient-cluster bootstrap percentile interval."""
    rng=np.random.default_rng(seed); ids=np.unique(patient_ids); values=[]
    for _ in range(repeats):
        sampled=rng.choice(ids,size=len(ids),replace=True); idx=np.concatenate([np.flatnonzero(patient_ids==p) for p in sampled])
        if np.unique(y[idx]).size==2: values.append(average_precision_score(y[idx],probability[idx]))
    return [float(x) for x in np.quantile(values,[.025,.975])] if values else [float("nan"),float("nan")]

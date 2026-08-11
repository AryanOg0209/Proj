"""Shared classification-evaluation helpers used by both the sklearn and
PyTorch CTR models, so both report metrics the same way."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    log_loss: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_binary(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> ClassificationMetrics:
    y_pred = (y_pred_proba >= threshold).astype(int)
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_pred_proba),
        log_loss=log_loss(y_true, y_pred_proba, labels=[0, 1]),
    )

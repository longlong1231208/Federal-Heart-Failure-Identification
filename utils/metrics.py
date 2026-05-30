# src/utils/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)


@dataclass
class EvalResult:
    auc: float
    auprc: float
    f1: float
    recall: float
    precision: float
    threshold: float
    y_true: np.ndarray
    y_prob: np.ndarray


@torch.no_grad()
def predict_proba(model: nn.Module, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device)
        prob = torch.sigmoid(model(x)).detach().cpu().numpy()
        y_true.extend(y.numpy())
        y_prob.extend(prob)
    return np.asarray(y_true), np.asarray(y_prob)


def search_best_threshold_by_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: Tuple[float, float, int] = (0.1, 0.6, 51),
) -> float:
    lo, hi, n = grid
    best_f1 = -1.0
    best_t = 0.5
    for t in np.linspace(lo, hi, n):
        y_pred = (y_prob > t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def evaluate_metrics(
    model: nn.Module,
    loader,
    device,
    threshold: Optional[float] = None,
    threshold_grid: Tuple[float, float, int] = (0.1, 0.6, 51),
) -> EvalResult:
    """
    threshold=None -> Validation mode: search best threshold on this loader (by F1).
    threshold=float -> Test mode: use fixed threshold, NEVER search.
    """
    y_true, y_prob = predict_proba(model, loader, device)

    if len(np.unique(y_true)) < 2:
        # cannot compute AUC/AUPRC properly
        t = 0.5 if threshold is None else float(threshold)
        return EvalResult(
            auc=0.5, auprc=0.0, f1=0.0, recall=0.0, precision=0.0,
            threshold=t, y_true=y_true, y_prob=y_prob
        )

    auc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))

    if threshold is None:
        t = search_best_threshold_by_f1(y_true, y_prob, threshold_grid)
    else:
        t = float(threshold)

    y_pred = (y_prob > t).astype(int)
    return EvalResult(
        auc=auc,
        auprc=auprc,
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        threshold=t,
        y_true=y_true,
        y_prob=y_prob,
    )


def gini(scores: list[float]) -> float:
    if len(scores) == 0:
        return 0.0
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    idx = np.arange(1, n + 1)
    denom = np.sum(s)
    if denom == 0:
        return 0.0
    return float((2 * np.sum(idx * s)) / (n * denom) - (n + 1) / n)
"""Выбор порога классификации по out-of-fold предсказаниям."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

DEFAULT_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def threshold_curve(y_true, proba, grid=DEFAULT_GRID) -> list[dict]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    return [
        {"threshold": float(t), "f1": float(f1_score(y_true, (proba >= t).astype(int), zero_division=0))}
        for t in grid
    ]


def select_threshold(y_true, proba, grid=DEFAULT_GRID) -> float:
    """Порог с максимальным F1. При равенстве берётся меньший порог."""
    curve = threshold_curve(y_true, proba, grid)
    best = max(curve, key=lambda row: (row["f1"], -row["threshold"]))
    return best["threshold"]

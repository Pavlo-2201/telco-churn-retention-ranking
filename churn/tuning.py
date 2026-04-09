"""Кросс-валидация и ограниченный подбор гиперпараметров."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from churn.models import make_model

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
SELECTION_METRIC = "pr_auc"

# Небольшие сетки: несколько осмысленных значений самых влиятельных параметров.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "dummy": {},
    "logreg": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
    },
    "random_forest": {
        "n_estimators": [300],
        "max_depth": [None, 8, 12],
        "min_samples_leaf": [1, 5, 20],
        "max_features": ["sqrt", 0.5],
    },
    "catboost": {
        "iterations": [500],
        "depth": [4, 6],
        "learning_rate": [0.03, 0.1],
        "l2_leaf_reg": [3, 10],
    },
}


def make_cv(n_splits: int = 5, seed: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def search_model(name: str, X: pd.DataFrame, y: pd.Series, cv, seed: int = 42, grid=None) -> dict:
    """Перебирает сетку для одной модели и возвращает лучшие параметры и все кандидаты."""
    grid = PARAM_GRIDS[name] if grid is None else grid
    pipeline = make_model(name, seed=seed)
    if not grid:
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING)
        row = {}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"std_{metric}"] = float(np.std(values))
            row[f"folds_{metric}"] = [float(v) for v in values]
        return {"best_params": {}, "cv": row, "candidates": [{"params": {}, **row}]}

    search = GridSearchCV(
        pipeline,
        {f"model__{key}": values for key, values in grid.items()},
        scoring=SCORING,
        refit=SELECTION_METRIC,
        cv=cv,
        n_jobs=1,
    )
    search.fit(X, y)
    res = search.cv_results_
    n_folds = cv.get_n_splits()

    def row_at(i: int) -> dict:
        row = {}
        for metric in SCORING:
            row[f"mean_{metric}"] = float(res[f"mean_test_{metric}"][i])
            row[f"std_{metric}"] = float(res[f"std_test_{metric}"][i])
            row[f"folds_{metric}"] = [float(res[f"split{k}_test_{metric}"][i]) for k in range(n_folds)]
        return row

    candidates = [
        {"params": {k.removeprefix("model__"): v for k, v in params.items()}, **row_at(i)}
        for i, params in enumerate(res["params"])
    ]
    best = search.best_index_
    return {"best_params": candidates[best]["params"], "cv": row_at(best), "candidates": candidates}

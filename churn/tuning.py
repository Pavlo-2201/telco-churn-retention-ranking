"""Кросс-валидация, out-of-fold предсказания и подбор гиперпараметров."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold, StratifiedKFold

from churn.metrics import LOWER_IS_BETTER, all_metrics
from churn.models import make_estimator


def make_cv(n_splits: int = 5, seed: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_repeated_cv(n_splits: int = 5, n_repeats: int = 5, seed: int = 42) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def _n_iterations(estimator) -> float:
    model = getattr(estimator, "named_steps", {}).get("model")
    return float(getattr(model, "n_iterations_", np.nan))


def cross_validate_spec(
    spec: dict, X: pd.DataFrame, y: pd.Series, cv, seed: int = 42, fractions=(0.1, 0.2), segments=None
) -> pd.DataFrame:
    """Метрики модели на каждом фолде. segments (например, Contract) добавляет
    ROC-AUC внутри каждого сегмента тестовой части фолда."""
    template = make_estimator(spec, seed=seed)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        model = clone(template).fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        y_te = y.iloc[te].to_numpy()
        row = {"fold": fold, "n_train": len(tr), "n_test": len(te), **all_metrics(y_te, proba, fractions)}
        row["n_iterations"] = _n_iterations(model)
        if segments is not None:
            seg = np.asarray(segments)[te]
            for value in np.unique(seg):
                mask = seg == value
                row[f"roc_auc[{value}]"] = (
                    all_metrics(y_te[mask], proba[mask], ())["roc_auc"] if len(np.unique(y_te[mask])) == 2 else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def oof_predict(spec: dict, X: pd.DataFrame, y: pd.Series, cv, seed: int = 42) -> np.ndarray:
    """Out-of-fold вероятности: каждая строка предсказана моделью, которая её не видела."""
    template = make_estimator(spec, seed=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(X, y):
        model = clone(template).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof


def summarize(folds: pd.DataFrame) -> dict[str, float]:
    metric_columns = [c for c in folds.columns if c not in ("fold", "n_train", "n_test")]
    out = {}
    for column in metric_columns:
        out[f"mean_{column}"] = float(folds[column].mean())
        out[f"std_{column}"] = float(folds[column].std(ddof=1)) if len(folds) > 1 else 0.0
    return out


def search_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    grid: dict,
    seed: int = 42,
    features=(),
    fractions=(0.1, 0.2),
    metric: str = "log_loss",
) -> dict:
    """Перебирает сетку для одной модели. Пустая сетка — модель без параметров.
    Лучшая комбинация — по среднему metric на фолдах (log-loss: меньше — лучше)."""
    candidates = []
    for params in ParameterGrid(grid) if grid else [{}]:
        spec = {"name": name, "params": dict(params), "features": list(features)}
        folds = cross_validate_spec(spec, X, y, cv, seed=seed, fractions=fractions)
        candidates.append({"params": dict(params), **summarize(folds)})
    sign = 1 if metric in LOWER_IS_BETTER else -1
    best = min(range(len(candidates)), key=lambda i: sign * candidates[i][f"mean_{metric}"])
    return {"best_params": candidates[best]["params"], "cv": candidates[best], "candidates": candidates}

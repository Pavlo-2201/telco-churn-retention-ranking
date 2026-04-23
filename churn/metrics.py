"""Метрики: ранжирование (lift, precision/recall на верхних K%), вероятностные
(ROC-AUC, PR-AUC, Brier, log-loss), калибровка и метрики по порогу."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOWER_IS_BETTER = {"log_loss", "brier"}


def frac_label(fraction: float) -> str:
    return f"{round(fraction * 100):g}"


def top_k(n: int, fraction: float) -> int:
    """Сколько клиентов попадает в верхние fraction списка."""
    return max(1, int(round(n * fraction)))


def ranking_order(score) -> np.ndarray:
    """Индексы по убыванию оценки; при равенстве сохраняется порядок строк."""
    return np.argsort(-np.asarray(score, dtype=float), kind="stable")


def precision_at(y_true, score, fraction: float) -> float:
    y_true = np.asarray(y_true)
    k = top_k(len(y_true), fraction)
    return float(y_true[ranking_order(score)[:k]].mean())


def recall_at(y_true, score, fraction: float) -> float:
    y_true = np.asarray(y_true)
    n_pos = y_true.sum()
    if n_pos == 0:
        return float("nan")
    k = top_k(len(y_true), fraction)
    return float(y_true[ranking_order(score)[:k]].sum() / n_pos)


def lift_at(y_true, score, fraction: float) -> float:
    """Во сколько раз доля ушедших в верхних fraction списка выше средней."""
    base = np.asarray(y_true).mean()
    if base == 0:
        return float("nan")
    return precision_at(y_true, score, fraction) / base


def ranking_metrics(y_true, score, fractions=(0.1, 0.2)) -> dict[str, float]:
    out = {}
    for fraction in fractions:
        label = frac_label(fraction)
        out[f"lift@{label}"] = lift_at(y_true, score, fraction)
        out[f"precision@{label}"] = precision_at(y_true, score, fraction)
        out[f"recall@{label}"] = recall_at(y_true, score, fraction)
    return out


def _safe_auc(fn, y_true, score) -> float:
    return float(fn(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan")


def probability_metrics(y_true, proba) -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "roc_auc": _safe_auc(roc_auc_score, y_true, proba),
        "pr_auc": _safe_auc(average_precision_score, y_true, proba),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
    }


def all_metrics(y_true, proba, fractions=(0.1, 0.2)) -> dict[str, float]:
    return {**probability_metrics(y_true, proba), **ranking_metrics(y_true, proba, fractions)}


def cumulative_gains(y_true, score) -> tuple[np.ndarray, np.ndarray]:
    """Доля охваченной базы -> доля найденных ушедших (обе от 0 до 1)."""
    y_true = np.asarray(y_true)
    captured = np.concatenate([[0], np.cumsum(y_true[ranking_order(score)])])
    contacted = np.arange(len(y_true) + 1) / len(y_true)
    return contacted, captured / max(y_true.sum(), 1)


def calibration_summary(y_true, proba, n_bins: int = 10) -> dict:
    """Brier, калибровочные intercept и slope (логистическая регрессия y на logit(p))
    и reliability-кривая по квантильным корзинам. Идеал: intercept 0, slope 1."""
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(proba / (1 - proba)).reshape(-1, 1)
    # Практически без регуляризации: нужна оценка, а не прогноз.
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y_true)
    edges = np.unique(np.quantile(proba, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, proba, side="right") - 1, 0, len(edges) - 2)
    curve = [
        {"mean_predicted": float(proba[bins == b].mean()), "observed_rate": float(y_true[bins == b].mean()),
         "n": int((bins == b).sum())}
        for b in np.unique(bins)
    ]
    return {
        "brier": float(brier_score_loss(y_true, proba)),
        "mean_predicted": float(proba.mean()),
        "observed_rate": float(y_true.mean()),
        "calibration_intercept": float(fit.intercept_[0]),
        "calibration_slope": float(fit.coef_[0, 0]),
        "reliability_curve": curve,
    }


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": _safe_auc(roc_auc_score, y_true, proba),
        "pr_auc": _safe_auc(average_precision_score, y_true, proba),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "contacted_share": float(pred.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

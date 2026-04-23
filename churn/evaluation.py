"""Оценка предсказаний с ответом на бизнес-вопрос: ранжирование, сегменты,
калибровка, прибыль. Везде — bootstrap-интервалы."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.economics import Economics, operating_point, profit_curve, segment_cutoffs, sensitivity
from churn.metrics import (
    all_metrics,
    calibration_summary,
    classification_metrics,
    frac_label,
    lift_at,
    probability_metrics,
)
from churn.stats import bootstrap, with_ci


def segment_metrics(y, score, segments, selected, fractions=(0.1, 0.2)) -> dict[str, float]:
    """Плоский словарь "сегмент|метрика". Ранжирующие метрики считаются внутри сегмента
    (lift относительно доли ушедших в сегменте); selected — клиенты, выбранные
    глобальным правилом (top-K всей базы)."""
    out = {}
    n_selected = max(selected.sum(), 1)
    for value in np.unique(segments):
        mask = segments == value
        y_s, p_s = y[mask], score[mask]
        n_pos = y_s.sum()
        prob = probability_metrics(y_s, p_s) if len(np.unique(y_s)) == 2 else {"roc_auc": np.nan, "pr_auc": np.nan}
        row = {
            "n": float(mask.sum()),
            "n_churned": float(n_pos),
            "churn_rate": float(y_s.mean()),
            "mean_predicted": float(p_s.mean()),
            "roc_auc": prob["roc_auc"],
            "pr_auc": prob["pr_auc"],
            "share_of_selected": float(selected[mask].sum() / n_selected),
            # доля ушедших сегмента, которых глобальный top-K не охватывает
            "missed_by_global_top": float(1 - (y_s * selected[mask]).sum() / n_pos) if n_pos else np.nan,
        }
        for f in fractions:
            row[f"lift@{frac_label(f)}"] = lift_at(y_s, p_s, f) if n_pos else np.nan
        out.update({f"{value}|{k}": v for k, v in row.items()})
    return out


def segment_table(flat: dict, ci: dict | None = None) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for key, value in flat.items():
        segment, metric = key.split("|", 1)
        row = rows.setdefault(segment, {"segment": segment})
        row[metric] = value
        if ci is not None:
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = ci[key]
    return pd.DataFrame(rows.values())


def evaluate_scores(y, score, segments, econ: Economics, config: dict, threshold: float | None = None) -> dict:
    """Полный набор метрик для одной выборки предсказаний (OOF или test).

    threshold=None: точка работы выбирается здесь по прибыли (только для OOF).
    Иначе используется заданный порог, выбранный ранее на OOF.
    """
    y, score, segments = np.asarray(y), np.asarray(score, dtype=float), np.asarray(segments)
    fractions = tuple(config["ranking"]["top_fractions"])
    boot = config["bootstrap"]
    business = config["business"]

    if threshold is None:
        point = operating_point(y, score, econ, business["budget_fraction"])
        threshold = point["threshold"]
    else:
        profit = (y * (score >= threshold)).sum() * econ.gain_per_churner - (
            (1 - y) * (score >= threshold)
        ).sum() * econ.loss_per_non_churner
        point = {"threshold": threshold, "k_fraction": float((score >= threshold).mean()), "profit": float(profit),
                 "profit_per_customer_in_base": float(profit / len(y))}
    selected = score >= threshold

    def stat(idx):
        y_b, s_b = y[idx], score[idx]
        sel = s_b >= threshold
        tp, fp = (y_b * sel).sum(), ((1 - y_b) * sel).sum()
        return {
            **all_metrics(y_b, s_b, fractions),
            "precision_at_threshold": float(tp / max(sel.sum(), 1)),
            "recall_at_threshold": float(tp / max(y_b.sum(), 1)),
            "profit_per_customer_in_base": float(
                (tp * econ.gain_per_churner - fp * econ.loss_per_non_churner) / len(idx)
            ),
        }

    overall_point = stat(np.arange(len(y)))
    overall = with_ci(overall_point, bootstrap(stat, len(y), boot["n_resamples"], boot["seed"], boot["confidence"]))

    seg_point = segment_metrics(y, score, segments, selected, fractions)
    seg_ci = bootstrap(lambda idx: segment_metrics(y[idx], score[idx], segments[idx], selected[idx], fractions),
                       len(y), boot["n_resamples"], boot["seed"], boot["confidence"])
    return {
        "n": int(len(y)),
        "churn_rate": float(y.mean()),
        "metrics": overall,
        "operating_point": point,
        "at_threshold": classification_metrics(y, score, threshold),
        "segments": segment_table(seg_point, seg_ci),
    }


def business_analysis(y, score, segments, econ: Economics, config: dict) -> dict:
    business = config["business"]
    return {
        "sensitivity": sensitivity(y, score, econ, business["sensitivity_factors"], business["budget_fraction"]),
        "segment_cutoffs": segment_cutoffs(y, score, segments, econ),
        "profit_curve": profit_curve(y, score, econ),
    }


def calibration_report(y, raw, calibrated, config: dict) -> dict:
    """Сравнение Brier без калибратора и с ним; применять ли калибровку."""
    y, raw, calibrated = np.asarray(y), np.asarray(raw), np.asarray(calibrated)
    boot = config["bootstrap"]
    n_bins = config["calibration"]["n_bins"]

    def stat(idx):
        brier_raw = np.mean((raw[idx] - y[idx]) ** 2)
        return {"brier_improvement": float(brier_raw - np.mean((calibrated[idx] - y[idx]) ** 2))}

    point = stat(np.arange(len(y)))
    improvement = with_ci(point, bootstrap(stat, len(y), boot["n_resamples"], boot["seed"], boot["confidence"]))
    apply = config["calibration"]["apply"]
    if apply == "auto":
        applied = improvement["brier_improvement"]["ci_low"] > 0
    else:
        applied = apply == "always"
    return {
        "method": config["calibration"]["method"],
        "rule": "auto: калибратор применяется, если 95% CI улучшения Brier на OOF целиком больше 0",
        "raw": calibration_summary(y, raw, n_bins),
        "calibrated": calibration_summary(y, calibrated, n_bins),
        "brier_improvement": improvement["brier_improvement"],
        "applied": bool(applied),
    }

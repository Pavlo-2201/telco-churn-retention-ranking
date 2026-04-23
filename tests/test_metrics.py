import numpy as np
import pytest

from churn.metrics import (
    calibration_summary,
    classification_metrics,
    cumulative_gains,
    lift_at,
    precision_at,
    ranking_metrics,
    recall_at,
    top_k,
)
from churn.stats import bootstrap, corrected_ttest

Y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
SCORE = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])


def test_metrics_on_known_example():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = classification_metrics(y, proba, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["roc_auc"] == 0.75


def test_ranking_metrics_on_known_example():
    # top-20% = 2 клиента: один ушедший из двух; всего ушедших 3 из 10
    assert top_k(10, 0.2) == 2
    assert precision_at(Y, SCORE, 0.2) == 0.5
    assert recall_at(Y, SCORE, 0.2) == pytest.approx(1 / 3)
    assert lift_at(Y, SCORE, 0.2) == pytest.approx(0.5 / 0.3)
    m = ranking_metrics(Y, SCORE, (0.1, 0.2))
    assert m["lift@10"] == pytest.approx(1 / 0.3)
    assert set(m) == {f"{k}@{f}" for k in ("lift", "precision", "recall") for f in (10, 20)}


def test_perfect_and_random_ranking():
    y = np.array([1] * 10 + [0] * 90)
    assert lift_at(y, -np.arange(100), 0.1) == pytest.approx(10.0)  # все ушедшие наверху
    assert lift_at(y, np.zeros(100), 0.1) == pytest.approx(10.0)  # равные оценки: порядок строк
    assert lift_at(y[::-1], np.zeros(100), 0.1) == 0.0


def test_cumulative_gains_endpoints():
    contacted, captured = cumulative_gains(Y, SCORE)
    assert contacted[0] == 0 and captured[0] == 0
    assert contacted[-1] == 1 and captured[-1] == 1
    assert np.all(np.diff(captured) >= 0)


def test_calibration_summary_on_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 20000)
    y = (rng.random(20000) < p).astype(int)
    s = calibration_summary(y, p, n_bins=10)
    assert abs(s["calibration_slope"] - 1) < 0.1 and abs(s["calibration_intercept"]) < 0.1
    assert len(s["reliability_curve"]) == 10
    shifted = calibration_summary(y, np.clip(p * 0.6, 0.01, 0.99))
    assert shifted["brier"] > s["brier"] and shifted["observed_rate"] > shifted["mean_predicted"]


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(1)
    x = rng.normal(5, 1, 500)
    ci = bootstrap(lambda idx: {"mean": x[idx].mean()}, len(x), n_resamples=300, seed=0)["mean"]
    assert ci[0] < x.mean() < ci[1]
    assert ci[1] - ci[0] < 0.3


def test_corrected_ttest_is_more_conservative():
    from scipy import stats

    rng = np.random.default_rng(2)
    diffs = rng.normal(0.01, 0.02, 25)
    res = corrected_ttest(diffs, n_train=4000, n_test=1000)
    naive_p = stats.ttest_1samp(diffs, 0).pvalue
    assert res["p_value"] > naive_p
    assert res["ci_low"] < res["mean_diff"] < res["ci_high"]
    # ручной расчёт поправки Nadeau-Bengio
    se = np.sqrt((1 / 25 + 1000 / 4000) * diffs.var(ddof=1))
    assert res["t"] == pytest.approx(diffs.mean() / se)


def test_corrected_ttest_ignores_undefined_folds():
    res = corrected_ttest([0.1, np.nan, 0.2, 0.15], n_train=80, n_test=20)
    assert res["n_folds"] == 3

import numpy as np

from churn.threshold import select_threshold, threshold_curve


def test_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7])
    t = select_threshold(y, proba)
    assert 0.3 < t <= 0.6
    assert max(r["f1"] for r in threshold_curve(y, proba)) == 1.0


def test_threshold_is_lowered_for_rare_class_with_low_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.25).astype(int)
    proba = np.clip(0.15 + 0.25 * y + rng.normal(0, 0.1, 2000), 0, 1)
    assert select_threshold(y, proba) < 0.5


def test_threshold_ties_choose_smaller_value():
    y = np.array([0, 1])
    proba = np.array([0.1, 0.9])
    assert select_threshold(y, proba) == 0.11

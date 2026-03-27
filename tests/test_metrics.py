import numpy as np

from churn.metrics import classification_metrics


def test_metrics_on_known_example():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = classification_metrics(y, proba, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["roc_auc"] == 0.75


def test_threshold_changes_predictions():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = classification_metrics(y, proba, threshold=0.3)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}

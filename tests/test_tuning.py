import numpy as np
import pytest
from sklearn.model_selection import cross_validate

from churn.data import NUMERIC_FEATURES
from churn.models import make_model
from churn.tuning import cross_validate_spec, make_cv, make_repeated_cv, oof_predict, search_model


def test_preprocessing_is_refit_inside_each_fold(dataset):
    X, y, _ = dataset
    cv = make_cv(n_splits=3, seed=0)
    out = cross_validate(make_model("logreg"), X, y, cv=cv, return_estimator=True, return_indices=True)
    for estimator, train_idx in zip(out["estimator"], out["indices"]["train"]):
        num = estimator.named_steps["prep"].named_transformers_["num"]
        fold_train = X.iloc[train_idx][NUMERIC_FEATURES]
        np.testing.assert_allclose(num.named_steps["scale"].mean_, fold_train.mean())
        assert not np.allclose(num.named_steps["scale"].mean_, X[NUMERIC_FEATURES].mean())


def test_cv_folds_are_stratified_and_disjoint(dataset):
    X, y, _ = dataset
    seen = []
    for train_idx, valid_idx in make_cv(n_splits=5, seed=0).split(X, y):
        assert set(train_idx).isdisjoint(valid_idx)
        assert abs(y.iloc[valid_idx].mean() - y.mean()) < 0.03
        seen.extend(valid_idx)
    assert sorted(seen) == list(range(len(X)))


def test_repeated_cv_returns_metrics_per_fold(dataset):
    X, y, _ = dataset
    folds = cross_validate_spec({"name": "logreg", "params": {}}, X, y, make_repeated_cv(3, 2, 0),
                                segments=X["Contract"].to_numpy())
    assert len(folds) == 6
    for column in ("lift@10", "pr_auc", "log_loss", "brier", "roc_auc[One year]"):
        assert column in folds
    assert (folds["n_train"] + folds["n_test"] == len(X)).all()


def test_oof_predictions_are_out_of_fold(dataset):
    """Если подменить цель строк одного фолда, их OOF-предсказания не меняются."""
    X, y, _ = dataset
    cv = make_cv(3, seed=0)
    a = oof_predict({"name": "logreg", "params": {}}, X, y, cv)
    _, first_test = next(cv.split(X, y))
    spec = {"name": "logreg", "params": {}}
    y_flipped = y.copy()
    y_flipped.iloc[first_test] = 1 - y_flipped.iloc[first_test]
    # стратификация зависит от y, поэтому фиксируем те же фолды явно
    folds = list(cv.split(X, y))
    b = oof_predict(spec, X, y_flipped, _FixedFolds(folds))
    np.testing.assert_allclose(a[first_test], b[first_test])
    assert not np.allclose(a, b)


class _FixedFolds:
    def __init__(self, folds):
        self.folds = folds

    def split(self, X, y=None):
        return iter(self.folds)


@pytest.mark.parametrize(
    "name,grid",
    [("logreg", {"C": [0.1, 1.0]}), ("catboost", {"iterations": [30], "depth": [2, 3], "one_hot_max_size": [4]}),
     ("dummy", {})],
)
def test_search_model_returns_best_by_log_loss(dataset, name, grid):
    X, y, _ = dataset
    result = search_model(name, X, y, make_cv(n_splits=3, seed=0), grid=grid)
    best = min(result["candidates"], key=lambda c: c["mean_log_loss"])
    assert result["best_params"] == best["params"]
    assert len(result["candidates"]) == max(1, int(np.prod([len(v) for v in grid.values()])))

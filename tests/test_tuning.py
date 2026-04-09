import numpy as np
import pytest
from sklearn.model_selection import cross_validate

from churn.data import NUMERIC_FEATURES, load_dataset
from churn.models import make_model
from churn.tuning import make_cv, search_model
from tests.conftest import DATA_PATH


@pytest.fixture(scope="module")
def dev(raw_df):
    X, y, _ = load_dataset(DATA_PATH)
    return X.iloc[:1500].reset_index(drop=True), y.iloc[:1500].reset_index(drop=True)


def test_preprocessing_is_refit_inside_each_fold(dev):
    X, y = dev
    cv = make_cv(n_splits=3, seed=0)
    out = cross_validate(make_model("logreg"), X, y, cv=cv, return_estimator=True, return_indices=True)
    for estimator, train_idx in zip(out["estimator"], out["indices"]["train"]):
        num = estimator.named_steps["prep"].named_transformers_["num"]
        fold_train = X.iloc[train_idx][NUMERIC_FEATURES]
        np.testing.assert_allclose(num.named_steps["scale"].mean_, fold_train.mean())
        assert not np.allclose(num.named_steps["scale"].mean_, X[NUMERIC_FEATURES].mean())


def test_cv_folds_are_stratified_and_disjoint(dev):
    X, y = dev
    seen = []
    for train_idx, valid_idx in make_cv(n_splits=5, seed=0).split(X, y):
        assert set(train_idx).isdisjoint(valid_idx)
        assert abs(y.iloc[valid_idx].mean() - y.mean()) < 0.02
        seen.extend(valid_idx)
    assert sorted(seen) == list(range(len(X)))


@pytest.mark.parametrize(
    "name,grid",
    [("logreg", {"C": [0.1, 1.0]}), ("catboost", {"iterations": [30], "depth": [3, 4]}), ("dummy", {})],
)
def test_search_model_returns_best_of_candidates(dev, name, grid):
    X, y = dev
    result = search_model(name, X, y, make_cv(n_splits=3, seed=0), grid=grid)
    best = max(result["candidates"], key=lambda c: c["mean_pr_auc"])
    assert result["best_params"] == best["params"]
    assert len(result["cv"]["folds_pr_auc"]) == 3

import numpy as np
import pandas as pd
import pytest

from churn.data import NUMERIC_FEATURES, load_dataset
from churn.models import make_model
from churn.splits import DEFAULT_SPLITS_PATH, make_splits, split_masks
from tests.conftest import DATA_PATH, ROOT


def test_splits_are_disjoint_and_complete(dataset):
    X, y, ids = dataset
    splits = make_splits(ids, y, seed=42)
    parts = {name: set(g["customerID"]) for name, g in splits.groupby("split")}
    assert set(parts) == {"train", "test"}
    assert parts["train"].isdisjoint(parts["test"])
    assert parts["train"] | parts["test"] == set(ids)


def test_splits_are_stratified_and_sized(dataset):
    X, y, ids = dataset
    masks = split_masks(ids, make_splits(ids, y, seed=42))
    assert abs(masks["test"].mean() - 0.2) < 0.01
    assert abs(y[masks["test"]].mean() - y.mean()) < 0.02


def test_splits_are_reproducible(dataset):
    X, y, ids = dataset
    a = make_splits(ids, y, seed=42)
    pd.testing.assert_frame_equal(a, make_splits(ids, y, seed=42))
    assert not a.equals(make_splits(ids, y, seed=7))


def test_saved_splits_match_seed(raw_df):
    """results/splits.csv совпадает с make_splits(seed=42), и test тот же, что в первой версии."""
    path = ROOT / DEFAULT_SPLITS_PATH
    if not path.exists():
        pytest.skip("results/splits.csv ещё не создан")
    X, y, ids = load_dataset(DATA_PATH)
    saved = pd.read_csv(path).set_index("customerID")["split"].sort_index()
    fresh = make_splits(ids, y, seed=42).set_index("customerID")["split"].sort_index()
    pd.testing.assert_series_equal(saved, fresh)


def test_preprocessing_statistics_come_from_train_only(dataset):
    X, y, ids = dataset
    masks = split_masks(ids, make_splits(ids, y, seed=42))
    X_train, y_train = X[masks["train"]], y[masks["train"]]
    model = make_model("logreg").fit(X_train, y_train)
    num = model.named_steps["prep"].named_transformers_["num"]
    np.testing.assert_allclose(num.named_steps["impute"].statistics_, X_train[NUMERIC_FEATURES].median())
    np.testing.assert_allclose(num.named_steps["scale"].mean_, X_train[NUMERIC_FEATURES].mean())


def test_changing_test_rows_does_not_change_model(dataset):
    """Если испортить строки test, модель, обученная на train, не меняется."""
    X, y, ids = dataset
    masks = split_masks(ids, make_splits(ids, y, seed=42))
    X_train, y_train = X[masks["train"]], y[masks["train"]]
    model_a = make_model("logreg").fit(X_train, y_train)
    X_spoiled = X.copy()
    X_spoiled.loc[masks["test"], "MonthlyCharges"] = 1e6
    model_b = make_model("logreg").fit(X_spoiled[masks["train"]], y_train)
    np.testing.assert_allclose(model_a.predict_proba(X_train), model_b.predict_proba(X_train))


def test_unknown_category_and_missing_values_at_inference(dataset):
    X, y, _ = dataset
    model = make_model("logreg", features=["contract_payment", "avg_charge"]).fit(X, y)
    sample = X.iloc[:3].copy()
    sample.loc[sample.index[0], "PaymentMethod"] = "Crypto"
    sample.loc[sample.index[1], "TotalCharges"] = np.nan
    proba = model.predict_proba(sample)[:, 1]
    assert np.isfinite(proba).all()

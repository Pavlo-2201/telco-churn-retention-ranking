import numpy as np
import pytest

from churn.data import CATEGORICAL_FEATURES, load_dataset
from churn.features import MISSING_CATEGORY, to_catboost_frame
from churn.models import MODEL_NAMES, make_model
from tests.conftest import DATA_PATH

FAST_PARAMS = {
    "random_forest": {"n_estimators": 30},
    "catboost": {"iterations": 50},
}


@pytest.fixture(scope="module")
def small(raw_df):
    X, y, ids = load_dataset(DATA_PATH)
    return X.iloc[:1500], y.iloc[:1500], X.iloc[1500:1800]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_outputs_probabilities(small, name):
    X, y, X_new = small
    model = make_model(name, **FAST_PARAMS.get(name, {})).fit(X, y)
    proba = model.predict_proba(X_new)[:, 1]
    assert proba.shape == (len(X_new),)
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize("name", ["logreg", "random_forest", "catboost"])
def test_training_is_reproducible(small, name):
    X, y, X_new = small
    a = make_model(name, seed=1, **FAST_PARAMS.get(name, {})).fit(X, y).predict_proba(X_new)
    b = make_model(name, seed=1, **FAST_PARAMS.get(name, {})).fit(X, y).predict_proba(X_new)
    np.testing.assert_allclose(a, b)


def test_catboost_frame_is_stateless_and_handles_missing(small):
    X, _, _ = small
    sample = X.head(3).copy()
    sample.loc[sample.index[0], "Contract"] = None
    out = to_catboost_frame(sample)
    assert out.loc[sample.index[0], "Contract"] == MISSING_CATEGORY
    assert all(out[c].map(type).eq(str).all() for c in CATEGORICAL_FEATURES)
    assert out["tenure"].dtype == sample["tenure"].dtype
    # исходный DataFrame не изменён
    assert sample["Contract"].isna().iloc[0]


@pytest.mark.parametrize("name", ["random_forest", "catboost"])
def test_tree_models_accept_unknown_category(small, name):
    X, y, X_new = small
    model = make_model(name, **FAST_PARAMS[name]).fit(X, y)
    sample = X_new.head(2).copy()
    sample.loc[sample.index[0], "PaymentMethod"] = "Crypto"
    assert np.isfinite(model.predict_proba(sample)).all()


def test_tree_preprocessor_fitted_on_train_only(small):
    X, y, _ = small
    model = make_model("random_forest", n_estimators=10).fit(X, y)
    encoder = model.named_steps["prep"].named_transformers_["cat"].named_steps["ordinal"]
    for column, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        assert set(categories) == set(X[column].unique())

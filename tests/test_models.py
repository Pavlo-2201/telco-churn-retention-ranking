import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from churn.data import CATEGORICAL_FEATURES
from churn.features import (
    ENGINEERED_FEATURES,
    INTERNET_ADDONS,
    MISSING_CATEGORY,
    FeatureBuilder,
    add_engineered,
    column_groups,
    linear_feature_sources,
    to_catboost_frame,
)
from churn.models import MODEL_NAMES, base_pipeline, make_estimator, make_model

FAST_PARAMS = {
    "random_forest": {"n_estimators": 30},
    "catboost": {"iterations": 50, "one_hot_max_size": 4},
}
ALL_FEATURES = [f for f in ENGINEERED_FEATURES if f != "drop_total_charges"]


@pytest.fixture(scope="module")
def small(dataset):
    X, y, _ = dataset
    return X.iloc[:240], y.iloc[:240], X.iloc[240:]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_outputs_probabilities(small, name):
    X, y, X_new = small
    model = make_model(name, **FAST_PARAMS.get(name, {})).fit(X, y)
    proba = model.predict_proba(X_new)[:, 1]
    assert proba.shape == (len(X_new),)
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize("name", ["logreg", "random_forest", "catboost"])
def test_models_accept_all_engineered_features(small, name):
    X, y, X_new = small
    model = make_model(name, features=ALL_FEATURES, **FAST_PARAMS.get(name, {})).fit(X, y)
    assert np.isfinite(model.predict_proba(X_new)).all()


@pytest.mark.parametrize("name", ["logreg", "random_forest", "catboost"])
def test_training_is_reproducible(small, name):
    X, y, X_new = small
    a = make_model(name, seed=1, **FAST_PARAMS.get(name, {})).fit(X, y).predict_proba(X_new)
    b = make_model(name, seed=1, **FAST_PARAMS.get(name, {})).fit(X, y).predict_proba(X_new)
    np.testing.assert_allclose(a, b)


def test_catboost_early_stopping_picks_iterations(small):
    X, y, _ = small
    model = make_model("catboost", iterations=400, early_stopping_rounds=20, one_hot_max_size=4).fit(X, y)
    n = model.named_steps["model"].n_iterations_
    assert 1 <= n <= 400
    assert model.named_steps["model"].model_.tree_count_ == n


def test_catboost_frame_is_stateless_and_handles_missing(small):
    X, _, _ = small
    sample = X.head(3).copy()
    sample.loc[sample.index[0], "Contract"] = None
    out = to_catboost_frame(sample)
    assert out.loc[sample.index[0], "Contract"] == MISSING_CATEGORY
    assert all(out[c].map(type).eq(str).all() for c in CATEGORICAL_FEATURES)
    assert out["tenure"].dtype == sample["tenure"].dtype
    assert sample["Contract"].isna().iloc[0]  # исходный DataFrame не изменён


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
    encoder = model.named_steps["prep"].named_transformers_["cat"].named_steps["encode"]
    for column, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        assert set(categories) == set(X[column].unique())


def test_engineered_features_are_rowwise(small):
    X, _, _ = small
    full = add_engineered(X, ALL_FEATURES)
    # признак строки не зависит от остальных строк
    one = add_engineered(X.iloc[[5]], ALL_FEATURES)
    pd.testing.assert_frame_equal(one, full.iloc[[5]])
    numeric, binary, categorical = column_groups(ALL_FEATURES)
    assert set(numeric + binary + categorical) <= set(full.columns)


def test_engineered_feature_values():
    X = pd.DataFrame({
        "tenure": [0, 10], "MonthlyCharges": [50.0, 80.0], "TotalCharges": [0.0, 600.0],
        "PhoneService": ["Yes", "No"], "MultipleLines": ["No", "No phone service"],
        "InternetService": ["No", "Fiber optic"], "Contract": ["One year", "Month-to-month"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        **{c: ["No internet service", "Yes"] for c in INTERNET_ADDONS},
    })
    out = add_engineered(X, ["n_services", "no_internet", "avg_charge", "contract_payment", "contract_tenure"])
    assert out["n_services"].tolist() == [1, 6]
    assert out["no_internet"].tolist() == [1, 0]
    assert (out.loc[0, INTERNET_ADDONS] == "No").all()
    assert out["avg_monthly_charge"].tolist() == [0.0, 60.0]
    assert np.isnan(out.loc[0, "charge_growth"]) and np.isclose(out.loc[1, "charge_growth"], 80 / 60 - 1)
    assert out["contract_payment"].tolist() == ["One year | Electronic check", "Month-to-month | Mailed check"]
    assert out["tenure_one_year"].tolist() == [0, 0]
    assert "TotalCharges" not in add_engineered(X, ["drop_total_charges"]).columns
    # avg_charge считается до удаления исходных столбцов
    both = add_engineered(X, ["avg_charge", "drop_total_charges", "drop_monthly_charges"])
    assert {"TotalCharges", "MonthlyCharges"}.isdisjoint(both.columns) and "charge_growth" in both


def test_feature_builder_rejects_unknown_feature(small):
    X, y, _ = small
    with pytest.raises(ValueError, match="Неизвестные признаки"):
        FeatureBuilder(["magic"]).fit(X, y)


def test_tenure_spline_replaces_linear_tenure(small):
    X, y, _ = small
    model = make_model("logreg", tenure_spline=True, n_knots=5).fit(X, y)
    sources = linear_feature_sources(model.named_steps["prep"])
    width = model.named_steps["prep"].transform(model.named_steps["fe"].transform(X.head(2))).shape[1]
    assert len(sources) == width
    assert sources.count("tenure") == 7  # n_knots + degree - 1 базисных функций


def test_calibrated_estimator(small):
    X, y, X_new = small
    est = make_estimator({"name": "logreg", "params": {"C": 1.0}, "calibration": "sigmoid"}).fit(X, y)
    assert isinstance(est, CalibratedClassifierCV)
    assert len(est.calibrated_classifiers_) == 1  # ensemble=False: одна базовая модель на всех данных
    assert np.isfinite(est.predict_proba(X_new)).all()
    assert "prep" in base_pipeline(est).named_steps

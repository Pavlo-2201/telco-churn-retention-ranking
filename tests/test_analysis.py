import numpy as np
import pytest
from scipy.special import logit

from churn.analysis import GROUP_COLUMNS, add_groups, error_table, fold_shap
from churn.models import make_model


def test_groups_cover_all_rows(dataset):
    X, _, _ = dataset
    groups = add_groups(X)
    assert groups["tenure_group"].notna().all()
    assert groups["monthly_charges_group"].notna().all()


def test_error_table_counts_are_consistent(dataset):
    X, y, _ = dataset
    selected = np.random.default_rng(0).random(len(y)) < 0.3
    table = error_table(X, y, selected)
    assert set(table["feature"]) == set(GROUP_COLUMNS)
    for _, part in table.groupby("feature"):
        assert part["n"].sum() == len(X)
        assert np.isclose(part["share_of_all_missed"].sum(), 1.0, atol=1e-3)
        assert np.isclose(part["share_of_all_wasted"].sum(), 1.0, atol=1e-3)
    assert table["missed_rate"].dropna().between(0, 1).all()


def test_error_table_without_errors_has_no_division_by_zero(dataset):
    X, y, _ = dataset
    table = error_table(X, y, y.to_numpy() == 1)  # идеальный список: ни пропусков, ни лишних
    assert (table["missed_rate"].fillna(0) == 0).all()
    assert table["share_of_all_missed"].isna().all()


@pytest.mark.parametrize(
    "name,params,features",
    [
        ("logreg", {"tenure_spline": True}, ["contract_payment", "n_services"]),
        ("catboost", {"iterations": 60, "one_hot_max_size": 4}, ["contract_payment"]),
    ],
)
def test_shap_is_additive(dataset, name, params, features):
    """Сумма SHAP по признакам + базовое значение = лог-шансы модели."""
    X, y, _ = dataset
    X_train, X_eval, y_train = X.iloc[:200], X.iloc[200:], y.iloc[:200]
    model = make_model(name, features=features, **params).fit(X_train, y_train)
    shap = fold_shap(model, X_train, X_eval)
    assert "Contract" in shap.columns and "tenure" in shap.columns
    raw = logit(np.clip(model.predict_proba(X_eval)[:, 1], 1e-12, 1 - 1e-12))
    offset = raw - shap.sum(axis=1).to_numpy()
    # базовое значение одно и то же для всех строк
    assert np.ptp(offset) < 1e-4

import numpy as np
import pandas as pd

from churn.analysis import GROUP_COLUMNS, add_groups, error_table
from churn.data import load_dataset
from tests.conftest import DATA_PATH


def test_groups_cover_all_rows(raw_df):
    X, _, _ = load_dataset(DATA_PATH)
    groups = add_groups(X)
    assert not groups["tenure_group"].eq("nan").any()
    assert not groups["monthly_charges_group"].eq("nan").any()


def test_error_table_counts_are_consistent(raw_df):
    X, y, _ = load_dataset(DATA_PATH)
    X, y = X.iloc[:800].reset_index(drop=True), y.iloc[:800].reset_index(drop=True)
    proba = np.random.default_rng(0).random(len(y))
    table = error_table(X, y, proba, threshold=0.5)
    assert set(table["feature"]) == set(GROUP_COLUMNS)
    for _, part in table.groupby("feature"):
        assert part["n"].sum() == len(X)
        assert np.isclose(part["share_of_all_fp"].sum(), 1.0, atol=1e-3)
        assert np.isclose(part["share_of_all_fn"].sum(), 1.0, atol=1e-3)
    assert table["fp_rate"].between(0, 1).all() and table["fn_rate"].between(0, 1).all()

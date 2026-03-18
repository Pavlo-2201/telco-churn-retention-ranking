import pytest

from churn.data import (
    DATA_SHA256,
    EXPECTED_ROWS,
    sha256_of,
    target_distribution,
    validate_schema,
)
from tests.conftest import DATA_PATH


def test_checksum_matches(raw_df):
    assert sha256_of(DATA_PATH) == DATA_SHA256


def test_real_data_passes_schema(raw_df):
    validate_schema(raw_df)
    assert len(raw_df) == EXPECTED_ROWS


def test_schema_rejects_missing_column(raw_df):
    with pytest.raises(ValueError, match="столбцы"):
        validate_schema(raw_df.drop(columns=["tenure"]))


def test_schema_rejects_bad_target(raw_df):
    broken = raw_df.copy()
    broken.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Churn"):
        validate_schema(broken)


def test_schema_rejects_duplicate_ids(raw_df):
    broken = raw_df.copy()
    broken.loc[1, "customerID"] = broken.loc[0, "customerID"]
    with pytest.raises(ValueError, match="дубликаты"):
        validate_schema(broken)


def test_target_distribution_sums_to_total(raw_df):
    dist = target_distribution(raw_df)
    assert sum(v["count"] for v in dist.values()) == len(raw_df)


def test_clean_data_total_charges(raw_df):
    from churn.data import clean_data

    blank = raw_df["TotalCharges"].str.strip() == ""
    assert blank.sum() == 11
    assert (raw_df.loc[blank, "tenure"] == 0).all()
    clean = clean_data(raw_df)
    assert clean["TotalCharges"].dtype == "float64"
    assert clean["TotalCharges"].isna().sum() == 0
    assert (clean.loc[blank, "TotalCharges"] == 0).all()
    assert set(clean["Churn"].unique()) == {0, 1}


def test_clean_data_keeps_unexplained_missing_as_nan(raw_df):
    from churn.data import clean_data

    df = raw_df.head(3).copy()
    df.loc[0, "TotalCharges"] = " "  # tenure > 0, пропуск не объясняется
    assert df.loc[0, "tenure"] > 0 or df.loc[1, "tenure"] > 0
    df.loc[0, "tenure"] = 5
    clean = clean_data(df)
    assert clean["TotalCharges"].isna().sum() == 1


def test_features_exclude_id_and_target(raw_df):
    from churn.data import FEATURES, load_dataset

    assert "customerID" not in FEATURES and "Churn" not in FEATURES
    X, y, ids = load_dataset(DATA_PATH)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == len(ids)
    assert set(FEATURES) | {"customerID", "Churn"} == set(raw_df.columns)

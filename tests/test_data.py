import pytest

from churn.data import (
    DATA_SHA256,
    EXPECTED_ROWS,
    FEATURES,
    clean_data,
    load_dataset,
    sha256_of,
    target_distribution,
    validate_schema,
)
from tests.conftest import DATA_PATH

# --- проверки настоящего файла (пропускаются без данных) ---


def test_checksum_matches(raw_df):
    assert sha256_of(DATA_PATH) == DATA_SHA256


def test_real_data_passes_schema(raw_df):
    validate_schema(raw_df)
    assert len(raw_df) == EXPECTED_ROWS


def test_real_blank_total_charges_only_for_new_customers(raw_df):
    blank = raw_df["TotalCharges"].str.strip() == ""
    assert blank.sum() == 11
    assert (raw_df.loc[blank, "tenure"] == 0).all()


# --- поведение на синтетическом наборе той же схемы ---


def test_synthetic_data_passes_schema(synthetic_raw):
    validate_schema(synthetic_raw)


def test_schema_rejects_missing_column(synthetic_raw):
    with pytest.raises(ValueError, match="столбцы"):
        validate_schema(synthetic_raw.drop(columns=["tenure"]))


def test_schema_rejects_bad_target(synthetic_raw):
    broken = synthetic_raw.copy()
    broken.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Churn"):
        validate_schema(broken)


def test_schema_rejects_duplicate_ids(synthetic_raw):
    broken = synthetic_raw.copy()
    broken.loc[1, "customerID"] = broken.loc[0, "customerID"]
    with pytest.raises(ValueError, match="дубликаты"):
        validate_schema(broken)


def test_target_distribution_sums_to_total(synthetic_raw):
    dist = target_distribution(synthetic_raw)
    assert sum(v["count"] for v in dist.values()) == len(synthetic_raw)


def test_clean_data_total_charges(synthetic_raw):
    blank = synthetic_raw["TotalCharges"].str.strip() == ""
    assert blank.sum() > 0 and (synthetic_raw.loc[blank, "tenure"] == 0).all()
    clean = clean_data(synthetic_raw)
    assert clean["TotalCharges"].dtype == "float64"
    assert clean["TotalCharges"].isna().sum() == 0
    assert (clean.loc[blank, "TotalCharges"] == 0).all()
    assert set(clean["Churn"].unique()) == {0, 1}


def test_clean_data_keeps_unexplained_missing_as_nan(synthetic_raw):
    df = synthetic_raw.head(3).copy()
    df.loc[0, "TotalCharges"] = " "
    df.loc[0, "tenure"] = 5  # tenure > 0: пропуск не объясняется
    clean = clean_data(df)
    assert clean["TotalCharges"].isna().sum() == 1


def test_features_exclude_id_and_target(synthetic_raw, synthetic_csv):
    assert "customerID" not in FEATURES and "Churn" not in FEATURES
    X, y, ids = load_dataset(synthetic_csv)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == len(ids)
    assert set(FEATURES) | {"customerID", "Churn"} == set(synthetic_raw.columns)

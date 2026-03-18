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

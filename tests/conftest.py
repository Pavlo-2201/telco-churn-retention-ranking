import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from churn.data import DEFAULT_DATA_PATH, load_dataset  # noqa: E402
from churn.synthetic import make_synthetic_raw, write_synthetic  # noqa: E402

DATA_PATH = ROOT / DEFAULT_DATA_PATH
SYNTHETIC_ROWS = 300


@pytest.fixture(scope="session")
def raw_df():
    """Настоящий датасет: только для проверок, завязанных на его содержимое."""
    if not DATA_PATH.exists():
        pytest.skip("нет данных: выполните python -m churn.download_data")
    from churn.data import load_raw

    return load_raw(DATA_PATH)


@pytest.fixture(scope="session")
def synthetic_raw():
    return make_synthetic_raw(SYNTHETIC_ROWS, seed=0)


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory):
    return write_synthetic(tmp_path_factory.mktemp("data") / "telco_synthetic.csv", SYNTHETIC_ROWS, seed=0)


@pytest.fixture(scope="session")
def dataset(synthetic_csv):
    """(X, y, ids) синтетического набора после той же очистки, что и у настоящего."""
    return load_dataset(synthetic_csv)

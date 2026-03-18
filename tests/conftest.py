import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from churn.data import DEFAULT_DATA_PATH  # noqa: E402

DATA_PATH = ROOT / DEFAULT_DATA_PATH


@pytest.fixture(scope="session")
def raw_df():
    if not DATA_PATH.exists():
        pytest.skip("нет данных: выполните python -m churn.download_data")
    from churn.data import load_raw

    return load_raw(DATA_PATH)

"""Фиксированные стратифицированные разбиения train/validation/test."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_SPLITS_PATH = Path("results/splits.csv")
SPLIT_NAMES = ("train", "val", "test")


def make_splits(
    ids: pd.Series, y: pd.Series, seed: int = 42, test_size: float = 0.2, val_size: float = 0.2
) -> pd.DataFrame:
    """Возвращает таблицу customerID -> split. Доли считаются от всего набора."""
    rest_ids, test_ids, rest_y, _ = train_test_split(
        ids, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_share_of_rest = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        rest_ids, test_size=val_share_of_rest, stratify=rest_y, random_state=seed
    )
    parts = [
        pd.DataFrame({"customerID": part.to_numpy(), "split": name})
        for name, part in zip(SPLIT_NAMES, (train_ids, val_ids, test_ids))
    ]
    return pd.concat(parts, ignore_index=True)


def save_splits(splits: pd.DataFrame, path: Path = DEFAULT_SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.sort_values(["split", "customerID"]).to_csv(path, index=False)


def load_or_create_splits(
    ids: pd.Series,
    y: pd.Series,
    path: Path = DEFAULT_SPLITS_PATH,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        splits = pd.read_csv(path)
        if set(splits["customerID"]) != set(ids):
            raise ValueError(f"{path} не соответствует текущим данным; удалите его и пересоздайте")
        return splits
    splits = make_splits(ids, y, seed=seed, test_size=test_size, val_size=val_size)
    save_splits(splits, path)
    return splits


def split_masks(ids: pd.Series, splits: pd.DataFrame) -> dict[str, pd.Series]:
    """Булевы маски строк исходного набора для каждой части."""
    mapping = splits.set_index("customerID")["split"]
    assigned = ids.map(mapping)
    return {name: (assigned == name).to_numpy() for name in SPLIT_NAMES}

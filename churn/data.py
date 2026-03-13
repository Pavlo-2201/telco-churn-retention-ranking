"""Получение данных и проверка их схемы."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

SOURCE_REPO = "https://github.com/IBM/telco-customer-churn-on-icp4d"
SOURCE_COMMIT = "d5371f5d83a446ad5673cbcca3b814b926491f8a"
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    f"{SOURCE_COMMIT}/data/Telco-Customer-Churn.csv"
)
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
DEFAULT_DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
EXPECTED_ROWS = 7043

# Ожидаемые столбцы в порядке файла и их тип в сыром CSV.
EXPECTED_COLUMNS: dict[str, str] = {
    "customerID": "object",
    "gender": "object",
    "SeniorCitizen": "int64",
    "Partner": "object",
    "Dependents": "object",
    "tenure": "int64",
    "PhoneService": "object",
    "MultipleLines": "object",
    "InternetService": "object",
    "OnlineSecurity": "object",
    "OnlineBackup": "object",
    "DeviceProtection": "object",
    "TechSupport": "object",
    "StreamingTV": "object",
    "StreamingMovies": "object",
    "Contract": "object",
    "PaperlessBilling": "object",
    "PaymentMethod": "object",
    "MonthlyCharges": "float64",
    # В сыром файле TotalCharges строковый: у части клиентов там пробел.
    "TotalCharges": "object",
    "Churn": "object",
}
TARGET_VALUES = {"No", "Yes"}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_data(path: Path = DEFAULT_DATA_PATH, force: bool = False) -> Path:
    """Скачивает CSV, если его нет, и проверяет контрольную сумму."""
    path = Path(path)
    if force or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".part")
        urllib.request.urlretrieve(DATA_URL, tmp_path)
        tmp_path.replace(path)
    actual = sha256_of(path)
    if actual != DATA_SHA256:
        raise ValueError(
            f"Контрольная сумма {path} не совпадает: {actual} != {DATA_SHA256}. "
            "Удалите файл и скачайте заново."
        )
    return path


def load_raw(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Читает сырой CSV без преобразований: все строки остаются строками."""
    return pd.read_csv(path, dtype={"TotalCharges": "object"})


def _kind(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "int64"
    if pd.api.types.is_float_dtype(dtype):
        return "float64"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return "object"
    return str(dtype)


def validate_schema(df: pd.DataFrame) -> None:
    """Проверяет состав и типы столбцов, уникальность id и значения цели."""
    problems = []
    if list(df.columns) != list(EXPECTED_COLUMNS):
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(EXPECTED_COLUMNS)
        problems.append(f"столбцы не совпадают: нет {sorted(missing)}, лишние {sorted(extra)}")
    else:
        for column, expected in EXPECTED_COLUMNS.items():
            actual = _kind(df[column].dtype)
            if actual != expected:
                problems.append(f"{column}: тип {actual}, ожидался {expected}")
        if df[ID_COLUMN].duplicated().any():
            problems.append(f"{ID_COLUMN} содержит дубликаты")
        bad_target = set(df[TARGET_COLUMN].dropna().unique()) - TARGET_VALUES
        if bad_target or df[TARGET_COLUMN].isna().any():
            problems.append(f"{TARGET_COLUMN} содержит неожиданные значения: {bad_target}")
    if problems:
        raise ValueError("Схема данных нарушена:\n- " + "\n- ".join(problems))


def target_distribution(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    counts = df[TARGET_COLUMN].value_counts()
    shares = df[TARGET_COLUMN].value_counts(normalize=True)
    return {
        label: {"count": int(counts[label]), "share": round(float(shares[label]), 4)}
        for label in sorted(counts.index)
    }

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


# Признаки модели. customerID не является признаком: это уникальный
# идентификатор без смысла для поведения клиента, модель могла бы только
# запомнить его.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_NUMERIC_FEATURES = ["SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + BINARY_NUMERIC_FEATURES + CATEGORICAL_FEATURES


def parse_total_charges(values: pd.Series) -> pd.Series:
    """Переводит TotalCharges в число; пустые строки становятся NaN."""
    stripped = values.astype("string").str.strip()
    return pd.to_numeric(stripped.mask(stripped == ""), errors="coerce").astype("float64")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Построчная очистка без статистик по выборке, поэтому утечки не создаёт.

    - TotalCharges -> float. Пустое значение встречается только у клиентов
      с tenure == 0 (подключились в последнем периоде, ещё ничего не оплатили),
      поэтому для них ставится 0. Остальные пропуски, если появятся в новых
      данных, заполняет импьютер внутри Pipeline.
    - Churn -> 0/1, если столбец есть (на инференсе его нет).
    """
    out = df.copy()
    total = parse_total_charges(out["TotalCharges"])
    new_customer = out["tenure"].eq(0) & total.isna()
    out["TotalCharges"] = total.mask(new_customer, 0.0)
    if TARGET_COLUMN in out.columns:
        out[TARGET_COLUMN] = out[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype("int64")
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Превращает сырые записи клиентов (в формате исходного CSV) в признаки модели.

    Используется на инференсе: проверяет наличие нужных столбцов, применяет те же
    построчные правила, что и clean_data, и отбрасывает лишние столбцы
    (customerID, Churn и любые другие).
    """
    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Во входных данных нет столбцов: {missing}")
    out = df[FEATURES].copy()
    if not pd.api.types.is_numeric_dtype(out["TotalCharges"]):
        out["TotalCharges"] = parse_total_charges(out["TotalCharges"])
    out["TotalCharges"] = out["TotalCharges"].mask(out["tenure"].eq(0) & out["TotalCharges"].isna(), 0.0)
    return out


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Возвращает признаки X (без customerID), цель y и идентификаторы."""
    raw = load_raw(path)
    validate_schema(raw)
    df = clean_data(raw)
    return df[FEATURES], df[TARGET_COLUMN], df[ID_COLUMN]

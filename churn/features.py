"""Предобработка признаков внутри обучаемого конвейера."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from churn.data import BINARY_NUMERIC_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_linear_preprocessor() -> ColumnTransformer:
    """Для линейных моделей: импутация, масштабирование чисел и one-hot для категорий."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("bin", binary, BINARY_NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )

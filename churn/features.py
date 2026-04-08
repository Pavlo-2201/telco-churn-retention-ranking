"""Предобработка признаков внутри обучаемого конвейера."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

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


def build_tree_preprocessor() -> ColumnTransformer:
    """Для случайного леса: масштабирование не нужно, а one-hot раздувает
    пространство и дробит категорию на много слабых сплитов. Категории
    кодируются порядковыми номерами, неизвестные получают -1."""
    numeric = SimpleImputer(strategy="median")
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES + BINARY_NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


MISSING_CATEGORY = "__missing__"


def to_catboost_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Готовит DataFrame для CatBoost без обучаемых статистик: категории
    становятся строками (пропуск -> отдельная категория), числа остаются
    числами, их пропуски CatBoost обрабатывает сам."""
    out = X.copy()
    for column in CATEGORICAL_FEATURES:
        out[column] = out[column].astype("object").where(out[column].notna(), MISSING_CATEGORY).astype(str)
    return out


def build_catboost_preprocessor() -> FunctionTransformer:
    return FunctionTransformer(to_catboost_frame, validate=False)

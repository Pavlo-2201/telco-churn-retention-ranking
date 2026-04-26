"""Построчные признаки и предобработка внутри обучаемого конвейера."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
)

from churn.data import BINARY_NUMERIC_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
SERVICE_COLUMNS = ["PhoneService", "MultipleLines", *INTERNET_ADDONS]
NO_INTERNET = "No internet service"
TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49+"]

# Кандидаты для абляции. Все признаки считаются по одной строке и не используют цель,
# поэтому утечки не создают. Для каждого: какие столбцы добавляются (числовые,
# бинарные, категориальные) и какие исходные удаляются.
ENGINEERED_FEATURES: dict[str, dict[str, list[str]]] = {
    # число подключённых услуг (значения "Yes" в столбцах услуг)
    "n_services": {"numeric": ["n_services"]},
    # одна колонка «нет интернета»; в шести столбцах доп. услуг значение
    # "No internet service" заменяется на "No", дубли исчезают
    "no_internet": {"binary": ["no_internet"]},
    # средний платёж за весь срок и относительное отличие текущего от него: прокси
    # роста цены. Относительное, а не MonthlyCharges − avg: разность дала бы точную
    # линейную зависимость MonthlyCharges = avg + delta (бесконечный VIF для LogReg)
    "avg_charge": {"numeric": ["avg_monthly_charge", "charge_growth"]},
    # взаимодействие контракта и способа оплаты
    "contract_payment": {"categorical": ["contract_payment"]},
    # корзины стажа
    "tenure_bins": {"categorical": ["tenure_group"]},
    # стаж отдельно для годовых и двухлетних контрактов (взаимодействие для LogReg)
    "contract_tenure": {"numeric": ["tenure_one_year", "tenure_two_year"]},
    # TotalCharges ≈ tenure × MonthlyCharges: проверка мультиколлинеарности для LogReg
    "drop_total_charges": {"drop": ["TotalCharges"]},
    # MonthlyCharges почти линейно складывается из набора услуг (R² ≈ 0.999 на train)
    "drop_monthly_charges": {"drop": ["MonthlyCharges"]},
}


def check_features(features) -> tuple[str, ...]:
    unknown = [f for f in features if f not in ENGINEERED_FEATURES]
    if unknown:
        raise ValueError(f"Неизвестные признаки: {unknown}; доступны {sorted(ENGINEERED_FEATURES)}")
    return tuple(features)


def column_groups(features=()) -> tuple[list[str], list[str], list[str]]:
    """Числовые, бинарные и категориальные столбцы после FeatureBuilder."""
    numeric, binary, categorical = list(NUMERIC_FEATURES), list(BINARY_NUMERIC_FEATURES), list(CATEGORICAL_FEATURES)
    for name in check_features(features):
        spec = ENGINEERED_FEATURES[name]
        numeric += spec.get("numeric", [])
        binary += spec.get("binary", [])
        categorical += spec.get("categorical", [])
        for column in spec.get("drop", []):
            for group in (numeric, binary, categorical):
                if column in group:
                    group.remove(column)
    return numeric, binary, categorical


def add_engineered(X: pd.DataFrame, features=()) -> pd.DataFrame:
    features = check_features(features)
    out = X.copy()
    if "n_services" in features:
        out["n_services"] = out[SERVICE_COLUMNS].eq("Yes").sum(axis=1).astype("float64")
    if "no_internet" in features:
        out["no_internet"] = out["InternetService"].eq("No").astype("int64")
        for column in INTERNET_ADDONS:
            out[column] = out[column].replace(NO_INTERNET, "No")
    if "avg_charge" in features:
        out["avg_monthly_charge"] = out["TotalCharges"] / out["tenure"].clip(lower=1)
        average = out["avg_monthly_charge"].where(out["avg_monthly_charge"] > 0)
        out["charge_growth"] = out["MonthlyCharges"] / average - 1
    if "contract_payment" in features:
        known = out["Contract"].notna() & out["PaymentMethod"].notna()
        combo = out["Contract"].astype(str) + " | " + out["PaymentMethod"].astype(str)
        out["contract_payment"] = combo.astype(object).where(known, np.nan)
    if "tenure_bins" in features:
        out["tenure_group"] = pd.cut(out["tenure"], TENURE_BINS, labels=TENURE_LABELS).astype(object)
    if "contract_tenure" in features:
        out["tenure_one_year"] = out["tenure"] * out["Contract"].eq("One year")
        out["tenure_two_year"] = out["tenure"] * out["Contract"].eq("Two year")
    dropped = [column for name in features for column in ENGINEERED_FEATURES[name].get("drop", [])]
    return out.drop(columns=dropped)


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Добавляет выбранные построчные признаки. Ничего не обучает: fit только
    запоминает имена столбцов."""

    def __init__(self, features=()):
        self.features = features

    def fit(self, X, y=None):
        check_features(self.features)
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return add_engineered(X, self.features)

    def get_feature_names_out(self, input_features=None):
        numeric, binary, categorical = column_groups(self.features)
        return np.asarray(numeric + binary + categorical, dtype=object)


def _categorical_pipeline(encoder) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", encoder)])


def build_linear_preprocessor(features=(), tenure_spline: bool = False, n_knots: int = 5) -> ColumnTransformer:
    """Для линейных моделей: импутация, масштабирование чисел, one-hot для категорий.
    tenure_spline заменяет линейный tenure на кубический B-сплайн."""
    numeric, binary, categorical = column_groups(features)
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    transformers = []
    if tenure_spline:
        numeric = [c for c in numeric if c != "tenure"]
        spline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("spline", SplineTransformer(n_knots=n_knots, degree=3, extrapolation="constant")),
            ]
        )
        transformers.append(("tenure_spline", spline, ["tenure"]))
    transformers += [
        ("num", numeric_pipeline, numeric),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
        ("cat", _categorical_pipeline(OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical),
    ]
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def build_tree_preprocessor(features=()) -> ColumnTransformer:
    """Для случайного леса: масштабирование не нужно, а one-hot раздувает
    пространство и дробит категорию на много слабых сплитов. Категории
    кодируются порядковыми номерами, неизвестные получают -1."""
    numeric, binary, categorical = column_groups(features)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric + binary),
            ("cat", _categorical_pipeline(encoder), categorical),
        ],
        verbose_feature_names_out=False,
    )


MISSING_CATEGORY = "__missing__"


def to_catboost_frame(X: pd.DataFrame, categorical=tuple(CATEGORICAL_FEATURES)) -> pd.DataFrame:
    """Готовит DataFrame для CatBoost без обучаемых статистик: категории
    становятся строками (пропуск -> отдельная категория), числа остаются
    числами, их пропуски CatBoost обрабатывает сам."""
    out = X.copy()
    for column in categorical:
        out[column] = out[column].astype("object").where(out[column].notna(), MISSING_CATEGORY).astype(str)
    return out


def build_catboost_preprocessor(features=()) -> FunctionTransformer:
    _, _, categorical = column_groups(features)
    return FunctionTransformer(to_catboost_frame, kw_args={"categorical": tuple(categorical)}, validate=False)


def linear_feature_sources(prep: ColumnTransformer) -> list[str]:
    """Для каждого выходного столбца обученного линейного препроцессора: из какого
    входного признака он получен (one-hot и сплайны дают несколько столбцов)."""
    sources: list[str] = []
    for name, transformer, columns in prep.transformers_:
        if name == "remainder" or transformer == "drop" or len(columns) == 0:
            continue
        if name == "cat":
            encoder = transformer.named_steps["encode"]
            for column, categories in zip(columns, encoder.categories_):
                sources += [column] * len(categories)
        elif name == "tenure_spline":
            n_out = transformer.named_steps["spline"].n_features_out_
            sources += [columns[0]] * n_out
        else:
            sources += list(columns)
    return sources

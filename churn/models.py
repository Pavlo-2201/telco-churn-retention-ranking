"""Фабрика моделей: каждая модель вместе со своей предобработкой."""

from __future__ import annotations

from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.data import CATEGORICAL_FEATURES
from churn.features import (
    build_catboost_preprocessor,
    build_linear_preprocessor,
    build_tree_preprocessor,
)

MODEL_NAMES = ("dummy", "logreg", "random_forest", "catboost")


def make_model(name: str, seed: int = 42, **params) -> Pipeline:
    if name == "dummy":
        # Предсказывает долю положительного класса из train: нижняя граница качества.
        return Pipeline([("model", DummyClassifier(strategy="prior", **params))])
    if name == "logreg":
        defaults = {"C": 1.0, "max_iter": 2000}
        defaults.update(params)
        return Pipeline(
            [
                ("prep", build_linear_preprocessor()),
                ("model", LogisticRegression(random_state=seed, **defaults)),
            ]
        )
    if name == "random_forest":
        defaults = {"n_estimators": 500, "min_samples_leaf": 5, "n_jobs": -1}
        defaults.update(params)
        return Pipeline(
            [
                ("prep", build_tree_preprocessor()),
                ("model", RandomForestClassifier(random_state=seed, **defaults)),
            ]
        )
    if name == "catboost":
        # Категории передаются как есть: CatBoost сам строит ordered target
        # statistics, вычисляя их без использования цели текущего объекта.
        defaults = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "thread_count": 4,
            "verbose": False,
            "allow_writing_files": False,
        }
        defaults.update(params)
        return Pipeline(
            [
                ("prep", build_catboost_preprocessor()),
                (
                    "model",
                    CatBoostClassifier(
                        random_seed=seed, cat_features=CATEGORICAL_FEATURES, **defaults
                    ),
                ),
            ]
        )
    raise ValueError(f"Неизвестная модель: {name}")

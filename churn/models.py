"""Фабрика моделей: каждая модель вместе со своей предобработкой."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.features import build_linear_preprocessor


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
    raise ValueError(f"Неизвестная модель: {name}")

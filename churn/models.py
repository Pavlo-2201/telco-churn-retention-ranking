"""Фабрика моделей: каждая модель вместе со своей предобработкой."""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
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


class CatBoostCatClassifier(ClassifierMixin, BaseEstimator):
    """Тонкая sklearn-обёртка над CatBoostClassifier.

    CatBoostClassifier копирует список cat_features в конструкторе, из-за чего
    sklearn.clone (а значит GridSearchCV и cross_validate) отказывается его
    клонировать. Обёртка хранит параметры как есть и создаёт CatBoost в fit.
    """

    def __init__(
        self,
        cat_features=tuple(CATEGORICAL_FEATURES),
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        thread_count=4,
    ):
        self.cat_features = cat_features
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(
            cat_features=list(self.cat_features),
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            thread_count=self.thread_count,
            verbose=False,
            allow_writing_files=False,
        )
        self.model_.fit(X, y)
        self.classes_ = np.asarray(self.model_.classes_)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


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
        return Pipeline(
            [
                ("prep", build_catboost_preprocessor()),
                ("model", CatBoostCatClassifier(random_seed=seed, **params)),
            ]
        )
    raise ValueError(f"Неизвестная модель: {name}")

"""Фабрика моделей: каждая модель вместе со своими признаками и предобработкой."""

from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from churn.data import CATEGORICAL_FEATURES
from churn.features import (
    FeatureBuilder,
    build_catboost_preprocessor,
    build_linear_preprocessor,
    build_tree_preprocessor,
    column_groups,
)

MODEL_NAMES = ("dummy", "logreg", "random_forest", "catboost")
CALIBRATION_METHODS = ("sigmoid", "isotonic")


class CatBoostCatClassifier(ClassifierMixin, BaseEstimator):
    """Тонкая sklearn-обёртка над CatBoostClassifier.

    CatBoostClassifier копирует список cat_features в конструкторе, из-за чего
    sklearn.clone (а значит и кросс-валидация) отказывается его клонировать.
    Обёртка хранит параметры как есть и создаёт CatBoost в fit.

    Если задан early_stopping_rounds, число деревьев выбирается по отложенному
    стратифицированному куску обучающих данных (validation_fraction), после чего
    модель переобучается на всех данных с найденным числом деревьев. Так внутри
    каждого фолда CV число итераций подбирается без доступа к его тестовой части.
    """

    def __init__(
        self,
        cat_features=tuple(CATEGORICAL_FEATURES),
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        one_hot_max_size=None,
        early_stopping_rounds=None,
        validation_fraction=0.15,
        random_seed=42,
        thread_count=4,
    ):
        self.cat_features = cat_features
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.one_hot_max_size = one_hot_max_size
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_seed = random_seed
        self.thread_count = thread_count

    def _catboost(self, iterations: int) -> CatBoostClassifier:
        return CatBoostClassifier(
            cat_features=list(self.cat_features),
            iterations=iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            one_hot_max_size=self.one_hot_max_size,
            random_seed=self.random_seed,
            thread_count=self.thread_count,
            verbose=False,
            allow_writing_files=False,
        )

    def fit(self, X, y):
        n_iterations = self.iterations
        if self.early_stopping_rounds:
            X_fit, X_stop, y_fit, y_stop = train_test_split(
                X, y, test_size=self.validation_fraction, stratify=y, random_state=self.random_seed
            )
            probe = self._catboost(self.iterations)
            probe.fit(
                X_fit, y_fit, eval_set=(X_stop, y_stop),
                early_stopping_rounds=self.early_stopping_rounds, use_best_model=True,
            )
            n_iterations = probe.get_best_iteration() + 1
        self.model_ = self._catboost(n_iterations).fit(X, y)
        self.n_iterations_ = n_iterations
        self.classes_ = np.asarray(self.model_.classes_)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def make_model(name: str, seed: int = 42, features=(), **params) -> Pipeline:
    """Pipeline: построчные признаки -> предобработка -> модель."""
    fe = ("fe", FeatureBuilder(tuple(features)))
    if name == "dummy":
        # Предсказывает долю положительного класса из train: нижняя граница качества.
        return Pipeline([("model", DummyClassifier(strategy="prior", **params))])
    if name == "logreg":
        params = dict(params)
        prep = build_linear_preprocessor(
            features, tenure_spline=params.pop("tenure_spline", False), n_knots=params.pop("n_knots", 5)
        )
        defaults = {"C": 1.0, "max_iter": 5000}
        defaults.update(params)
        return Pipeline([fe, ("prep", prep), ("model", LogisticRegression(random_state=seed, **defaults))])
    if name == "random_forest":
        defaults = {"n_estimators": 500, "min_samples_leaf": 5, "n_jobs": -1}
        defaults.update(params)
        model = RandomForestClassifier(random_state=seed, **defaults)
        return Pipeline([fe, ("prep", build_tree_preprocessor(features)), ("model", model)])
    if name == "catboost":
        # Категории передаются как есть: CatBoost сам строит ordered target
        # statistics, вычисляя их без использования цели текущего объекта.
        _, _, categorical = column_groups(features)
        model = CatBoostCatClassifier(cat_features=tuple(categorical), random_seed=seed, **params)
        return Pipeline([fe, ("prep", build_catboost_preprocessor(features)), ("model", model)])
    raise ValueError(f"Неизвестная модель: {name}")


def make_estimator(spec: dict, seed: int = 42):
    """Модель по спецификации {name, params, features, calibration}.

    При calibration = sigmoid|isotonic модель оборачивается в CalibratedClassifierCV
    с ensemble=False: калибратор обучается на out-of-fold предсказаниях внутренних
    фолдов, базовая модель затем переобучается на всех переданных данных. Внутри
    внешней CV калибратор не видит тестовую часть фолда.
    """
    pipeline = make_model(spec["name"], seed=seed, features=spec.get("features", ()), **spec.get("params", {}))
    method = spec.get("calibration")
    if not method:
        return pipeline
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Неизвестный метод калибровки: {method}")
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(pipeline, method=method, cv=inner_cv, ensemble=False)


def base_pipeline(estimator) -> Pipeline:
    """Обученный Pipeline без калибратора (для интерпретации)."""
    if isinstance(estimator, CalibratedClassifierCV):
        return estimator.calibrated_classifiers_[0].estimator
    return estimator

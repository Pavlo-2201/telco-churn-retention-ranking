"""Инференс-артефакт: обученная модель вместе с порогом и метаданными."""

from __future__ import annotations

import platform
import warnings
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn.data import prepare_features
from churn.metrics import ranking_order

TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "catboost", "joblib")


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def version_mismatches(saved: dict[str, str], current: dict[str, str] | None = None) -> dict[str, tuple[str, str]]:
    current = current or library_versions()
    return {k: (v, current.get(k, "not installed")) for k, v in saved.items() if current.get(k) != v}


def select_for_retention(proba, threshold: float, budget: int | None = None) -> np.ndarray:
    """1 — клиент в списке: оценка не ниже порога точки работы, а при заданном бюджете
    он ещё и среди budget клиентов с наибольшей оценкой (при равенстве — по порядку строк)."""
    proba = np.asarray(proba, dtype=float)
    selected = proba >= threshold
    if budget is not None:
        if budget < 0:
            raise ValueError(f"Бюджет должен быть неотрицательным, получено {budget}")
        top = np.zeros(len(proba), dtype=bool)
        top[ranking_order(proba)[:budget]] = True
        selected &= top
    return selected.astype(int)


@dataclass
class ChurnModel:
    """Всё, что нужно для прогноза на сырых записях клиентов.

    pipeline содержит всю обучаемую предобработку (и калибратор, если он применён),
    prepare_features выполняет построчные правила очистки. threshold — оценка
    клиента на границе выбранного по прибыли top-K (на OOF-предсказаниях train).
    """

    pipeline: object
    threshold: float
    model_name: str
    params: dict
    features: list[str]
    metadata: dict = field(default_factory=dict)

    def predict_proba(self, records: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(prepare_features(records))[:, 1]

    def predict(self, records: pd.DataFrame, budget: int | None = None) -> np.ndarray:
        """1 — клиент попадает в список для удержания (не больше budget клиентов, если он задан)."""
        return select_for_retention(self.predict_proba(records), self.threshold, budget)

    @property
    def budget_fraction(self) -> float | None:
        """Предел бюджета из конфигурации обучения, доля базы."""
        return self.metadata.get("config", {}).get("business", {}).get("budget_fraction")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> ChurnModel:
        model = joblib.load(path)
        if not isinstance(model, ChurnModel):
            raise TypeError(f"{path} не содержит ChurnModel")
        mismatches = version_mismatches(model.metadata.get("versions", {}))
        if mismatches:
            details = ", ".join(f"{k}: обучена на {saved}, сейчас {now}" for k, (saved, now) in mismatches.items())
            warnings.warn(f"Версии библиотек отличаются от версий при обучении ({details}). "
                          "Предсказания могут отличаться; переобучите модель: python train.py", stacklevel=2)
        return model

"""Инференс-артефакт: обученный Pipeline вместе с порогом и метаданными."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn.data import prepare_features


@dataclass
class ChurnModel:
    """Всё, что нужно для прогноза на сырых записях клиентов.

    pipeline содержит всю обучаемую предобработку, prepare_features выполняет
    построчные правила очистки, threshold выбран по OOF-предсказаниям.
    """

    pipeline: object
    threshold: float
    model_name: str
    params: dict
    features: list[str]
    metadata: dict = field(default_factory=dict)

    def predict_proba(self, records: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(prepare_features(records))[:, 1]

    def predict(self, records: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(records) >= self.threshold).astype(int)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ChurnModel":
        model = joblib.load(path)
        if not isinstance(model, ChurnModel):
            raise TypeError(f"{path} не содержит ChurnModel")
        return model

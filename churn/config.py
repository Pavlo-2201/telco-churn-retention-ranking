"""Конфигурация обучения (JSON: в отличие от TOML, поддерживает null для None)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from churn.models import MODEL_NAMES

REQUIRED_SECTIONS = ("seed", "data", "cv", "threshold", "models", "output")
SELECTION_METRICS = ("pr_auc", "roc_auc")


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text())
    missing = [key for key in REQUIRED_SECTIONS if key not in config]
    if missing:
        raise ValueError(f"В конфигурации {path} нет разделов: {missing}")
    unknown = set(config["models"]) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Неизвестные модели в конфигурации: {sorted(unknown)}")
    if config["cv"]["selection_metric"] not in SELECTION_METRICS:
        raise ValueError(f"selection_metric должен быть одним из {SELECTION_METRICS}")
    if config["threshold"]["criterion"] != "f1":
        raise ValueError("Поддерживается только criterion = 'f1'")
    return config


def threshold_grid(config: dict) -> np.ndarray:
    t = config["threshold"]
    return np.round(np.arange(t["grid_start"], t["grid_stop"] + 1e-9, t["grid_step"]), 4)

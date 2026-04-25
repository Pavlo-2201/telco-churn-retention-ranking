"""Конфигурация (JSON: в отличие от TOML, поддерживает null для None).

Единственный источник настроек для train.py, compare_models, ablation и analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

from churn.economics import PARAMETERS
from churn.features import check_features
from churn.models import CALIBRATION_METHODS, MODEL_NAMES

DEFAULT_CONFIG_PATH = Path("configs/train.json")
REQUIRED_SECTIONS = (
    "seed", "data", "cv", "ranking", "selection", "business", "calibration", "bootstrap", "features", "models",
    "analysis", "output",
)
SELECTION_METRICS = ("lift", "precision", "recall", "pr_auc", "roc_auc", "log_loss", "brier")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_SECTIONS if key not in config]
    if missing:
        raise ValueError(f"В конфигурации {path} нет разделов: {missing}")
    unknown = set(config["models"]) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Неизвестные модели в конфигурации: {sorted(unknown)}")
    sel = config["selection"]
    if sel["baseline"] not in config["models"]:
        raise ValueError(f"Базовая модель {sel['baseline']} должна быть в разделе models")
    for key in ("metric", "tuning_metric"):
        if sel[key].split("@")[0] not in SELECTION_METRICS:
            raise ValueError(f"selection.{key} должен начинаться с одного из {SELECTION_METRICS}")
    fractions = [f"{round(f * 100):g}" for f in config["ranking"]["top_fractions"]]
    if "@" in sel["metric"] and sel["metric"].split("@")[1] not in fractions:
        raise ValueError(f"{sel['metric']}: доля должна быть в ranking.top_fractions")
    missing_business = [p for p in (*PARAMETERS, "budget_fraction") if p not in config["business"]]
    if missing_business:
        raise ValueError(f"В разделе business нет допущений: {missing_business}")
    calibration = config["calibration"]
    if calibration["method"] not in CALIBRATION_METHODS or calibration["apply"] not in ("auto", "always", "never"):
        raise ValueError("calibration: method in {sigmoid, isotonic}, apply in {auto, always, never}")
    check_features(config["features"]["engineered"])
    check_features(config["features"]["candidates"])
    return config

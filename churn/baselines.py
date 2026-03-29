"""Базовые модели на фиксированном разбиении."""

import argparse
import json
from pathlib import Path

from churn.data import DEFAULT_DATA_PATH, download_data, load_dataset
from churn.metrics import classification_metrics
from churn.models import make_model
from churn.splits import DEFAULT_SPLITS_PATH, load_or_create_splits, split_masks

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--output", type=Path, default=Path("results/baselines.json"))
    args = parser.parse_args()

    X, y, ids = load_dataset(download_data(args.data))
    splits = load_or_create_splits(ids, y, args.splits, seed=SEED)
    masks = split_masks(ids, splits)
    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_val, y_val = X[masks["val"]], y[masks["val"]]

    results = {"seed": SEED, "evaluated_on": "val", "split_sizes": {}, "models": {}}
    for name, mask in masks.items():
        results["split_sizes"][name] = {"n": int(mask.sum()), "churn_rate": round(float(y[mask].mean()), 4)}

    for name in ["dummy", "logreg"]:
        model = make_model(name, seed=SEED)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        metrics = classification_metrics(y_val, proba, threshold=0.5)
        results["models"][name] = {"params": model.named_steps["model"].get_params(), "val": metrics}
        print(
            f"{name:7s} ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
            f"F1@0.5={metrics['f1']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"Результаты: {args.output}")


if __name__ == "__main__":
    main()

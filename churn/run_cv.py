"""Кросс-валидация с подбором гиперпараметров."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from churn.data import DEFAULT_DATA_PATH, download_data, load_dataset
from churn.models import MODEL_NAMES
from churn.splits import DEFAULT_SPLITS_PATH, load_or_create_splits, split_masks
from churn.tuning import PARAM_GRIDS, SELECTION_METRIC, make_cv, search_model

SEED = 42
N_FOLDS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    X, y, ids = load_dataset(download_data(args.data))
    masks = split_masks(ids, load_or_create_splits(ids, y, args.splits, seed=SEED))
    dev = masks["train"] | masks["val"]
    X_dev, y_dev = X[dev], y[dev]
    cv = make_cv(N_FOLDS, SEED)

    summary = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "data": "train+val (development set), test excluded",
        "n_dev": int(dev.sum()),
        "selection_metric": SELECTION_METRIC,
        "param_grids": {name: PARAM_GRIDS[name] for name in args.models},
        "models": {},
    }
    candidate_rows = []
    for name in args.models:
        start = time.perf_counter()
        result = search_model(name, X_dev, y_dev, cv, seed=SEED)
        elapsed = time.perf_counter() - start
        summary["models"][name] = {
            "best_params": result["best_params"],
            "cv": result["cv"],
            "n_candidates": len(result["candidates"]),
            "seconds": round(elapsed, 1),
        }
        for cand in result["candidates"]:
            candidate_rows.append(
                {
                    "model": name,
                    "params": json.dumps(cand["params"], sort_keys=True),
                    "mean_roc_auc": cand["mean_roc_auc"],
                    "std_roc_auc": cand["std_roc_auc"],
                    "mean_pr_auc": cand["mean_pr_auc"],
                    "std_pr_auc": cand["std_pr_auc"],
                }
            )
        cv_res = result["cv"]
        print(
            f"{name:13s} PR-AUC={cv_res['mean_pr_auc']:.4f}±{cv_res['std_pr_auc']:.4f} "
            f"ROC-AUC={cv_res['mean_roc_auc']:.4f}±{cv_res['std_roc_auc']:.4f} "
            f"({len(result['candidates'])} комбинаций, {elapsed:.0f} c) {result['best_params']}"
        )

    best_model = max(summary["models"], key=lambda m: summary["models"][m]["cv"][f"mean_{SELECTION_METRIC}"])
    summary["best_model"] = best_model
    print(f"Лучшая модель по CV {SELECTION_METRIC}: {best_model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cv_search.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame(candidate_rows).round(5).to_csv(args.output_dir / "cv_candidates.csv", index=False)


if __name__ == "__main__":
    main()

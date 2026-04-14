"""Порог по OOF-предсказаниям и однократная оценка на test."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_predict

from churn.data import DEFAULT_DATA_PATH, download_data, load_dataset
from churn.metrics import classification_metrics
from churn.models import make_model
from churn.splits import DEFAULT_SPLITS_PATH, load_or_create_splits, split_masks
from churn.threshold import select_threshold, threshold_curve
from churn.tuning import make_cv

SEED = 42
OOF_SEED = SEED + 1
N_FOLDS = 5


def plot_curves(y_test, proba_test, curve, threshold, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fpr, tpr, _ = roc_curve(y_test, proba_test)
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
    axes[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    prec, rec, _ = precision_recall_curve(y_test, proba_test)
    axes[1].plot(rec, prec)
    axes[1].axhline(y_test.mean(), color="grey", lw=1, ls="--")
    axes[1].set(title="Precision-Recall (test)", xlabel="recall", ylabel="precision")
    axes[2].plot([r["threshold"] for r in curve], [r["f1"] for r in curve])
    axes[2].axvline(threshold, color="black", lw=1, ls="--")
    axes[2].set(title="F1 от порога (OOF, development)", xlabel="порог", ylabel="F1")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--cv-results", type=Path, default=Path("results/cv_search.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    cv_summary = json.loads(args.cv_results.read_text())
    name = cv_summary["best_model"]
    params = cv_summary["models"][name]["best_params"]

    X, y, ids = load_dataset(download_data(args.data))
    masks = split_masks(ids, load_or_create_splits(ids, y, args.splits, seed=SEED))
    dev = masks["train"] | masks["val"]
    test = masks["test"]
    X_dev, y_dev, X_test, y_test = X[dev], y[dev], X[test], y[test]

    # 1. Out-of-fold вероятности и порог: только development-набор.
    oof = cross_val_predict(
        make_model(name, seed=SEED, **params), X_dev, y_dev,
        cv=make_cv(N_FOLDS, OOF_SEED), method="predict_proba",
    )[:, 1]
    threshold = select_threshold(y_dev, oof)
    curve = threshold_curve(y_dev, oof)
    oof_metrics = classification_metrics(y_dev, oof, threshold)
    oof_default = classification_metrics(y_dev, oof, 0.5)

    # 2. Финальная модель на всём development-наборе, однократная оценка на test.
    model = make_model(name, seed=SEED, **params).fit(X_dev, y_dev)
    proba_test = model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, proba_test, threshold)

    report = {
        "model": name,
        "params": params,
        "seed": SEED,
        "threshold_selection": {
            "data": "out-of-fold predictions on train+val",
            "n_folds": N_FOLDS,
            "cv_seed": OOF_SEED,
            "criterion": "max F1",
            "threshold": threshold,
        },
        "oof_dev": {"at_selected_threshold": oof_metrics, "at_0.5": oof_default},
        "test": test_metrics,
        "n_dev": int(dev.sum()),
        "n_test": int(test.sum()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "final_evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame({"customerID": ids[dev].to_numpy(), "y_true": y_dev.to_numpy(), "proba": oof.round(6)}).to_csv(
        args.output_dir / "oof_predictions.csv", index=False
    )
    pd.DataFrame(
        {"customerID": ids[test].to_numpy(), "y_true": y_test.to_numpy(), "proba": proba_test.round(6)}
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)
    plot_curves(y_test, proba_test, curve, threshold, args.figures_dir / "final_curves.png")

    m = test_metrics
    print(f"Модель: {name} {params}")
    print(f"Порог по OOF F1: {threshold:.2f} (OOF F1 {oof_metrics['f1']:.4f}, при 0.5: {oof_default['f1']:.4f})")
    print(
        f"Test: ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f} "
        f"precision={m['precision']:.4f} recall={m['recall']:.4f} CM={m['confusion_matrix']}"
    )


if __name__ == "__main__":
    main()

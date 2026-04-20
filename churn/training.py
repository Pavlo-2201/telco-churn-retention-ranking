"""Полный цикл обучения, который запускает train.py."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import catboost
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_predict

from churn.config import threshold_grid
from churn.data import DATA_SHA256, DATA_URL, FEATURES, download_data, load_dataset
from churn.inference import ChurnModel
from churn.metrics import classification_metrics
from churn.models import make_model
from churn.splits import load_or_create_splits, split_masks
from churn.threshold import select_threshold, threshold_curve
from churn.tuning import make_cv, search_model


def load_split_data(config: dict):
    data = config["data"]
    X, y, ids = load_dataset(download_data(Path(data["path"])))
    splits = load_or_create_splits(
        ids, y, Path(data["splits_path"]), seed=config["seed"],
        test_size=data["test_size"], val_size=data["val_size"],
    )
    masks = split_masks(ids, splits)
    return X, y, ids, masks


def run_model_selection(config: dict, X_dev, y_dev, log=print) -> dict:
    seed, metric = config["seed"], config["cv"]["selection_metric"]
    cv = make_cv(config["cv"]["n_folds"], seed)
    models, candidates = {}, []
    for name, grid in config["models"].items():
        start = time.perf_counter()
        result = search_model(name, X_dev, y_dev, cv, grid=grid, seed=seed, selection_metric=metric)
        elapsed = time.perf_counter() - start
        models[name] = {
            "best_params": result["best_params"],
            "cv": result["cv"],
            "n_candidates": len(result["candidates"]),
            "seconds": round(elapsed, 1),
        }
        for cand in result["candidates"]:
            candidates.append({
                "model": name,
                "params": json.dumps(cand["params"], sort_keys=True),
                **{k: cand[k] for k in ("mean_roc_auc", "std_roc_auc", "mean_pr_auc", "std_pr_auc")},
            })
        cv_res = result["cv"]
        log(
            f"{name:13s} PR-AUC={cv_res['mean_pr_auc']:.4f}±{cv_res['std_pr_auc']:.4f} "
            f"ROC-AUC={cv_res['mean_roc_auc']:.4f}±{cv_res['std_roc_auc']:.4f} "
            f"({len(result['candidates'])} комбинаций, {elapsed:.0f} c) {result['best_params']}"
        )
    best = max(models, key=lambda m: models[m]["cv"][f"mean_{metric}"])
    return {
        "seed": seed,
        "n_folds": config["cv"]["n_folds"],
        "data": "train+val (development set), test excluded",
        "n_dev": int(len(y_dev)),
        "selection_metric": metric,
        "param_grids": config["models"],
        "models": models,
        "best_model": best,
        "candidates": candidates,
    }


def fit_and_evaluate(config: dict, name: str, params: dict, X_dev, y_dev, X_test, y_test) -> dict:
    seed = config["seed"]
    grid = threshold_grid(config)
    oof = cross_val_predict(
        make_model(name, seed=seed, **params), X_dev, y_dev,
        cv=make_cv(config["cv"]["n_folds"], config["cv"]["oof_seed"]), method="predict_proba",
    )[:, 1]
    threshold = select_threshold(y_dev, oof, grid)

    pipeline = make_model(name, seed=seed, **params).fit(X_dev, y_dev)
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    return {
        "pipeline": pipeline,
        "threshold": threshold,
        "oof": oof,
        "curve": threshold_curve(y_dev, oof, grid),
        "proba_test": proba_test,
        "oof_metrics": classification_metrics(y_dev, oof, threshold),
        "oof_metrics_at_05": classification_metrics(y_dev, oof, 0.5),
        "test_metrics": classification_metrics(y_test, proba_test, threshold),
    }


def comparison_table(selection: dict, final_name: str, test_metrics: dict) -> pd.DataFrame:
    rows = []
    for name, info in selection["models"].items():
        cv = info["cv"]
        selected = name == final_name
        row = {
            "model": name,
            "cv_pr_auc_mean": round(cv["mean_pr_auc"], 4),
            "cv_pr_auc_std": round(cv["std_pr_auc"], 4),
            "cv_roc_auc_mean": round(cv["mean_roc_auc"], 4),
            "cv_roc_auc_std": round(cv["std_roc_auc"], 4),
            "best_params": json.dumps(info["best_params"], sort_keys=True),
            "selected": selected,
        }
        for key in ("roc_auc", "pr_auc", "f1", "precision", "recall"):
            # Test оценивается только для выбранной модели, остальные намеренно не измерены.
            row[f"test_{key}"] = round(test_metrics[key], 4) if selected else "не измерено"
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Модель | CV PR-AUC | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Параметры |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        mark = " (выбрана)" if r["selected"] else ""
        lines.append(
            f"| {r['model']}{mark} | {r['cv_pr_auc_mean']:.4f} ± {r['cv_pr_auc_std']:.4f} "
            f"| {r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f} "
            f"| {r['test_roc_auc']} | {r['test_pr_auc']} | {r['test_f1']} | `{r['best_params']}` |"
        )
    return "\n".join(lines) + "\n"


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


def run_training(config: dict, log=print) -> ChurnModel:
    X, y, ids, masks = load_split_data(config)
    dev, test = masks["train"] | masks["val"], masks["test"]
    X_dev, y_dev, X_test, y_test = X[dev], y[dev], X[test], y[test]
    log(f"Development: {dev.sum()} клиентов, test: {test.sum()} (test используется один раз в конце)")

    selection = run_model_selection(config, X_dev, y_dev, log=log)
    name = selection["best_model"]
    params = selection["models"][name]["best_params"]
    log(f"Лучшая модель по CV {selection['selection_metric']}: {name} {params}")

    final = fit_and_evaluate(config, name, params, X_dev, y_dev, X_test, y_test)
    m = final["test_metrics"]
    log(f"Порог по OOF F1: {final['threshold']:.2f} (OOF F1 {final['oof_metrics']['f1']:.4f})")
    log(
        f"Test: ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f} "
        f"precision={m['precision']:.4f} recall={m['recall']:.4f} CM={m['confusion_matrix']}"
    )

    evaluation = {
        "model": name,
        "params": params,
        "seed": config["seed"],
        "threshold_selection": {
            "data": "out-of-fold predictions on train+val",
            "n_folds": config["cv"]["n_folds"],
            "cv_seed": config["cv"]["oof_seed"],
            "criterion": config["threshold"]["criterion"],
            "threshold": final["threshold"],
        },
        "oof_dev": {"at_selected_threshold": final["oof_metrics"], "at_0.5": final["oof_metrics_at_05"]},
        "test": m,
        "n_dev": int(dev.sum()),
        "n_test": int(test.sum()),
    }
    metadata = {
        "data_url": DATA_URL,
        "data_sha256": DATA_SHA256,
        "config": config,
        "cv": selection["models"][name]["cv"],
        "oof": final["oof_metrics"],
        "test": m,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "catboost": catboost.__version__,
            "pandas": pd.__version__,
        },
    }
    model = ChurnModel(final["pipeline"], final["threshold"], name, params, list(FEATURES), metadata)

    out = config["output"]
    results_dir, figures_dir = Path(out["results_dir"]), Path(out["figures_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    candidates = selection.pop("candidates")
    (results_dir / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    (results_dir / "cv_search.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame(candidates).round(5).to_csv(results_dir / "cv_candidates.csv", index=False)
    (results_dir / "final_evaluation.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame({"customerID": ids[dev].to_numpy(), "y_true": y_dev.to_numpy(), "proba": final["oof"].round(6)}).to_csv(
        results_dir / "oof_predictions.csv", index=False
    )
    pd.DataFrame(
        {"customerID": ids[test].to_numpy(), "y_true": y_test.to_numpy(), "proba": final["proba_test"].round(6)}
    ).to_csv(results_dir / "test_predictions.csv", index=False)
    table = comparison_table(selection, name, m)
    table.to_csv(results_dir / "model_comparison.csv", index=False)
    (results_dir / "model_comparison.md").write_text(comparison_markdown(table))
    plot_curves(y_test, final["proba_test"], final["curve"], final["threshold"], figures_dir / "final_curves.png")

    model.save(Path(out["artifact_path"]))
    log(f"Артефакт: {out['artifact_path']}; результаты: {results_dir}")
    return model

"""Важность признаков, анализ ошибок и проверка на утечку."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from churn.data import CATEGORICAL_FEATURES, DEFAULT_DATA_PATH, FEATURES, download_data, load_dataset
from churn.models import make_model
from churn.splits import DEFAULT_SPLITS_PATH, load_or_create_splits, split_masks
from churn.tuning import make_cv

SEED = 42
N_FOLDS = 5
N_REPEATS = 5
SUSPICIOUS_SINGLE_FEATURE_AUC = 0.9

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]
CHARGES_BINS = [0, 35, 70, 90, 200]
CHARGES_LABELS = ["<35", "35-70", "70-90", ">90"]
GROUP_COLUMNS = [
    "Contract", "tenure_group", "InternetService", "PaymentMethod",
    "monthly_charges_group", "SeniorCitizen", "PaperlessBilling", "TechSupport",
]


def cv_permutation_importance(name, params, X, y) -> pd.DataFrame:
    rows = []
    for fold, (tr, va) in enumerate(make_cv(N_FOLDS, SEED).split(X, y)):
        model = make_model(name, seed=SEED, **params).fit(X.iloc[tr], y.iloc[tr])
        result = permutation_importance(
            model, X.iloc[va], y.iloc[va], scoring="average_precision",
            n_repeats=N_REPEATS, random_state=SEED,
        )
        rows.append(pd.Series(result.importances_mean, index=X.columns, name=fold))
    per_fold = pd.concat(rows, axis=1)
    return pd.DataFrame({"perm_pr_auc_drop_mean": per_fold.mean(axis=1), "perm_pr_auc_drop_std": per_fold.std(axis=1)})


def catboost_importance(model, X) -> pd.DataFrame:
    estimator = model.named_steps["model"].model_
    frame = model.named_steps["prep"].transform(X)
    pool = Pool(frame, cat_features=CATEGORICAL_FEATURES)
    pvc = estimator.get_feature_importance(type="PredictionValuesChange")
    shap = estimator.get_feature_importance(pool, type="ShapValues")[:, :-1]
    return pd.DataFrame(
        {"catboost_prediction_values_change": pvc, "catboost_mean_abs_shap": np.abs(shap).mean(axis=0)},
        index=frame.columns,
    )


def add_groups(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    out["tenure_group"] = pd.cut(out["tenure"], TENURE_BINS, labels=TENURE_LABELS).astype(str)
    out["monthly_charges_group"] = pd.cut(out["MonthlyCharges"], CHARGES_BINS, labels=CHARGES_LABELS).astype(str)
    return out


def error_table(X: pd.DataFrame, y: pd.Series, proba: np.ndarray, threshold: float) -> pd.DataFrame:
    df = add_groups(X)
    pred = (proba >= threshold).astype(int)
    df["fp"] = (pred == 1) & (y.to_numpy() == 0)
    df["fn"] = (pred == 0) & (y.to_numpy() == 1)
    df["neg"] = y.to_numpy() == 0
    df["pos"] = y.to_numpy() == 1
    total_fp, total_fn = df["fp"].sum(), df["fn"].sum()
    rows = []
    for column in GROUP_COLUMNS:
        for value, g in df.groupby(column):
            rows.append({
                "feature": column,
                "value": str(value),
                "n": len(g),
                "churn_rate": g["pos"].mean(),
                "fp_rate": g["fp"].sum() / max(g["neg"].sum(), 1),  # доля ложных тревог среди оставшихся
                "fn_rate": g["fn"].sum() / max(g["pos"].sum(), 1),  # доля пропущенных среди ушедших
                "share_of_all_fp": g["fp"].sum() / total_fp,
                "share_of_all_fn": g["fn"].sum() / total_fn,
            })
    return pd.DataFrame(rows).round(4)


def single_feature_auc(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """ROC-AUC неглубокого дерева на одном признаке (5-fold CV). Подозрительно
    высокое значение означает, что признак почти совпадает с целью."""
    scores = {}
    for column in FEATURES:
        encoder = (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            if column in CATEGORICAL_FEATURES else "passthrough"
        )
        model = Pipeline([
            ("prep", ColumnTransformer([("f", encoder, [column])])),
            ("tree", DecisionTreeClassifier(max_leaf_nodes=8, min_samples_leaf=50, random_state=SEED)),
        ])
        scores[column] = cross_val_score(model, X, y, cv=make_cv(N_FOLDS, SEED), scoring="roc_auc").mean()
    return pd.Series(scores, name="single_feature_roc_auc").sort_values(ascending=False)


def plot_importance(importance: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    perm = importance.sort_values("perm_pr_auc_drop_mean")
    axes[0].barh(perm.index, perm["perm_pr_auc_drop_mean"], xerr=perm["perm_pr_auc_drop_std"])
    axes[0].set(title="Permutation importance (падение PR-AUC, CV)", xlabel="падение PR-AUC")
    shap = importance.sort_values("catboost_mean_abs_shap")
    axes[1].barh(shap.index, shap["catboost_mean_abs_shap"])
    axes[1].set(title="CatBoost: средний |SHAP|", xlabel="лог-шансы")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_errors(errors: pd.DataFrame, path: Path) -> None:
    columns = ["Contract", "tenure_group", "InternetService", "PaymentMethod"]
    fig, axes = plt.subplots(1, len(columns), figsize=(17, 4))
    for ax, column in zip(axes, columns):
        sub = errors[errors["feature"] == column]
        pos = np.arange(len(sub))
        ax.barh(pos - 0.2, sub["fp_rate"], height=0.4, label="FP rate (среди оставшихся)")
        ax.barh(pos + 0.2, sub["fn_rate"], height=0.4, label="FN rate (среди ушедших)")
        ax.set_yticks(pos, sub["value"])
        ax.set(title=column, xlim=(0, 1))
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--final", type=Path, default=Path("results/final_evaluation.json"))
    parser.add_argument("--oof", type=Path, default=Path("results/oof_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    final = json.loads(args.final.read_text())
    name, params = final["model"], final["params"]
    threshold = final["threshold_selection"]["threshold"]

    X, y, ids = load_dataset(download_data(args.data))
    masks = split_masks(ids, load_or_create_splits(ids, y, args.splits, seed=SEED))
    dev = masks["train"] | masks["val"]
    X_dev, y_dev, ids_dev = X[dev].reset_index(drop=True), y[dev].reset_index(drop=True), ids[dev].reset_index(drop=True)

    oof = pd.read_csv(args.oof).set_index("customerID").loc[ids_dev]
    assert (oof["y_true"].to_numpy() == y_dev.to_numpy()).all()

    importance = cv_permutation_importance(name, params, X_dev, y_dev)
    if name == "catboost":
        model = make_model(name, seed=SEED, **params).fit(X_dev, y_dev)
        importance = importance.join(catboost_importance(model, X_dev))
    importance = importance.sort_values("perm_pr_auc_drop_mean", ascending=False).round(5)

    errors = error_table(X_dev, y_dev, oof["proba"].to_numpy(), threshold)
    single_auc = single_feature_auc(X_dev, y_dev).round(4)
    duplicated_rows = int(X_dev.duplicated(keep=False).sum())
    conflicting = int(pd.concat([X_dev, y_dev], axis=1).groupby(FEATURES, dropna=False)["Churn"].nunique().gt(1).sum())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    importance.rename_axis("feature").to_csv(args.output_dir / "feature_importance.csv")
    errors.to_csv(args.output_dir / "error_analysis.csv", index=False)
    plot_importance(importance, args.figures_dir / "feature_importance.png")
    plot_errors(errors, args.figures_dir / "error_rates_by_group.png")

    big = errors[errors["n"] >= 100]
    summary = {
        "model": name,
        "threshold": threshold,
        "data": "train+val only (OOF predictions, CV permutation importance)",
        "top_permutation_features": importance.index[:5].tolist(),
        "highest_fp_rate_groups": big.nlargest(5, "fp_rate")[["feature", "value", "n", "fp_rate"]].to_dict("records"),
        "highest_fn_rate_groups": big.nlargest(5, "fn_rate")[["feature", "value", "n", "fn_rate"]].to_dict("records"),
        "leakage_check": {
            "single_feature_roc_auc": single_auc.to_dict(),
            "suspicious_threshold": SUSPICIOUS_SINGLE_FEATURE_AUC,
            "suspicious_features": single_auc[single_auc > SUSPICIOUS_SINGLE_FEATURE_AUC].index.tolist(),
            "id_in_features": "customerID" in FEATURES,
            "rows_with_duplicate_features": duplicated_rows,
            "duplicate_feature_groups_with_conflicting_labels": conflicting,
        },
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print("Permutation importance (падение PR-AUC):")
    print(importance.head(10).to_string())
    print("\nОдномерный ROC-AUC признаков:")
    print(single_auc.to_string())
    print(f"\nПодозрительные признаки (AUC > {SUSPICIOUS_SINGLE_FEATURE_AUC}): {summary['leakage_check']['suspicious_features']}")
    print(f"Строк с дублирующимися признаками: {duplicated_rows}, групп с разными метками: {conflicting}")
    print("\nГруппы с наибольшей долей FP:\n", big.nlargest(5, "fp_rate").to_string(index=False))
    print("\nГруппы с наибольшей долей FN:\n", big.nlargest(5, "fn_rate").to_string(index=False))


if __name__ == "__main__":
    main()

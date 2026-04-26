"""Интерпретация и анализ ошибок выбранной модели (после train.py).

Всё считается только на train и только на данных, которые модель не видела:
SHAP и permutation importance — на отложенной части каждого фолда CV, ошибки —
по OOF-предсказаниям в выбранной точке работы (top-K по прибыли).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from churn.config import DEFAULT_CONFIG_PATH, load_config
from churn.data import CATEGORICAL_FEATURES, FEATURES
from churn.features import SERVICE_COLUMNS, linear_feature_sources
from churn.models import make_model
from churn.plots import AQUA, BLUE, GREY, MUTED, ORANGE, TEXT, _save, plt
from churn.training import dev_and_test
from churn.tuning import make_cv

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]
CHARGES_BINS = [0, 35, 70, 90, 200]
CHARGES_LABELS = ["<35", "35-70", "70-90", ">90"]
GROUP_COLUMNS = [
    "Contract", "tenure_group", "InternetService", "PaymentMethod",
    "monthly_charges_group", "SeniorCitizen", "PaperlessBilling", "TechSupport",
]
CONTRACT_COLORS = {"Month-to-month": BLUE, "One year": ORANGE, "Two year": AQUA}


def add_groups(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    # Упорядоченные категории: группы в таблицах и на графиках идут по возрастанию.
    out["tenure_group"] = pd.cut(out["tenure"], TENURE_BINS, labels=TENURE_LABELS)
    out["monthly_charges_group"] = pd.cut(out["MonthlyCharges"], CHARGES_BINS, labels=CHARGES_LABELS)
    return out


def _share(numerator, denominator) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def error_table(X: pd.DataFrame, y: pd.Series, selected: np.ndarray) -> pd.DataFrame:
    """Ошибки правила «контакт с выбранными клиентами» по группам.
    missed_rate — доля ушедших группы, не попавших в список (FN среди ушедших);
    wasted_rate — доля оставшихся, попавших в список (FP среди оставшихся)."""
    df = add_groups(X)
    y = np.asarray(y)
    df["selected"], df["pos"] = selected, y == 1
    df["missed"] = df["pos"] & ~df["selected"]
    df["wasted"] = ~df["pos"] & df["selected"]
    total_missed, total_wasted = df["missed"].sum(), df["wasted"].sum()
    rows = []
    for column in GROUP_COLUMNS:
        for value, g in df.groupby(column, observed=True):
            rows.append({
                "feature": column,
                "value": str(value),
                "n": len(g),
                "churn_rate": g["pos"].mean(),
                "selected_share": g["selected"].mean(),
                "precision_in_selected": _share((g["selected"] & g["pos"]).sum(), g["selected"].sum()),
                "missed_rate": _share(g["missed"].sum(), g["pos"].sum()),
                "wasted_rate": _share(g["wasted"].sum(), (~g["pos"]).sum()),
                "share_of_all_missed": _share(g["missed"].sum(), total_missed),
                "share_of_all_wasted": _share(g["wasted"].sum(), total_wasted),
            })
    return pd.DataFrame(rows).round(4)


def fold_shap(pipeline: Pipeline, X_train: pd.DataFrame, X_eval: pd.DataFrame) -> pd.DataFrame:
    """SHAP-значения (в лог-шансах) обученного Pipeline для X_eval.

    CatBoost: встроенные TreeSHAP-значения. LogReg: точные SHAP линейной модели
    w·(z − E[z]) с фоном — средним по обучающей части фолда; столбцы one-hot и
    сплайнов суммируются обратно в исходный признак."""
    transform, model = pipeline[:-1], pipeline.named_steps["model"]
    Z_eval = transform.transform(X_eval)
    if hasattr(model, "model_"):  # CatBoostCatClassifier
        pool = Pool(Z_eval, cat_features=list(model.cat_features))
        values = model.model_.get_feature_importance(pool, type="ShapValues")[:, :-1]
        return pd.DataFrame(values, columns=Z_eval.columns, index=X_eval.index)
    if hasattr(model, "coef_"):
        Z_train = np.asarray(transform.transform(X_train), dtype=float)
        contrib = (np.asarray(Z_eval, dtype=float) - Z_train.mean(axis=0)) * model.coef_[0]
        sources = linear_feature_sources(pipeline.named_steps["prep"])
        frame = pd.DataFrame(contrib, columns=sources, index=X_eval.index)
        return frame.T.groupby(level=0, sort=False).sum().T
    raise ValueError(f"SHAP не поддерживается для {type(model).__name__}")


def oof_shap(spec: dict, X: pd.DataFrame, y: pd.Series, cv, seed: int) -> pd.DataFrame:
    template = make_model(spec["name"], seed=seed, features=spec.get("features", ()), **spec["params"])
    parts = []
    for tr, te in cv.split(X, y):
        fitted = clone(template).fit(X.iloc[tr], y.iloc[tr])
        parts.append(fold_shap(fitted, X.iloc[tr], X.iloc[te]))
    return pd.concat(parts).sort_index()


def cv_permutation_importance(spec: dict, X, y, cv, seed: int, n_repeats: int) -> pd.DataFrame:
    template = make_model(spec["name"], seed=seed, features=spec.get("features", ()), **spec["params"])
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        model = clone(template).fit(X.iloc[tr], y.iloc[tr])
        result = permutation_importance(
            model, X.iloc[te], y.iloc[te], scoring="average_precision", n_repeats=n_repeats, random_state=seed,
        )
        rows.append(pd.Series(result.importances_mean, index=X.columns, name=fold))
    per_fold = pd.concat(rows, axis=1)
    return pd.DataFrame({"perm_pr_auc_drop_mean": per_fold.mean(axis=1), "perm_pr_auc_drop_std": per_fold.std(axis=1)})


def single_feature_auc(X: pd.DataFrame, y: pd.Series, cv, seed: int) -> pd.Series:
    """ROC-AUC неглубокого дерева на одном признаке. Значение, близкое к 1, означает,
    что признак почти совпадает с целью (так ловят случайно попавшую в признаки цель)."""
    scores = {}
    for column in FEATURES:
        encoder = (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            if column in CATEGORICAL_FEATURES else "passthrough"
        )
        model = Pipeline([
            ("prep", ColumnTransformer([("f", encoder, [column])])),
            ("tree", DecisionTreeClassifier(max_leaf_nodes=8, min_samples_leaf=50, random_state=seed)),
        ])
        scores[column] = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()
    return pd.Series(scores, name="single_feature_roc_auc").sort_values(ascending=False)


def missed_profile(X: pd.DataFrame, y: pd.Series, proba: np.ndarray, selected: np.ndarray, contract: str) -> dict:
    """Чем пропущенные ушедшие сегмента отличаются от найденных."""
    mask = (X["Contract"] == contract).to_numpy() & (np.asarray(y) == 1)

    def describe(m):
        part = X[m]
        if not m.any():
            return {"n": 0}
        return {
            "n": int(m.sum()),
            "median_tenure": float(part["tenure"].median()),
            "median_monthly_charges": float(part["MonthlyCharges"].median()),
            "share_fiber": float((part["InternetService"] == "Fiber optic").mean()),
            "share_electronic_check": float((part["PaymentMethod"] == "Electronic check").mean()),
            "median_proba": float(np.median(proba[m])),
        }

    return {"caught": describe(mask & selected), "missed": describe(mask & ~selected)}


def plot_importance(importance: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    perm = importance.sort_values("perm_pr_auc_drop_mean").dropna(subset=["perm_pr_auc_drop_mean"])
    axes[0].barh(perm.index, perm["perm_pr_auc_drop_mean"], xerr=perm["perm_pr_auc_drop_std"], color=BLUE,
                 ecolor=MUTED, height=0.6)
    axes[0].set(title="Permutation importance (падение PR-AUC, отложенные фолды)", xlabel="падение PR-AUC")
    shap = importance.sort_values("mean_abs_shap_oof").dropna(subset=["mean_abs_shap_oof"])
    axes[1].barh(shap.index, shap["mean_abs_shap_oof"], color=BLUE, height=0.6)
    axes[1].set(title="Средний |SHAP| на OOF", xlabel="лог-шансы")
    _save(fig, path)


def dependence_values(shap: pd.DataFrame, column: str) -> tuple[pd.Series, str]:
    """SHAP признака; если MonthlyCharges нет в модели — суммарный SHAP услуг,
    из которых он почти линейно складывается."""
    if column in shap.columns:
        return shap[column], f"SHAP {column}, лог-шансы"
    if column == "MonthlyCharges":
        services = [c for c in ["InternetService", *SERVICE_COLUMNS] if c in shap.columns]
        return shap[services].sum(axis=1), "сумма SHAP услуг, лог-шансы\n(MonthlyCharges не в модели)"
    raise KeyError(column)


def plot_dependence(X: pd.DataFrame, shap: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, column in zip(axes, ["tenure", "MonthlyCharges"]):
        values, ylabel = dependence_values(shap, column)
        for contract, color in CONTRACT_COLORS.items():
            m = (X["Contract"] == contract).to_numpy()
            ax.scatter(X.loc[m, column], values[m], s=8, alpha=0.45, color=color, label=contract,
                       edgecolors="none")
        ax.axhline(0, color=GREY, lw=1)
        ax.set(title=f"SHAP dependence: {column} (OOF)", xlabel=column, ylabel=ylabel)
        ax.legend(title="Contract", markerscale=2.5, fontsize=8, title_fontsize=8)
    _save(fig, path)


def plot_errors(errors: pd.DataFrame, path: Path) -> None:
    columns = ["Contract", "tenure_group", "InternetService", "PaymentMethod"]
    fig, axes = plt.subplots(1, len(columns), figsize=(17, 4))
    for ax, column in zip(axes, columns):
        sub = errors[errors["feature"] == column]
        pos = np.arange(len(sub))
        ax.barh(pos - 0.2, sub["missed_rate"], height=0.38, color=ORANGE,
                label="доля ушедших, не попавших в список")
        ax.barh(pos + 0.2, sub["wasted_rate"], height=0.38, color=BLUE,
                label="доля оставшихся, попавших в список")
        ax.set_yticks(pos, sub["value"])
        ax.invert_yaxis()
        ax.set(title=column, xlim=(0, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    results_dir, figures_dir = Path(config["output"]["results_dir"]), Path(config["output"]["figures_dir"])
    seed = config["seed"]

    final = json.loads((results_dir / "final_evaluation.json").read_text(encoding="utf-8"))
    spec = final["model"]
    point = final["business"]["operating_point"]
    (X, y, ids), _ = dev_and_test(config)
    oof = pd.read_csv(results_dir / "oof_predictions.csv").set_index("customerID").loc[ids]
    if not (oof["y_true"].to_numpy() == y.to_numpy()).all():
        raise ValueError("oof_predictions.csv не соответствует текущему train; перезапустите train.py")
    proba = oof["proba"].to_numpy()
    selected = proba >= point["threshold"]
    cv = make_cv(config["cv"]["n_splits"], config["cv"]["oof_seed"])

    importance = cv_permutation_importance(spec, X, y, cv, seed, config["analysis"]["permutation_repeats"])
    shap = oof_shap(spec, X, y, cv, seed)
    importance = importance.join(shap.abs().mean().rename("mean_abs_shap_oof"), how="outer")
    importance = importance.sort_values("mean_abs_shap_oof", ascending=False).round(5)

    errors = error_table(X, y, selected)
    single_auc = single_feature_auc(X, y, cv, seed).round(4)
    threshold_auc = config["analysis"]["dominant_feature_auc"]
    conflicting = int(pd.concat([X, y], axis=1).groupby(FEATURES, dropna=False)["Churn"].nunique().gt(1).sum())

    importance.rename_axis("feature").to_csv(results_dir / "feature_importance.csv")
    errors.to_csv(results_dir / "error_analysis.csv", index=False)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_importance(importance, figures_dir / "feature_importance.png")
    plot_dependence(X, shap, figures_dir / "shap_dependence.png")
    plot_errors(errors, figures_dir / "error_rates_by_group.png")

    contract_rows = errors[errors["feature"] == "Contract"].set_index("value")
    summary = {
        "model": spec,
        "operating_point": {"k_fraction": point["k_fraction"], "threshold": point["threshold"]},
        "data": "train only: OOF-предсказания, SHAP и permutation importance на отложенных фолдах",
        "top_features_by_mean_abs_shap": importance.index[:6].tolist(),
        "contract_errors": contract_rows[["n", "churn_rate", "selected_share", "missed_rate", "wasted_rate",
                                          "share_of_all_missed"]].to_dict("index"),
        "one_year_missed_vs_caught": missed_profile(X, y, proba, selected, "One year"),
        "two_year_missed_vs_caught": missed_profile(X, y, proba, selected, "Two year"),
        "dominant_feature_check": {
            "description": "одномерный ROC-AUC каждого признака; значение выше порога означало бы, "
                           "что признак почти совпадает с целью",
            "single_feature_roc_auc": single_auc.to_dict(),
            "threshold": threshold_auc,
            "flagged": single_auc[single_auc > threshold_auc].index.tolist(),
        },
        "data_checks": {
            "id_in_features": "customerID" in FEATURES,
            "rows_with_duplicate_features": int(X.duplicated(keep=False).sum()),
            "duplicate_feature_groups_with_conflicting_labels": conflicting,
        },
    }
    (results_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8"
    )

    print("Важность признаков (OOF):")
    print(importance.head(10).to_string())
    print("\nОшибки по типу контракта в точке работы:")
    print(contract_rows.to_string())
    profile = json.dumps(summary["one_year_missed_vs_caught"], ensure_ascii=False)
    print("\nOne year, найденные vs пропущенные ушедшие:", profile)
    print(f"\nПризнаки с одномерным AUC > {threshold_auc}: {summary['dominant_feature_check']['flagged']}")


if __name__ == "__main__":
    main()

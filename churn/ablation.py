"""Абляция построчных признаков и проверка мультиколлинеарности (VIF).

Каждый кандидат добавляется к исходным признакам по одному и сравнивается с
базой на одних и тех же фолдах повторной CV; разность проверяется
корректированным t-тестом. Правило (записано до запуска): признак оставляем,
если 95% CI прироста selection.metric целиком выше нуля. Отдельно смотрим главный
вопрос этапа — ROC-AUC внутри годовых контрактов.

Параметры моделей берутся из results/cv_search.json (после train.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import DEFAULT_CONFIG_PATH, load_config
from churn.features import SERVICE_COLUMNS, add_engineered, column_groups
from churn.metrics import LOWER_IS_BETTER
from churn.stats import corrected_ttest
from churn.training import dev_and_test
from churn.tuning import cross_validate_spec, make_repeated_cv

REPORTED = ("lift@10", "pr_auc", "log_loss", "roc_auc[One year]", "roc_auc[Two year]")


def vif(frame: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 − R²_j), R²_j — от регрессии признака j на остальные (с константой)."""
    values = frame.to_numpy(dtype=float)
    values = (values - values.mean(axis=0)) / values.std(axis=0)
    out = {}
    for j, column in enumerate(frame.columns):
        others = np.column_stack([np.ones(len(values)), np.delete(values, j, axis=1)])
        coef, *_ = np.linalg.lstsq(others, values[:, j], rcond=None)
        resid = values[:, j] - others @ coef
        r2 = 1 - resid.var() / values[:, j].var()
        out[column] = 1 / max(1 - r2, 1e-12)
    return pd.Series(out, name="vif")


def numeric_vif(X: pd.DataFrame, features=(), with_services: bool = False) -> pd.Series:
    """VIF числовых признаков; with_services добавляет в регрессию one-hot услуг
    (MonthlyCharges почти линейно складывается из них)."""
    numeric, _, _ = column_groups(features)
    frame = add_engineered(X, features)
    values = frame[numeric].fillna(frame[numeric].median())
    if with_services:
        services = pd.get_dummies(frame[["InternetService", *SERVICE_COLUMNS]], drop_first=True).astype(float)
        return vif(pd.concat([values, services], axis=1))[numeric]
    return vif(values)


def run_ablation(config: dict, X, y, model: str, params: dict, log=print) -> pd.DataFrame:
    cv_conf = config["cv"]
    rcv = make_repeated_cv(cv_conf["n_splits"], cv_conf["n_repeats"], cv_conf["comparison_seed"])
    segments = X["Contract"].to_numpy()
    fractions = tuple(config["ranking"]["top_fractions"])
    metric = config["selection"]["metric"]

    def run(features):
        spec = {"name": model, "params": params, "features": list(features)}
        return cross_validate_spec(spec, X, y, rcv, seed=config["seed"], fractions=fractions, segments=segments)

    base = run([])
    n_train, n_test = int(base["n_train"].mean()), int(base["n_test"].mean())
    rows = [{"model": model, "variant": "base", **{f"{m}_mean": base[m].mean() for m in REPORTED}}]
    for feature in config["features"]["candidates"]:
        folds = run([feature])
        row = {"model": model, "variant": f"+{feature}", **{f"{m}_mean": folds[m].mean() for m in REPORTED}}
        for m in REPORTED:
            test = corrected_ttest(folds[m].to_numpy() - base[m].to_numpy(), n_train, n_test)
            row.update({f"{m}_diff": test["mean_diff"], f"{m}_ci_low": test["ci_low"], f"{m}_ci_high": test["ci_high"]})
        sign = -1 if metric in LOWER_IS_BETTER else 1
        low = row[f"{metric}_ci_low"] if sign > 0 else -row[f"{metric}_ci_high"]
        row["keep"] = bool(low > 0)
        rows.append(row)
        log(f"  {model} +{feature:20s} Δ{metric}={row[f'{metric}_diff']:+.3f} "
            f"[{row[f'{metric}_ci_low']:+.3f}; {row[f'{metric}_ci_high']:+.3f}], "
            f"ΔAUC(One year)={row['roc_auc[One year]_diff']:+.3f}, оставить: {row['keep']}")
    kept = [r["variant"][1:] for r in rows[1:] if r["keep"]]
    candidates = config["features"]["candidates"]
    configured = list(config["features"]["engineered"])
    combos = [("all_candidates", candidates), ("kept", kept), ("configured", configured)]
    for label, features in combos:
        if not features or (label != "all_candidates" and features == candidates):
            continue
        if label == "configured" and (len(features) == 1 or features == kept):
            continue  # уже есть в таблице отдельной строкой
        folds = run(features)
        row = {"model": model, "variant": label + ": " + ",".join(features),
               **{f"{m}_mean": folds[m].mean() for m in REPORTED}}
        for m in REPORTED:
            test = corrected_ttest(folds[m].to_numpy() - base[m].to_numpy(), n_train, n_test)
            row.update({f"{m}_diff": test["mean_diff"], f"{m}_ci_low": test["ci_low"], f"{m}_ci_high": test["ci_high"]})
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_markdown(table: pd.DataFrame, metric: str) -> str:
    lines = [
        f"| Модель | Вариант | {metric} | Δ{metric} [95% CI] | ΔPR-AUC [95% CI] | Δlog-loss "
        "| ΔROC-AUC One year [95% CI] | Оставить |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def ci(r, m, fmt="+.3f"):
        if pd.isna(r.get(f"{m}_diff")):
            return "–"
        return f"{r[f'{m}_diff']:{fmt}} [{r[f'{m}_ci_low']:{fmt}}; {r[f'{m}_ci_high']:{fmt}}]"

    for _, r in table.iterrows():
        keep = "–" if pd.isna(r.get("keep")) else ("да" if r["keep"] else "нет")
        loss = "–" if pd.isna(r.get("log_loss_diff")) else f"{r['log_loss_diff']:+.4f}"
        lines.append(
            f"| {r['model']} | {r['variant']} | {r[f'{metric}_mean']:.3f} | {ci(r, metric)} "
            f"| {ci(r, 'pr_auc', '+.4f')} "
            f"| {loss} | {ci(r, 'roc_auc[One year]')} | {keep} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models", nargs="+", default=["logreg", "catboost"])
    args = parser.parse_args()
    config = load_config(args.config)
    results_dir = Path(config["output"]["results_dir"])
    tuned = json.loads((results_dir / "cv_search.json").read_text(encoding="utf-8"))["models"]

    (X, y, _), _ = dev_and_test(config)
    vif_table = pd.DataFrame({
        "base": numeric_vif(X),
        "base_with_service_dummies": numeric_vif(X, with_services=True),
        "drop_total_charges": numeric_vif(X, ["drop_total_charges"]),
        "drop_both_with_service_dummies": numeric_vif(
            X, ["drop_total_charges", "drop_monthly_charges"], with_services=True
        ),
        "all_numeric_candidates": numeric_vif(X, ["n_services", "avg_charge", "contract_tenure"]),
    }).round(2)
    vif_table.rename_axis("feature").to_csv(results_dir / "vif.csv")
    print("VIF числовых признаков (train):\n", vif_table.to_string())

    tables = []
    for model in args.models:
        print(f"Абляция {model} {tuned[model]['best_params']}:")
        tables.append(run_ablation(config, X, y, model, tuned[model]["best_params"]))
    table = pd.concat(tables, ignore_index=True)
    previous = results_dir / "ablation.csv"
    if previous.exists():  # строки моделей, которые не пересчитывались, сохраняются
        old = pd.read_csv(previous)
        table = pd.concat([old[~old["model"].isin(args.models)], table], ignore_index=True)
        table = table.sort_values("model", key=lambda s: s.map({"logreg": 0}).fillna(1), kind="stable")
    table.round(5).to_csv(results_dir / "ablation.csv", index=False)
    (results_dir / "ablation.md").write_text(ablation_markdown(table, config["selection"]["metric"]), encoding="utf-8")
    print(f"Результаты: {results_dir / 'ablation.csv'}, {results_dir / 'vif.csv'}")


if __name__ == "__main__":
    main()

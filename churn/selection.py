"""Выбор модели: подбор параметров каждой модели и сравнение на повторной CV.

1. Подбор гиперпараметров: 5-fold CV, критерий cv.tuning_metric (log-loss).
2. Сравнение лучших конфигураций: RepeatedStratifiedKFold (n_splits × n_repeats),
   одни и те же фолды для всех моделей, метрика selection.metric (lift@10%).
3. Разности с базовой моделью проверяются корректированным t-тестом Nadeau–Bengio.
4. Правило, записанное в конфиге до запуска: если ни одна модель не лучше базовой
   значимо (p < alpha), выбирается базовая (LogReg).
"""

from __future__ import annotations

import time

import pandas as pd

from churn.metrics import LOWER_IS_BETTER
from churn.stats import corrected_ttest
from churn.tuning import cross_validate_spec, make_cv, make_repeated_cv, search_model, summarize


def fractions_of(config: dict) -> tuple[float, ...]:
    return tuple(config["ranking"]["top_fractions"])


def tune_models(config: dict, X, y, log=print) -> dict:
    cv_conf, features = config["cv"], config["features"]["engineered"]
    cv = make_cv(cv_conf["n_splits"], cv_conf["tuning_seed"])
    tuned = {}
    for name, grid in config["models"].items():
        start = time.perf_counter()
        result = search_model(
            name, X, y, cv, grid, seed=config["seed"], features=features,
            fractions=fractions_of(config), metric=config["selection"]["tuning_metric"],
        )
        result["seconds"] = round(time.perf_counter() - start, 1)
        tuned[name] = result
        log(f"  подбор {name:13s}: {len(result['candidates'])} комбинаций, {result['seconds']:.0f} c, "
            f"{result['best_params']}")
    return tuned


def compare_models(config: dict, X, y, specs: dict[str, dict], log=print) -> dict:
    """Все модели на одних и тех же фолдах повторной CV + t-тест против базовой."""
    cv_conf, sel = config["cv"], config["selection"]
    rcv = make_repeated_cv(cv_conf["n_splits"], cv_conf["n_repeats"], cv_conf["comparison_seed"])
    folds = {}
    for name, spec in specs.items():
        start = time.perf_counter()
        folds[name] = cross_validate_spec(
            spec, X, y, rcv, seed=config["seed"], fractions=fractions_of(config), segments=X["Contract"].to_numpy()
        )
        log(f"  сравнение {name:13s}: {sel['metric']}={folds[name][sel['metric']].mean():.4f} "
            f"({time.perf_counter() - start:.0f} c)")

    baseline = sel["baseline"]
    metric = sel["metric"]
    n_train, n_test = int(folds[baseline]["n_train"].mean()), int(folds[baseline]["n_test"].mean())
    tests = {}
    for name in specs:
        if name in (baseline, "dummy"):
            continue
        tests[name] = {}
        for m in sorted({metric, "pr_auc", "roc_auc", "log_loss", "roc_auc[One year]"}):
            diff = folds[name][m].to_numpy() - folds[baseline][m].to_numpy()
            tests[name][m] = corrected_ttest(diff, n_train, n_test)

    sign = -1 if metric in LOWER_IS_BETTER else 1
    better = [
        name for name, t in tests.items()
        if t[metric]["p_value"] < sel["alpha"] and sign * t[metric]["mean_diff"] > 0
    ]
    chosen = max(better, key=lambda n: sign * folds[n][metric].mean()) if better else baseline
    return {
        "metric": metric,
        "baseline": baseline,
        "alpha": sel["alpha"],
        "rule": sel["rule"],
        "n_splits": cv_conf["n_splits"],
        "n_repeats": cv_conf["n_repeats"],
        "seed": cv_conf["comparison_seed"],
        "summary": {name: summarize(f) for name, f in folds.items()},
        "tests_vs_baseline": tests,
        "significantly_better": better,
        "chosen": chosen,
        "folds": folds,
    }


def run_model_selection(config: dict, X, y, log=print) -> dict:
    log(f"Подбор параметров ({config['cv']['n_splits']}-fold CV, критерий {config['selection']['tuning_metric']}):")
    tuned = tune_models(config, X, y, log=log)
    features = config["features"]["engineered"]
    specs = {name: {"name": name, "params": r["best_params"], "features": list(features)} for name, r in tuned.items()}
    log(f"Сравнение на RepeatedStratifiedKFold {config['cv']['n_splits']}×{config['cv']['n_repeats']}:")
    comparison = compare_models(config, X, y, specs, log=log)
    return {"tuned": tuned, "specs": specs, "comparison": comparison}


def comparison_table(selection: dict, fractions=(0.1, 0.2)) -> pd.DataFrame:
    comp = selection["comparison"]
    rows = []
    for name, s in comp["summary"].items():
        row = {"model": name, "selected": name == comp["chosen"]}
        for m in [*[f"lift@{round(f * 100)}" for f in fractions], "pr_auc", "roc_auc", "log_loss", "brier",
                  "roc_auc[One year]", "roc_auc[Two year]"]:
            row[f"{m}_mean"] = round(s[f"mean_{m}"], 4)
            row[f"{m}_std"] = round(s[f"std_{m}"], 4)
        test = comp["tests_vs_baseline"].get(name, {}).get(comp["metric"])
        row[f"diff_vs_{comp['baseline']}"] = round(test["mean_diff"], 4) if test else None
        row["diff_ci_low"] = round(test["ci_low"], 4) if test else None
        row["diff_ci_high"] = round(test["ci_high"], 4) if test else None
        row["p_value"] = round(test["p_value"], 4) if test else None
        row["params"] = selection["specs"][name]["params"]
        rows.append(row)
    return pd.DataFrame(rows)

"""Полный цикл обучения, который запускает train.py.

Без флага evaluate_test test-выборка не читается для оценки: всё решается на
кросс-валидации внутри train. Каждая оценка на test записывается в журнал
results/test_access_log.json.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import pandas as pd

from churn.data import DATA_SHA256, DATA_URL, FEATURES, download_data, load_dataset
from churn.economics import Economics
from churn.evaluation import business_analysis, calibration_report, evaluate_scores
from churn.inference import ChurnModel, library_versions
from churn.models import make_estimator
from churn.plots import plot_calibration, plot_gains_and_profit, plot_segments, plot_test_gains
from churn.selection import comparison_table, run_model_selection
from churn.splits import load_or_create_splits, split_masks
from churn.tuning import make_cv, oof_predict

SEGMENT = "Contract"


def load_split_data(config: dict):
    data = config["data"]
    path = Path(data["path"])
    if data.get("download", True):
        path = download_data(path)
    elif not path.exists():
        raise FileNotFoundError(f"Нет файла {path}. Для синтетических данных: python -m churn.synthetic")
    X, y, ids = load_dataset(path)
    splits = load_or_create_splits(ids, y, Path(data["splits_path"]), seed=config["seed"], test_size=data["test_size"])
    masks = split_masks(ids, splits)
    return X, y, ids, masks


def dev_and_test(config: dict):
    X, y, ids, masks = load_split_data(config)
    parts = {}
    for name in ("train", "test"):
        m = masks[name]
        parts[name] = (X[m].reset_index(drop=True), y[m].reset_index(drop=True), ids[m].reset_index(drop=True))
    return parts["train"], parts["test"]


def run_id(config: dict, spec: dict) -> str:
    payload = json.dumps({"config": config, "spec": spec}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=float) + "\n"


def _round_floats(obj, digits=4):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, digits) for v in obj]
    return obj


def comparison_markdown(table: pd.DataFrame, comparison: dict) -> str:
    metric, baseline = comparison["metric"], comparison["baseline"]
    lines = [
        f"Повторная CV {comparison['n_splits']}×{comparison['n_repeats']} на train, среднее ± std по фолдам. "
        f"Разность с {baseline}: corrected t-test Nadeau–Bengio, 95% CI.",
        "",
        f"| Модель | {metric} | PR-AUC | ROC-AUC | log-loss | ROC-AUC внутри One year "
        f"| Δ{metric} vs {baseline} [95% CI] | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        mark = " (выбрана)" if r["selected"] else ""
        diff = "–"
        if not pd.isna(r["p_value"]):
            diff = f"{r[f'diff_vs_{baseline}']:+.3f} [{r['diff_ci_low']:+.3f}; {r['diff_ci_high']:+.3f}]"
        p = "–" if pd.isna(r["p_value"]) else f"{r['p_value']:.3f}"
        lines.append(
            f"| {r['model']}{mark} | {r[f'{metric}_mean']:.3f} ± {r[f'{metric}_std']:.3f} "
            f"| {r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f} | {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} "
            f"| {r['log_loss_mean']:.4f} | {r['roc_auc[One year]_mean']:.3f} ± {r['roc_auc[One year]_std']:.3f} "
            f"| {diff} | {p} |"
        )
    lines += ["", comparison["rule"], f"Выбрана: **{comparison['chosen']}**."]
    return "\n".join(lines) + "\n"


def run_training(config: dict, evaluate_test: bool = False, log=print) -> ChurnModel:
    (X_dev, y_dev, ids_dev), (X_test, y_test, ids_test) = dev_and_test(config)
    log(f"Train: {len(y_dev)} клиентов (вся работа на CV внутри него), test: {len(y_test)} "
        f"({'будет оценён' if evaluate_test else 'не используется, нужен флаг --evaluate-test'})")
    out = config["output"]
    results_dir, figures_dir = Path(out["results_dir"]), Path(out["figures_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    econ = Economics.from_config(config["business"])
    fractions = tuple(config["ranking"]["top_fractions"])

    # 1. Подбор параметров и сравнение моделей на повторной CV.
    selection = run_model_selection(config, X_dev, y_dev, log=log)
    comparison = selection["comparison"]
    name = comparison["chosen"]
    spec = dict(selection["specs"][name])
    log(f"Выбрана модель: {name} {spec['params']} (значимо лучше {comparison['baseline']}: "
        f"{comparison['significantly_better'] or 'нет'})")

    # 2. OOF выбранной модели, с калибратором и без; калибратор обучается только внутри фолдов.
    oof_cv = make_cv(config["cv"]["n_splits"], config["cv"]["oof_seed"])
    oof_raw = oof_predict(spec, X_dev, y_dev, oof_cv, seed=config["seed"])
    oof_cal = oof_predict({**spec, "calibration": config["calibration"]["method"]}, X_dev, y_dev, oof_cv,
                          seed=config["seed"])
    calibration = calibration_report(y_dev, oof_raw, oof_cal, config)
    if calibration["applied"]:
        spec["calibration"] = config["calibration"]["method"]
    oof = oof_cal if calibration["applied"] else oof_raw
    imp = calibration["brier_improvement"]
    log(f"Калибровка: Brier {calibration['raw']['brier']:.4f} -> {calibration['calibrated']['brier']:.4f} "
        f"(улучшение {imp['value']:+.5f} [{imp['ci_low']:+.5f}; {imp['ci_high']:+.5f}]), "
        f"применяется: {calibration['applied']}")

    # 3. Оценка на OOF и выбор точки работы по прибыли (порог — после калибровки).
    segments_dev = X_dev[SEGMENT].to_numpy()
    oof_eval = evaluate_scores(y_dev, oof, segments_dev, econ, config)
    business = business_analysis(y_dev, oof, segments_dev, econ, config)
    point = oof_eval["operating_point"]
    m = oof_eval["metrics"]
    log(f"OOF: lift@10={m['lift@10']['value']:.2f} [{m['lift@10']['ci_low']:.2f}; {m['lift@10']['ci_high']:.2f}], "
        f"PR-AUC={m['pr_auc']['value']:.4f}")
    log(f"Точка работы: контакт с верхними {point['k_fraction']:.1%} (порог {point['threshold']:.3f}), "
        f"прибыль {point['profit']:.0f} у.е. на {len(y_dev)} клиентов")

    # 4. Финальная модель на всём train.
    estimator = make_estimator(spec, seed=config["seed"]).fit(X_dev, y_dev)
    rid = run_id(config, spec)
    versions = library_versions()
    evaluation = {
        "run_id": rid,
        "task": "ранжирование клиентов по риску ухода по снимку данных; бюджет на удержание K клиентов",
        "model": spec,
        "selection": {
            k: comparison[k] for k in ("metric", "baseline", "alpha", "rule", "significantly_better", "chosen")
        },
        "calibration": calibration,
        "oof_train": {
            "n": oof_eval["n"],
            "cv": f"StratifiedKFold({config['cv']['n_splits']}, seed={config['cv']['oof_seed']})",
            "metrics": oof_eval["metrics"],
            "at_threshold": oof_eval["at_threshold"],
        },
        "business": {
            "assumptions": config["business"],
            "break_even_probability": econ.break_even_probability(),
            "operating_point": point,
            "segment_cutoffs": business["segment_cutoffs"],
        },
        "test": "не оценивался в этом запуске (python train.py --evaluate-test)",
        "versions": versions,
        "data": {"url": DATA_URL if config["data"].get("download", True) else config["data"]["path"],
                 "sha256": DATA_SHA256 if config["data"].get("download", True) else None},
    }

    # 5. Сохранение.
    tuned = selection["tuned"]
    (results_dir / "train_config.json").write_text(_json(config), encoding="utf-8")
    (results_dir / "cv_search.json").write_text(_json(_round_floats({
        "tuning_metric": config["selection"]["tuning_metric"],
        "cv": f"StratifiedKFold({config['cv']['n_splits']}, seed={config['cv']['tuning_seed']})",
        "models": {n: {"best_params": r["best_params"], "cv": r["cv"], "n_candidates": len(r["candidates"]),
                       "seconds": r["seconds"]} for n, r in tuned.items()},
    })), encoding="utf-8")
    pd.DataFrame([
        {"model": n, "params": json.dumps(c["params"], sort_keys=True), **{k: v for k, v in c.items() if k != "params"}}
        for n, r in tuned.items() for c in r["candidates"]
    ]).round(5).to_csv(results_dir / "cv_candidates.csv", index=False)
    pd.concat([f.assign(model=n) for n, f in comparison["folds"].items()]).round(5).to_csv(
        results_dir / "cv_folds.csv", index=False)
    table = comparison_table(selection, fractions)
    table.to_csv(results_dir / "model_comparison.csv", index=False)
    (results_dir / "model_comparison.md").write_text(comparison_markdown(table, comparison), encoding="utf-8")
    (results_dir / "model_comparison.json").write_text(_json(_round_floats(
        {k: v for k, v in comparison.items() if k != "folds"}, 5)), encoding="utf-8")
    oof_eval["segments"].round(4).to_csv(results_dir / "segment_metrics_oof.csv", index=False)
    business["sensitivity"].round(4).to_csv(results_dir / "profit_sensitivity.csv", index=False)
    pd.DataFrame({
        "customerID": ids_dev, "y_true": y_dev, SEGMENT: segments_dev,
        "proba_raw": oof_raw.round(6), "proba": oof.round(6),
    }).to_csv(results_dir / "oof_predictions.csv", index=False)
    plot_gains_and_profit(y_dev, oof, econ, point, business["sensitivity"], figures_dir / "gains_profit.png", fractions)
    plot_calibration(calibration, figures_dir / "calibration.png")
    plot_segments(oof_eval["segments"], figures_dir / "segments_oof.png", fractions)

    model = ChurnModel(
        pipeline=estimator, threshold=point["threshold"], model_name=name, params=spec["params"],
        features=list(FEATURES),
        metadata={"run_id": rid, "spec": spec, "operating_point": point, "versions": versions,
                  "config": config, "oof": {k: v["value"] for k, v in oof_eval["metrics"].items()}},
    )

    # 6. Test — только по флагу.
    if evaluate_test:
        evaluation["test"] = evaluate_on_test(config, model, X_test, y_test, ids_test, y_dev, oof, econ, log=log)
        model.metadata["test"] = {k: v["value"] for k, v in evaluation["test"]["metrics"].items()}
    (results_dir / "final_evaluation.json").write_text(_json(_round_floats(evaluation, 5)), encoding="utf-8")
    model.save(Path(out["artifact_path"]))
    log(f"Артефакт: {out['artifact_path']}; результаты: {results_dir}")
    return model


def evaluate_on_test(config, model: ChurnModel, X_test, y_test, ids_test, y_oof, oof, econ, log=print) -> dict:
    results_dir, figures_dir = Path(config["output"]["results_dir"]), Path(config["output"]["figures_dir"])
    log_path = results_dir / "test_access_log.json"
    history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    if history:
        warnings.warn(
            f"Test уже использовался {len(history)} раз(а): оценка — повторный взгляд и может быть оптимистичной.",
            stacklevel=2,
        )
    proba = model.pipeline.predict_proba(X_test)[:, 1]
    result = evaluate_scores(y_test, proba, X_test[SEGMENT].to_numpy(), econ, config, threshold=model.threshold)
    m = result["metrics"]
    log(f"Test: lift@10={m['lift@10']['value']:.2f} [{m['lift@10']['ci_low']:.2f}; {m['lift@10']['ci_high']:.2f}], "
        f"PR-AUC={m['pr_auc']['value']:.4f}, ROC-AUC={m['roc_auc']['value']:.4f}, "
        f"прибыль на клиента {m['profit_per_customer_in_base']['value']:.2f} у.е.")
    result["segments"].round(4).to_csv(results_dir / "segment_metrics_test.csv", index=False)
    pd.DataFrame({"customerID": ids_test, "y_true": y_test, "proba": proba.round(6)}).to_csv(
        results_dir / "test_predictions.csv", index=False)
    plot_test_gains(y_test, proba, y_oof, oof, figures_dir / "test_gains.png")
    history.append({
        "look": len(history) + 1,
        "run_id": model.metadata["run_id"],
        "model": model.model_name,
        "note": "повторный взгляд на test" if history else "первая оценка на test",
    })
    log_path.write_text(_json(history), encoding="utf-8")
    return {
        "look_number": len(history),
        "warning": "test уже видели в предыдущих версиях; оценка немного оптимистична" if len(history) > 1 else None,
        "n": result["n"],
        "churn_rate": result["churn_rate"],
        "metrics": result["metrics"],
        "at_threshold": result["at_threshold"],
        "operating_point": result["operating_point"],
        "segments_note": (
            "сегментные метрики на test — шум (ушедших с двухлетним контрактом в test: "
            f"{int(y_test[X_test[SEGMENT].eq('Two year')].sum())}); смотрите OOF"
        ),
    }


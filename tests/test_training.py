"""Сквозные проверки train.py/predict.py на синтетическом наборе (без сети)."""

import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from churn.config import load_config
from churn.data import FEATURES, clean_data, prepare_features
from churn.inference import ChurnModel, select_for_retention
from churn.training import run_training
from tests.conftest import ROOT


def smoke_config(out_dir, data_path):
    config = load_config(ROOT / "configs/smoke.json")
    config["data"].update({"path": str(data_path), "download": False, "splits_path": str(out_dir / "splits.csv")})
    config["output"] = {
        "results_dir": str(out_dir / "results"),
        "figures_dir": str(out_dir / "figures"),
        "artifact_path": str(out_dir / "model.joblib"),
    }
    return config


def quiet(*_):
    return None


@pytest.fixture(scope="module")
def trained(synthetic_csv, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    config = smoke_config(out, synthetic_csv)
    model = run_training(config, evaluate_test=False, log=quiet)
    return config, model, out


def test_run_without_flag_does_not_touch_test(trained):
    _, model, out = trained
    results = out / "results"
    evaluation = json.loads((results / "final_evaluation.json").read_text(encoding="utf-8"))
    assert isinstance(evaluation["test"], str)
    assert not (results / "test_predictions.csv").exists()
    assert not (results / "test_access_log.json").exists()
    assert "test" not in model.metadata


def test_results_are_machine_readable(trained):
    _, _, out = trained
    results = out / "results"
    for name in ["cv_search.json", "final_evaluation.json", "train_config.json", "model_comparison.json"]:
        json.loads((results / name).read_text(encoding="utf-8"))
    table = pd.read_csv(results / "model_comparison.csv")
    assert table["selected"].sum() == 1
    segments = pd.read_csv(results / "segment_metrics_oof.csv")
    assert set(segments["segment"]) == {"Month-to-month", "One year", "Two year"}
    assert {"lift@10", "lift@10_ci_low", "lift@10_ci_high", "roc_auc"} <= set(segments.columns)
    sens = pd.read_csv(results / "profit_sensitivity.csv")
    assert set(sens["parameter"]) == {"base", "contact_cost", "discount_cost", "retention_rate", "customer_value"}
    evaluation = json.loads((results / "final_evaluation.json").read_text(encoding="utf-8"))
    assert {"python", "numpy", "pandas", "scikit-learn", "catboost"} <= set(evaluation["versions"])
    lift = evaluation["oof_train"]["metrics"]["lift@10"]
    assert lift["ci_low"] <= lift["value"] <= lift["ci_high"]
    for figure in ("gains_profit.png", "calibration.png", "segments_oof.png"):
        assert (out / "figures" / figure).stat().st_size > 0


def test_selection_rule_prefers_baseline_without_significance(trained):
    _, model, out = trained
    comparison = json.loads((out / "results/model_comparison.json").read_text(encoding="utf-8"))
    if not comparison["significantly_better"]:
        assert comparison["chosen"] == comparison["baseline"] == model.model_name


def test_operating_point_within_budget(trained):
    config, model, _ = trained
    point = model.metadata["operating_point"]
    assert point["k_fraction"] <= config["business"]["budget_fraction"] + 1e-9
    assert model.threshold == point["threshold"]


def test_artifact_roundtrip_gives_same_predictions(trained, synthetic_raw):
    _, model, out = trained
    loaded = ChurnModel.load(out / "model.joblib")
    assert loaded.threshold == model.threshold and loaded.features == FEATURES
    np.testing.assert_allclose(loaded.predict_proba(synthetic_raw), model.predict_proba(synthetic_raw))
    np.testing.assert_array_equal(loaded.predict(synthetic_raw), model.predict(synthetic_raw))


def test_load_warns_on_version_mismatch(trained, tmp_path):
    _, model, _ = trained
    model.metadata["versions"] = {**model.metadata["versions"], "scikit-learn": "0.0.1"}
    model.save(tmp_path / "old.joblib")
    with pytest.warns(UserWarning, match="scikit-learn"):
        ChurnModel.load(tmp_path / "old.joblib")


def test_training_is_reproducible(trained, synthetic_csv, tmp_path):
    _, model, out = trained
    again = run_training(smoke_config(tmp_path, synthetic_csv), log=quiet)
    assert again.threshold == model.threshold
    assert again.model_name == model.model_name and again.params == model.params
    first = pd.read_csv(out / "results/oof_predictions.csv")
    second = pd.read_csv(tmp_path / "results/oof_predictions.csv")
    pd.testing.assert_frame_equal(first, second)


def test_evaluate_test_flag_and_access_log(trained, synthetic_csv, synthetic_raw, tmp_path):
    config = smoke_config(tmp_path, synthetic_csv)
    model = run_training(config, evaluate_test=True, log=quiet)
    results = tmp_path / "results"
    evaluation = json.loads((results / "final_evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["test"]["look_number"] == 1 and evaluation["test"]["warning"] is None
    # сохранённые test-предсказания воспроизводятся артефактом
    saved = pd.read_csv(results / "test_predictions.csv")
    records = synthetic_raw.set_index("customerID").loc[saved["customerID"]].reset_index()
    loaded = ChurnModel.load(tmp_path / "model.joblib")
    np.testing.assert_allclose(loaded.predict_proba(records), saved["proba"], atol=1e-6)
    # второй взгляд на test фиксируется в журнале и сопровождается предупреждением
    with pytest.warns(UserWarning, match="Test уже использовался"):
        run_training(config, evaluate_test=True, log=quiet)
    log = json.loads((results / "test_access_log.json").read_text(encoding="utf-8"))
    assert [entry["look"] for entry in log] == [1, 2]
    assert model.metadata["test"]["lift@10"] > 0


def run_cli(*args) -> str:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    done = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True, capture_output=True,
                          text=True, encoding="utf-8", env=env)
    return done.stdout


def test_user_scenario_cli(tmp_path):
    """Сценарий пользователя только через команды: данные → обучение → прогноз CSV → бюджет."""
    data = tmp_path / "telco.csv"
    run_cli("-m", "churn.synthetic", "--output", data)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(smoke_config(tmp_path, data), ensure_ascii=False), encoding="utf-8")
    run_cli("train.py", "--config", config_path)

    # новые клиенты: другой seed, без целевого столбца
    run_cli("-m", "churn.synthetic", "--output", tmp_path / "new_raw.csv", "--rows", 200, "--seed", 1)
    new = pd.read_csv(tmp_path / "new_raw.csv", dtype={"TotalCharges": "object"}).drop(columns=["Churn"])
    new.to_csv(tmp_path / "new.csv", index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # артефакт обучен в этом же окружении — расхождения версий быть не должно
        model = ChurnModel.load(tmp_path / "model.joblib")
    proba = model.predict_proba(new)
    above = proba >= model.threshold
    assert above.sum() >= 2

    run_cli("predict.py", "--model", tmp_path / "model.joblib", "--input", tmp_path / "new.csv",
            "--output", tmp_path / "pred.csv")
    pred = pd.read_csv(tmp_path / "pred.csv")
    assert list(pred.columns) == ["customerID", "churn_probability", "in_retention_list"]
    assert pred["customerID"].tolist() == new["customerID"].tolist()
    np.testing.assert_allclose(pred["churn_probability"], proba, atol=1e-6)
    np.testing.assert_array_equal(pred["in_retention_list"], above.astype(int))

    budget = int(above.sum()) // 2
    stdout = run_cli("predict.py", "--model", tmp_path / "model.joblib", "--input", tmp_path / "new.csv",
                     "--output", tmp_path / "pred_budget.csv", "--budget", budget)
    assert f"бюджет {budget}" in stdout
    chosen = pd.read_csv(tmp_path / "pred_budget.csv")["in_retention_list"].to_numpy().astype(bool)
    assert chosen.sum() == budget
    assert (proba[chosen] >= model.threshold).all()
    assert proba[chosen].min() >= proba[above & ~chosen].max()  # взяты лучшие по оценке


def test_select_for_retention_respects_threshold_and_budget():
    proba = np.array([0.9, 0.2, 0.5, 0.5, 0.7])
    np.testing.assert_array_equal(select_for_retention(proba, 0.5), [1, 0, 1, 1, 1])
    np.testing.assert_array_equal(select_for_retention(proba, 0.5, budget=10), [1, 0, 1, 1, 1])
    np.testing.assert_array_equal(select_for_retention(proba, 0.5, budget=3), [1, 0, 1, 0, 1])  # равные — по порядку
    np.testing.assert_array_equal(select_for_retention(proba, 0.5, budget=0), [0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        select_for_retention(proba, 0.5, budget=-1)


def test_prepare_features_matches_training_cleaning(synthetic_raw):
    from_raw = prepare_features(synthetic_raw)
    from_clean = clean_data(synthetic_raw)[FEATURES]
    pd.testing.assert_frame_equal(from_raw, from_clean)
    pd.testing.assert_frame_equal(prepare_features(from_raw), from_raw)  # повторное применение ничего не меняет


def test_prepare_features_reports_missing_columns(synthetic_raw):
    with pytest.raises(ValueError, match="Contract"):
        prepare_features(synthetic_raw.drop(columns=["Contract"]))


def test_config_validation(tmp_path):
    config = json.loads((ROOT / "configs/train.json").read_text(encoding="utf-8"))
    config["models"]["xgboost"] = {}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="xgboost"):
        load_config(path)


def test_example_csv_is_valid_input():
    records = pd.read_csv(ROOT / "examples/new_customers.csv", dtype={"TotalCharges": "object"})
    assert len(prepare_features(records)) == len(records)

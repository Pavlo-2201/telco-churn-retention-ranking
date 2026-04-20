"""Сквозные проверки train.py/predict.py на маленькой конфигурации configs/smoke.json."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from churn.config import load_config
from churn.data import FEATURES, clean_data, prepare_features
from churn.inference import ChurnModel
from churn.training import fit_and_evaluate, load_split_data, run_training
from tests.conftest import ROOT


def smoke_config(out_dir):
    config = load_config(ROOT / "configs/smoke.json")
    config["output"] = {
        "results_dir": str(out_dir / "results"),
        "figures_dir": str(out_dir / "figures"),
        "artifact_path": str(out_dir / "model.joblib"),
    }
    return config


@pytest.fixture(scope="module")
def trained(raw_df, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(ROOT)
        config = smoke_config(out)
        model = run_training(config, log=lambda *_: None)
    return config, model, out


def test_artifact_roundtrip_gives_same_predictions(trained, raw_df):
    config, model, out = trained
    loaded = ChurnModel.load(out / "model.joblib")
    assert loaded.threshold == model.threshold and loaded.features == FEATURES
    sample = raw_df.sample(300, random_state=0)
    np.testing.assert_allclose(loaded.predict_proba(sample), model.predict_proba(sample))
    np.testing.assert_array_equal(loaded.predict(sample), model.predict(sample))


def test_artifact_reproduces_saved_test_predictions(trained, raw_df):
    _, _, out = trained
    loaded = ChurnModel.load(out / "model.joblib")
    saved = pd.read_csv(out / "results/test_predictions.csv")
    records = raw_df.set_index("customerID").loc[saved["customerID"]].reset_index()
    np.testing.assert_allclose(loaded.predict_proba(records), saved["proba"], atol=1e-6)
    metrics = json.loads((out / "results/final_evaluation.json").read_text())["test"]
    cm = metrics["confusion_matrix"]
    assert (loaded.predict(records) == 1).sum() == cm["tp"] + cm["fp"]


def test_training_is_reproducible(trained, tmp_path, monkeypatch):
    config, model, out = trained
    monkeypatch.chdir(ROOT)
    again = run_training(smoke_config(tmp_path), log=lambda *_: None)
    assert again.threshold == model.threshold
    assert again.model_name == model.model_name and again.params == model.params
    assert again.metadata["test"] == model.metadata["test"]
    first = pd.read_csv(out / "results/oof_predictions.csv")
    second = pd.read_csv(tmp_path / "results/oof_predictions.csv")
    pd.testing.assert_frame_equal(first, second)


def test_threshold_does_not_depend_on_test_labels(trained, monkeypatch):
    config, model, _ = trained
    monkeypatch.chdir(ROOT)
    X, y, ids, masks = load_split_data(config)
    dev, test = masks["train"] | masks["val"], masks["test"]
    y_test_shuffled = pd.Series(np.random.default_rng(0).permutation(y[test].to_numpy()), index=y[test].index)
    a = fit_and_evaluate(config, model.model_name, model.params, X[dev], y[dev], X[test], y[test])
    b = fit_and_evaluate(config, model.model_name, model.params, X[dev], y[dev], X[test], y_test_shuffled)
    assert a["threshold"] == b["threshold"] == model.threshold
    np.testing.assert_allclose(a["oof"], b["oof"])
    np.testing.assert_allclose(a["proba_test"], b["proba_test"])
    assert a["test_metrics"] != b["test_metrics"]


def test_results_are_machine_readable(trained):
    _, _, out = trained
    results = out / "results"
    for name in ["cv_search.json", "final_evaluation.json", "train_config.json"]:
        json.loads((results / name).read_text())
    table = pd.read_csv(results / "model_comparison.csv")
    assert table["selected"].sum() == 1
    assert (table.loc[~table["selected"], "test_roc_auc"] == "не измерено").all()


def test_predict_cli(trained, tmp_path):
    _, _, out = trained
    output = tmp_path / "pred.csv"
    subprocess.run(
        [sys.executable, "predict.py", "--model", str(out / "model.joblib"),
         "--input", "examples/new_customers.csv", "--output", str(output)],
        cwd=ROOT, check=True, capture_output=True,
    )
    pred = pd.read_csv(output)
    examples = pd.read_csv(ROOT / "examples/new_customers.csv")
    assert list(pred.columns) == ["customerID", "churn_probability", "churn_prediction"]
    assert pred["customerID"].tolist() == examples["customerID"].tolist()
    assert pred["churn_probability"].between(0, 1).all()


def test_prepare_features_matches_training_cleaning(raw_df):
    from_raw = prepare_features(raw_df)
    from_clean = clean_data(raw_df)[FEATURES]
    pd.testing.assert_frame_equal(from_raw, from_clean)
    # повторное применение не меняет данные
    pd.testing.assert_frame_equal(prepare_features(from_raw), from_raw)


def test_prepare_features_reports_missing_columns(raw_df):
    with pytest.raises(ValueError, match="Contract"):
        prepare_features(raw_df.drop(columns=["Contract"]))


def test_config_validation(tmp_path):
    config = json.loads((ROOT / "configs/train.json").read_text())
    config["models"]["xgboost"] = {}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="xgboost"):
        load_config(path)

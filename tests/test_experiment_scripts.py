"""Structural regression tests for experiments/scripts/run_baselines.py.

These check the *contract* of build_models()/run_dataset() (right keys,
right classes, no leaked internals) rather than statistical correctness
of the forecasts themselves — the latter is Task 15's job (a full
verification sweep across all 35 datasets). LSTM's epoch count is
overridden to a small value here purely for test speed; every other
hyperparameter comes from the real experiments/configs/hyperparams.yaml.
"""
import yaml

import experiments.scripts.run_baselines as run_baselines
from experiments.scripts.run_baselines import CONFIG_PATH, build_models, run_dataset
from shortseq.datasets.dmart import load_dmart
from shortseq.models.arima import ARIMAForecaster
from shortseq.models.hybrid import HybridForecaster
from shortseq.models.lstm_model import LSTMForecaster
from shortseq.models.naive import SeasonalNaiveForecaster
from shortseq.models.prophet_model import ProphetForecaster
from shortseq.models.sarima import SARIMAForecaster
from shortseq.models.xgboost_model import XGBoostForecaster

EXPECTED_MODEL_CLASSES = {
    "ARIMA": ARIMAForecaster,
    "SARIMA": SARIMAForecaster,
    "Prophet": ProphetForecaster,
    "XGBoost": XGBoostForecaster,
    "LSTM": LSTMForecaster,
    "Hybrid": HybridForecaster,
    "Naive": SeasonalNaiveForecaster,
}


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_build_models_returns_all_seven_baselines_with_real_config():
    config = _load_config()
    models = build_models(config, freq="D", season_length=7)

    assert set(models) == set(EXPECTED_MODEL_CLASSES)
    for model_name, expected_cls in EXPECTED_MODEL_CLASSES.items():
        assert isinstance(models[model_name], expected_cls)


def test_run_dataset_structural_contract_on_real_dmart_food(tmp_path, monkeypatch):
    config = _load_config()
    config["lstm"]["epochs"] = 3  # keep the real config, just cap LSTM's epoch budget for test speed

    # run_dataset() unconditionally writes a `{name}_results.json` under the
    # module-level RESULTS_DIR (it doesn't take an output-dir parameter, unlike
    # the visualization.figures functions in test_figures.py). Redirect it to
    # pytest's tmp_path so this test doesn't leave a results file behind in
    # the real experiments/results/ directory.
    monkeypatch.setattr(run_baselines, "RESULTS_DIR", tmp_path)

    dataset = load_dmart()["dmart_food"]
    output = run_dataset("dmart_food", dataset, config)

    assert (tmp_path / "dmart_food_results.json").exists()

    assert set(output) == {"dataset", "n_train", "n_test", "metrics", "dm_tests", "failed_models"}

    assert set(output["metrics"]) == set(EXPECTED_MODEL_CLASSES)
    for metrics in output["metrics"].values():
        assert "residuals" not in metrics

    assert set(output["dm_tests"]) == set(EXPECTED_MODEL_CLASSES) - {"ARIMA"}
    for dm in output["dm_tests"].values():
        assert set(dm) == {"dm_stat", "p_value", "significant_bonferroni"}

    # Every DM-tested model must have a corresponding metrics entry: a model
    # whose predictions diverge (e.g. NaN) should be excluded from dm_tests
    # exactly like it's excluded from metrics, never show up in one but not
    # the other (the bug this test is designed to catch: preds_by_model used
    # to be populated before compute_metrics could raise, so a NaN-producing
    # model could get a dm_tests entry with no metrics entry at all).
    assert set(output["dm_tests"]) <= set(output["metrics"]) - {"ARIMA"}

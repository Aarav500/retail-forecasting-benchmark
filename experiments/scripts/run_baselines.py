"""CLI to run all baseline models against every implemented ShortSeq dataset.

Generalizes `code/real_experiment.py` and `code/m4_experiment.py`'s role
of "run the baseline set against a dataset and save results" across
every dataset in the registry (D-Mart, UCI, Walmart, M5, M4), using the
ported BaseForecaster models. The M4-specific naive/AR1/seasonal-naive
win-count micro-analysis from the original `m4_experiment.py` is
superseded here by running the full baseline set (including the new
SeasonalNaiveForecaster) uniformly, consistent with the benchmark's
uniform evaluation protocol.

Fixes a bug present in `code/experiment.py:run_dataset_experiment`: the
original DM test call passed each model's *residuals, reversed* as the
comparison series instead of its actual predictions. This version passes
each model's real predictions.

Also applies Bonferroni correction (`shortseq.evaluation.dm_test.bonferroni_correct`)
across each dataset's family of pairwise DM tests (every non-ARIMA model
vs. ARIMA, on that one dataset) — see the comment at the correction call
site for why datasets are treated as separate families rather than
correcting globally across all datasets.
"""
import argparse
import json
from pathlib import Path

import yaml

from shortseq.datasets.registry import load_all
from shortseq.evaluation.dm_test import bonferroni_correct, diebold_mariano_test
from shortseq.evaluation.metrics import compute_metrics
from shortseq.models.arima import ARIMAForecaster
from shortseq.models.hybrid import HybridForecaster
from shortseq.models.lstm_model import LSTMForecaster
from shortseq.models.naive import SeasonalNaiveForecaster
from shortseq.models.prophet_model import ProphetForecaster
from shortseq.models.sarima import SARIMAForecaster
from shortseq.models.xgboost_model import XGBoostForecaster

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "hyperparams.yaml"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

SEASON_LENGTH_BY_FREQ = {"D": 7, "W": 52, "M": 12}


def build_models(config: dict, freq: str, season_length: int) -> dict:
    return {
        "ARIMA": ARIMAForecaster(max_p=config["arima"]["max_p"], max_q=config["arima"]["max_q"]),
        "SARIMA": SARIMAForecaster(
            max_p=config["sarima"]["max_p"], max_q=config["sarima"]["max_q"],
            max_P=config["sarima"]["max_P"], max_Q=config["sarima"]["max_Q"],
        ),
        "Prophet": ProphetForecaster(
            freq=freq, yearly_seasonality=config["prophet"]["yearly_seasonality"],
            weekly_seasonality=config["prophet"]["weekly_seasonality"],
            daily_seasonality=config["prophet"]["daily_seasonality"],
            seasonality_mode=config["prophet"]["seasonality_mode"],
            interval_width=config["prophet"]["interval_width"],
        ),
        "XGBoost": XGBoostForecaster(
            lags=config["xgboost"]["lag_window"], n_estimators=config["xgboost"]["n_estimators"],
            max_depth=config["xgboost"]["max_depth"], learning_rate=config["xgboost"]["learning_rate"],
            subsample=config["xgboost"]["subsample"], colsample_bytree=config["xgboost"]["colsample_bytree"],
            random_state=config["xgboost"]["random_state"],
        ),
        "LSTM": LSTMForecaster(
            lags=config["lstm"]["lag_window"], units_layer1=config["lstm"]["units_layer1"],
            units_layer2=config["lstm"]["units_layer2"], dropout=config["lstm"]["dropout"],
            epochs=config["lstm"]["epochs"], batch_size=config["lstm"]["batch_size"],
            patience=config["lstm"]["patience"], validation_split=config["lstm"]["validation_split"],
            random_seed=config["lstm"]["random_seed"],
        ),
        "Hybrid": HybridForecaster(
            lags=config["hybrid"]["xgboost_residual"]["lag_window"],
            arima_max_p=config["hybrid"]["arima"]["max_p"], arima_max_q=config["hybrid"]["arima"]["max_q"],
            xgb_n_estimators=config["hybrid"]["xgboost_residual"]["n_estimators"],
            xgb_max_depth=config["hybrid"]["xgboost_residual"]["max_depth"],
            xgb_learning_rate=config["hybrid"]["xgboost_residual"]["learning_rate"],
            random_state=config["hybrid"]["xgboost_residual"]["random_state"],
        ),
        "Naive": SeasonalNaiveForecaster(season_length=season_length),
    }


def run_dataset(name: str, dataset, config: dict, split: float = None) -> dict:
    if split is None:
        split = config["evaluation"]["train_test_split"]

    series = dataset.series
    split_idx = int(len(series) * split)
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]

    season_length = SEASON_LENGTH_BY_FREQ.get(dataset.freq, 7)
    models = build_models(config, dataset.freq, season_length)

    results = {}
    preds_by_model = {}
    failed_models = {}
    for model_name, model in models.items():
        try:
            model.fit(train)
            forecast = model.predict_rolling(test)
            preds = forecast.point[: len(test)]
            # compute_metrics is called BEFORE preds_by_model is populated: if a
            # model produces non-finite predictions, compute_metrics (sklearn)
            # raises here and the except block below keeps it out of both
            # `results` and `preds_by_model` — otherwise it would silently
            # end up in the DM-test loop (which iterates over preds_by_model)
            # with no corresponding `metrics` entry, surfacing as a bogus
            # "not significant" DM result instead of a visible failure.
            metrics = compute_metrics(
                test.values, preds, model.name, model.train_time_, model.pred_time_
            )
            preds_by_model[model_name] = preds
            results[model_name] = metrics
            print(f"  {model_name}: RMSE={results[model_name]['rmse']:.2f}")
        except Exception as exc:
            failed_models[model_name] = str(exc)
            print(f"  [{name}/{model_name}] failed: {exc}")

    dm_tests = {}
    if "ARIMA" in preds_by_model:
        arima_preds = preds_by_model["ARIMA"]
        dm_h = config["evaluation"]["dm_test_h"]
        for model_name, preds in preds_by_model.items():
            if model_name == "ARIMA":
                continue
            dm_stat, p_val = diebold_mariano_test(test.values, arima_preds, preds, h=dm_h)
            dm_tests[model_name] = {"dm_stat": dm_stat, "p_value": p_val}

        # Bonferroni-correct across this dataset's family of pairwise DM tests
        # (every non-ARIMA model vs. ARIMA, on this one dataset). Each dataset
        # is treated as its own family since the comparisons within it share
        # the same baseline and the same test set; correcting across datasets
        # too would be double-counting when this dict is combined by later
        # aggregate analysis (Task 11's ranking).
        p_values = {model_name: dm["p_value"] for model_name, dm in dm_tests.items()}
        alpha = config["evaluation"]["significance_levels"][1]  # the 0.05 entry
        corrected = bonferroni_correct(p_values, alpha=alpha)
        for model_name, dm in dm_tests.items():
            dm["significant_bonferroni"] = corrected[model_name]["significant"]

    output = {
        "dataset": name,
        "n_train": split_idx,
        "n_test": len(test),
        "metrics": {k: {mk: mv for mk, mv in v.items() if mk != "residuals"} for k, v in results.items()},
        "dm_tests": dm_tests,
        "failed_models": failed_models,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}_results.json", "w") as f:
        json.dump(output, f, indent=2)
    return output


def main(dataset_names: list[str] = None):
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    datasets = load_all()
    if dataset_names:
        datasets = {k: v for k, v in datasets.items() if k in dataset_names}
    print(f"Running baselines on {len(datasets)} datasets...")
    for name, dataset in datasets.items():
        print(f"\n{'=' * 60}\n{name} | n={dataset.n} | freq={dataset.freq}\n{'=' * 60}")
        run_dataset(name, dataset, config)
    print(f"\nDone. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+",
        help="Only run these dataset names (e.g. dmart_food walmart), for smoke-testing "
             "before committing to the full ~30-60 minute, 35-dataset sweep",
    )
    args = parser.parse_args()
    main(dataset_names=args.datasets)

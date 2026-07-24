"""CLI to run ChronosForecaster against every implemented ShortSeq dataset
and compare it to ARIMA via the same metrics/DM-test protocol
run_baselines.py uses for the classical baselines.

Kept separate from run_baselines.py so the CPU-only orchestration script
never needs torch/chronos-forecasting as a dependency — this script only
runs on the GPU instance, where those are installed (see
requirements-gpu.txt).
"""
import argparse
import json
from pathlib import Path

from shortseq.datasets.registry import load_all
from shortseq.evaluation.dm_test import bonferroni_correct, diebold_mariano_test
from shortseq.evaluation.metrics import compute_metrics
from shortseq.models.arima import ARIMAForecaster
from shortseq.models.foundation.chronos import ChronosForecaster

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "foundation"


def run_dataset(name: str, dataset, size: str, split: float = 0.8) -> dict:
    series = dataset.series
    split_idx = int(len(series) * split)
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]

    results = {}
    preds_by_model = {}
    failed_models = {}

    for model_name, model in [("ARIMA", ARIMAForecaster()), (f"Chronos-{size}", ChronosForecaster(size=size))]:
        try:
            model.fit(train)
            forecast = model.predict_rolling(test)
            preds = forecast.point[: len(test)]
            metrics = compute_metrics(test.values, preds, model.name, model.train_time_, model.pred_time_)
            preds_by_model[model_name] = preds
            results[model_name] = metrics
            print(f"  {model_name}: RMSE={metrics['rmse']:.2f}")
        except Exception as exc:
            failed_models[model_name] = str(exc)
            print(f"  [{name}/{model_name}] failed: {exc}")

    dm_tests = {}
    if "ARIMA" in preds_by_model and f"Chronos-{size}" in preds_by_model:
        dm_stat, p_val = diebold_mariano_test(test.values, preds_by_model["ARIMA"], preds_by_model[f"Chronos-{size}"])
        dm_tests[f"Chronos-{size}"] = {"dm_stat": dm_stat, "p_value": p_val}
        corrected = bonferroni_correct({f"Chronos-{size}": p_val})
        dm_tests[f"Chronos-{size}"]["significant_bonferroni"] = corrected[f"Chronos-{size}"]["significant"]

    output = {
        "dataset": name,
        "n_train": split_idx,
        "n_test": len(test),
        "metrics": {k: {mk: mv for mk, mv in v.items() if mk != "residuals"} for k, v in results.items()},
        "dm_tests": dm_tests,
        "failed_models": failed_models,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}_chronos_{size}_results.json", "w") as f:
        json.dump(output, f, indent=2)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", choices=["tiny", "mini", "small", "base", "large"])
    parser.add_argument("--datasets", nargs="+", help="Only run these dataset names")
    args = parser.parse_args()

    datasets = load_all()
    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    print(f"Running ARIMA vs Chronos-{args.size} on {len(datasets)} datasets...")
    for name, dataset in datasets.items():
        print(f"\n{'=' * 60}\n{name} | n={dataset.n} | freq={dataset.freq}\n{'=' * 60}")
        run_dataset(name, dataset, args.size)
    print(f"\nDone. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

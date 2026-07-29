"""CLI to run ChronosForecaster against every implemented ShortSeq dataset
and compare it to ARIMA via the same metrics/DM-test protocol
run_baselines.py uses for the classical baselines.

Kept separate from run_baselines.py so the CPU-only orchestration script
never needs torch/chronos-forecasting as a dependency — this script only
runs on the GPU instance, where those are installed (see
requirements-gpu.txt).

Per-timestep residuals are stripped from the output by default. Diebold-
Mariano testing needs them (a DM test compares two error series point-by-
point; RMSE cannot reconstruct them), so pass --keep-residuals when the
run is intended to support DM analysis. NOTE: the committed results under
experiments/results/foundation/ were produced WITHOUT it, so post-hoc DM
tests on that corpus require re-running the sweep on a GPU instance — the
`dm_tests` block written inline below is computed from live predictions
and does not survive into the file as a reusable error series. Rationale:
docs/superpowers/specs/2026-07-29-friedman-nemenyi-ranking-design.md,
"Why not DM tests".
"""
import argparse
from pathlib import Path

from shortseq.datasets.registry import load_all
from shortseq.evaluation.dm_test import bonferroni_correct, diebold_mariano_test
from shortseq.evaluation.metrics import compute_metrics, without_residuals
from shortseq.models.arima import ARIMAForecaster
from shortseq.models.foundation.chronos import ChronosForecaster
from shortseq.results_io import ResultWriteError, print_write_failures, write_result_json

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "foundation"


def run_dataset(name: str, dataset, size: str, split: float = 0.8,
                keep_residuals: bool = False) -> dict:
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
        "metrics": {
            k: (dict(v) if keep_residuals else without_residuals(v))
            for k, v in results.items()
        },
        "dm_tests": dm_tests,
        "failed_models": failed_models,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Raises ResultWriteError (naming the offending metric) rather than
    # writing a bare `NaN` literal, which is invalid strict JSON. main()
    # catches it per dataset so one unwritable payload does not abort the
    # sweep -- this script has no skip-existing, so an abort loses the run.
    write_result_json(RESULTS_DIR / f"{name}_chronos_{size}_results.json", output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", choices=["tiny", "mini", "small", "base", "large"])
    parser.add_argument("--datasets", nargs="+", help="Only run these dataset names")
    parser.add_argument(
        "--keep-residuals", action="store_true",
        help="Retain per-timestep residuals in the output JSON. Needed for "
             "Diebold-Mariano tests; off by default because residuals "
             "inflate result files roughly 7-10x (measured: the committed "
             "1,840-file scaling sweep would grow from ~0.8 MB to ~8 MB, and "
             "results are git-tracked, so that growth is permanent in history).",
    )
    args = parser.parse_args()

    datasets = load_all()
    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    print(f"Running ARIMA vs Chronos-{args.size} on {len(datasets)} datasets...")
    # Mirrors run_dataset's own `failed_models` convention one level up: a
    # dataset whose result cannot be serialised is recorded and skipped, not
    # allowed to kill the sweep. Like run_baselines.py this script re-runs
    # every dataset on every invocation, so an abort part-way discards every
    # dataset already completed -- on GPU time.
    unwritten = {}
    for name, dataset in datasets.items():
        print(f"\n{'=' * 60}\n{name} | n={dataset.n} | freq={dataset.freq}\n{'=' * 60}")
        try:
            run_dataset(name, dataset, args.size, keep_residuals=args.keep_residuals)
        except ResultWriteError as exc:
            unwritten[name] = exc
            print(f"  [{name}] {exc}")
    print(f"\nDone. Results saved to {RESULTS_DIR}")
    # Last, so it cannot scroll past.
    print_write_failures(unwritten)


if __name__ == "__main__":
    main()

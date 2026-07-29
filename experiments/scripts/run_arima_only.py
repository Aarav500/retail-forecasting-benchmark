"""Backfill ARIMA-only baselines for datasets that lack them.

`run_baselines.py` runs all seven classical models with no way to select a
subset. That is prohibitively slow on the long Phase B series: SARIMA's
seasonal `auto_arima` search and LSTM's rolling forecast both scale badly
past n~1000 (Favorita n=1684, Rossmann n=942), turning a backfill into a
multi-day job.

The size-scaling campaign only needs ARIMA as its reference, so this
script runs ARIMA alone and writes results in the same schema
run_baselines.py uses. Files are written incrementally and existing ones
are skipped, so the job is resumable.

The output deliberately records ONLY ARIMA under "metrics". A later job
can add the remaining classical baselines; because this writes the same
schema, that job can merge into these files rather than replacing them.

Per-timestep residuals are stripped from the output by default. Diebold-
Mariano testing needs them (a DM test compares two error series point-by-
point; RMSE cannot reconstruct them), so pass --keep-residuals when the
run is intended to support DM analysis. NOTE: the committed results under
experiments/results/ were produced WITHOUT it, so DM tests against this
ARIMA reference require re-running the backfill. Rationale:
docs/superpowers/specs/2026-07-29-friedman-nemenyi-ranking-design.md,
"Why not DM tests".
"""
import argparse
from pathlib import Path

from shortseq.datasets.registry import load_all
from shortseq.evaluation.metrics import compute_metrics, without_residuals
from shortseq.models.arima import ARIMAForecaster
from shortseq.results_io import ResultWriteError, print_write_failures, write_result_json

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"


def run_dataset(name: str, dataset, split: float = 0.8,
                keep_residuals: bool = False) -> dict:
    series = dataset.series
    split_idx = int(len(series) * split)
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]

    metrics = {}
    failed_models = {}
    try:
        model = ARIMAForecaster()
        model.fit(train)
        forecast = model.predict_rolling(test)
        preds = forecast.point[: len(test)]
        m = compute_metrics(
            test.values, preds, model.name, model.train_time_, model.pred_time_
        )
        metrics["ARIMA"] = dict(m) if keep_residuals else without_residuals(m)
        print(f"  ARIMA: RMSE={m['rmse']:.2f}", flush=True)
    except Exception as exc:
        failed_models["ARIMA"] = f"{type(exc).__name__}: {exc}"
        print(f"  [{name}/ARIMA] FAILED: {type(exc).__name__}: {exc}", flush=True)

    output = {
        "dataset": name,
        "n_train": split_idx,
        "n_test": len(test),
        "metrics": metrics,
        "dm_tests": {},
        "failed_models": failed_models,
        "partial": "ARIMA-only backfill; other classical baselines not run",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Raises ResultWriteError (naming the offending metric) rather than
    # writing a bare `NaN` literal, which is invalid strict JSON. main()
    # catches it per dataset so one unwritable payload does not abort the
    # backfill.
    write_result_json(RESULTS_DIR / f"{name}_results.json", output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if a result file already exists")
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
    todo = [
        (name, ds) for name, ds in datasets.items()
        if args.force or not (RESULTS_DIR / f"{name}_results.json").exists()
    ]
    # Shortest series first. ARIMA's rolling forecast does one sequential
    # update+predict per test point, so cost grows with n -- a single
    # n=2296 series can take minutes while a short one takes seconds.
    # Ordering short-first means the short stratum (n<200), which is what
    # the size-scaling campaign's headline claim rests on, completes early;
    # the expensive long tail then becomes optional rather than blocking.
    todo.sort(key=lambda pair: pair[1].n)
    print(f"ARIMA-only backfill: {len(todo)} of {len(datasets)} datasets need results "
          f"(shortest first; n range {todo[0][1].n}-{todo[-1][1].n})" if todo
          else "ARIMA-only backfill: nothing to do", flush=True)

    # Mirrors run_dataset's own `failed_models` convention one level up: a
    # dataset whose result cannot be serialised is recorded and skipped, not
    # allowed to kill the backfill.
    unwritten = {}
    for i, (name, ds) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {name} | n={ds.n} | freq={ds.freq}", flush=True)
        try:
            run_dataset(name, ds, keep_residuals=args.keep_residuals)
        except ResultWriteError as exc:
            unwritten[name] = exc
            print(f"  [{name}] {exc}", flush=True)

    print(f"\nDone. Results in {RESULTS_DIR}", flush=True)
    # Last, so it cannot scroll past. It matters here even though the job is
    # resumable: these datasets stay in `todo` on the next invocation and
    # will fail again, so the operator needs to know why now.
    print_write_failures(unwritten)


if __name__ == "__main__":
    main()

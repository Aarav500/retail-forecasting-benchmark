"""Chronos model-size scaling campaign.

Runs all five Chronos sizes (tiny 8M -> large 710M) across every
registry series and compares each to the ARIMA baseline already
committed under experiments/results/.

Deliberately does NOT re-run ARIMA: run_chronos.py re-fits ARIMA
alongside Chronos on every invocation, so sweeping five sizes through
it would fit ARIMA five times per series and could yield five slightly
different reference numbers, muddying the very comparison this campaign
exists to make. ARIMA's RMSE and predictions come from the committed
baseline results instead.

Only runs on a GPU instance (imports torch via ChronosForecaster); see
requirements-gpu.txt.

Per-timestep residuals are stripped from the output by default. Diebold-
Mariano testing needs them (a DM test compares two error series point-by-
point; RMSE cannot reconstruct them), so pass --keep-residuals when the
run is intended to support DM analysis. NOTE: the committed results under
experiments/results/scaling/ were produced WITHOUT it, so DM tests on that
corpus require re-running the sweep on a GPU instance. Rationale:
docs/superpowers/specs/2026-07-29-friedman-nemenyi-ranking-design.md,
"Why not DM tests".
"""
import argparse
import json
from pathlib import Path

from shortseq.constants import CHRONOS_SIZES
from shortseq.datasets.registry import load_all
from shortseq.evaluation.metrics import compute_metrics, without_residuals
from shortseq.results_io import ResultWriteError, print_write_failures, write_result_json

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "experiments" / "results"
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "scaling"

# Re-exported under this module's long-standing public name. The canonical
# list lives in shortseq.constants so run_ranking.py can read it without
# importing this module - and with it the whole modelling stack it needs
# for the sweep but that a data-only ranking script does not.
SIZES = CHRONOS_SIZES
SHORT_MAX_N = 200  # series with n < SHORT_MAX_N are the "short" stratum


def load_arima_rmse(dataset_name: str) -> float | None:
    """ARIMA's RMSE for one dataset, from the committed baseline results.

    Returns None if the baseline file is missing or ARIMA failed there —
    the caller records that rather than silently treating it as a win.
    """
    path = BASELINE_DIR / f"{dataset_name}_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("metrics", {}).get("ARIMA", {}).get("rmse")


def result_path(dataset_name: str, size: str) -> Path:
    return RESULTS_DIR / f"{dataset_name}_chronos_{size}.json"


def run_one(dataset_name: str, dataset, size: str, split: float = 0.8,
            keep_residuals: bool = False) -> dict:
    """Run one (dataset, size) pair. Returns the result dict it wrote."""
    # Imported lazily so the pure-Python summary/stratification helpers in
    # this module can be imported and tested on a CPU-only machine, where
    # torch/chronos-forecasting are deliberately absent.
    from shortseq.models.foundation.chronos import ChronosForecaster

    series = dataset.series
    split_idx = int(len(series) * split)
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]

    model_key = f"Chronos-{size}"
    metrics = {}
    failed_models = {}

    try:
        model = ChronosForecaster(size=size)
        model.fit(train)
        forecast = model.predict_rolling(test)
        preds = forecast.point[: len(test)]
        m = compute_metrics(
            test.values, preds, model.name, model.train_time_, model.pred_time_
        )
        metrics[model_key] = dict(m) if keep_residuals else without_residuals(m)
        print(f"  {model_key}: RMSE={m['rmse']:.2f}")
    except Exception as exc:
        failed_models[model_key] = f"{type(exc).__name__}: {exc}"
        print(f"  [{dataset_name}/{model_key}] FAILED: {type(exc).__name__}: {exc}")

    arima_rmse = load_arima_rmse(dataset_name)
    output = {
        "dataset": dataset_name,
        "size": size,
        "n": dataset.n,
        "stratum": "short" if dataset.n < SHORT_MAX_N else "long",
        "n_train": split_idx,
        "n_test": len(test),
        "arima_rmse": arima_rmse,
        "metrics": metrics,
        "failed_models": failed_models,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Raises ResultWriteError (naming the offending metric) rather than
    # writing a bare `NaN` literal, which is invalid strict JSON. main()
    # catches it per run so one unwritable payload does not abort the sweep.
    write_result_json(result_path(dataset_name, size), output)
    return output


def summarize() -> dict:
    """Stratified summary over whatever result files exist.

    Reports short (n < 200) and long (n >= 200) strata SEPARATELY and
    never emits a pooled figure — the registry is 199/169, so a pooled
    mean would average two regimes with opposite expected outcomes and
    hide the effect this campaign measures. See the Phase B spec's
    stratification addendum.
    """
    per_size = {}
    for path in sorted(RESULTS_DIR.glob("*_chronos_*.json")):
        with open(path) as f:
            d = json.load(f)
        size = d["size"]
        key = f"Chronos-{size}"
        rmse = d.get("metrics", {}).get(key, {}).get("rmse")
        arima = d.get("arima_rmse")
        bucket = per_size.setdefault(
            size, {"short": [], "long": [], "failed": 0, "no_baseline": 0}
        )
        if rmse is None:
            bucket["failed"] += 1
            continue
        if arima is None or arima == 0:
            bucket["no_baseline"] += 1
            continue
        bucket[d["stratum"]].append(rmse / arima)

    summary = {}
    for size in SIZES:
        b = per_size.get(size)
        if not b:
            continue
        entry = {"failed": b["failed"], "no_baseline": b["no_baseline"]}
        for stratum in ("short", "long"):
            ratios = b[stratum]
            if ratios:
                wins = sum(1 for r in ratios if r < 1.0)
                entry[stratum] = {
                    "n_series": len(ratios),
                    "mean_ratio": round(sum(ratios) / len(ratios), 4),
                    "median_ratio": round(sorted(ratios)[len(ratios) // 2], 4),
                    "chronos_wins": wins,
                    "arima_wins": len(ratios) - wins,
                }
            else:
                entry[stratum] = None
        summary[size] = entry
    return summary


def print_summary(summary: dict) -> None:
    print(f"\n{'=' * 70}")
    print("SIZE-SCALING SUMMARY (Chronos RMSE / ARIMA RMSE; <1.0 = Chronos better)")
    print("Strata reported separately by design — never pooled.")
    print(f"{'=' * 70}")
    for stratum in ("short", "long"):
        label = "SHORT (n<200)" if stratum == "short" else "LONG (n>=200)"
        print(f"\n{label}")
        print(f"{'size':<8}{'series':>8}{'mean':>10}{'median':>10}{'C wins':>9}{'A wins':>9}")
        for size in SIZES:
            s = summary.get(size, {}).get(stratum)
            if not s:
                continue
            print(
                f"{size:<8}{s['n_series']:>8}{s['mean_ratio']:>10.4f}"
                f"{s['median_ratio']:>10.4f}{s['chronos_wins']:>9}{s['arima_wins']:>9}"
            )
    print("\nfailures / missing-baseline per size:")
    for size in SIZES:
        e = summary.get(size)
        if e:
            print(f"  {size:<8} failed={e['failed']:<5} no_baseline={e['no_baseline']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=SIZES, choices=SIZES)
    parser.add_argument("--datasets", nargs="+", help="Only these dataset names")
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if a result file exists"
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="Just print the summary and exit"
    )
    parser.add_argument(
        "--keep-residuals", action="store_true",
        help="Retain per-timestep residuals in the output JSON. Needed for "
             "Diebold-Mariano tests; off by default because residuals "
             "inflate result files roughly 7-10x (measured: the committed "
             "1,840-file scaling sweep would grow from ~0.8 MB to ~8 MB, and "
             "results are git-tracked, so that growth is permanent in history).",
    )
    args = parser.parse_args()

    if args.summary_only:
        print_summary(summarize())
        return

    datasets = load_all()
    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    total = len(datasets) * len(args.sizes)
    print(f"Size-scaling sweep: {len(datasets)} datasets x {len(args.sizes)} sizes = {total} runs")

    done = 0
    # Same shape as the per-size `failed` counting in summarize(): a run
    # whose result cannot be serialised is recorded and skipped rather than
    # aborting the remaining runs. This sweep IS resumable, so an abort
    # costs only the current file -- but there is no reason to make the
    # operator re-invoke, and the diagnostic below is worth having either way.
    write_failures = {}
    for size in args.sizes:
        for name, dataset in datasets.items():
            done += 1
            if not args.force and result_path(name, size).exists():
                print(f"[{done}/{total}] {name} @ {size}: cached, skipping")
                continue
            print(f"[{done}/{total}] {name} | n={dataset.n} | size={size}")
            try:
                run_one(name, dataset, size, keep_residuals=args.keep_residuals)
            except ResultWriteError as exc:
                write_failures[f"{name} @ {size}"] = exc
                print(f"  [{name}/{size}] {exc}")

    print_summary(summarize())
    print(f"\nDone. Results in {RESULTS_DIR}")
    # After print_summary, and last: summarize() reads the files on disk, so
    # a run that never produced one is invisible to it -- neither counted in
    # `failed` nor in any stratum. This block is the only place it shows up.
    print_write_failures(write_failures)


if __name__ == "__main__":
    main()

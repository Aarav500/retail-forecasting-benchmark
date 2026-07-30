"""Training-window-size ablation on D-Mart Food, using the ported baselines.

Ported from `code/ablation.py`.

This is the one result-writing script that produces a failure marker BY
DESIGN: a model that raises has no RMSE for that window, and the run must
still report the models that succeeded on the same window. So it does NOT
route through `shortseq.results_io.write_result_json` like the sweeps do
-- that refuses to write anything when the payload is not strict JSON,
which here would discard every model that worked. The marker is JSON
`null` instead: valid strict JSON, and the failure stays visible, where a
bare `NaN` literal is unreadable to any strict consumer and a 0.0 would
read as a perfect forecast.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from shortseq.datasets.dmart import load_dmart
from shortseq.models.arima import ARIMAForecaster
from shortseq.models.hybrid import HybridForecaster
from shortseq.models.lstm_model import LSTMForecaster
from shortseq.models.xgboost_model import XGBoostForecaster

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"


def run_ablation(include_lstm: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = load_dmart()["dmart_food"].series
    test_n = 37
    train_sizes = [50, 80, 110, 144]

    model_builders = {
        "ARIMA": ARIMAForecaster,
        "XGBoost": XGBoostForecaster,
        "Hybrid": HybridForecaster,
    }
    if include_lstm:
        model_builders["LSTM"] = LSTMForecaster

    results = {}
    print("=== Ablation: Training Window Size (D-Mart Food) ===")
    print(f"{'n_train':<10}" + "".join(f"{m:<12}" for m in model_builders))
    print("-" * (10 + 12 * len(model_builders)))

    for n_train in train_sizes:
        train = series.iloc[:n_train]
        test = series.iloc[n_train:n_train + test_n]
        row, row_str = {"n_train": n_train}, f"{n_train:<10}"
        for mname, builder in model_builders.items():
            try:
                model = builder().fit(train)
                preds = model.predict_rolling(test).point
                rmse = float(np.sqrt(mean_squared_error(test.values, preds[:len(test)])))
            except Exception as exc:
                print(f"  [{mname} n={n_train}] Error: {exc}")
                rmse = None
            if rmse is not None and not math.isfinite(rmse):
                # Nothing raised, but the number is meaningless. Reachable
                # without an exception: `mean_squared_error` validates its
                # inputs for finiteness, so a diverging model's predictions
                # pass at 1e200 and then overflow to inf when squared.
                # Distinct wording from the `except` branch above, because
                # "Error" would send the operator looking for a traceback
                # that does not exist.
                print(f"  [{mname} n={n_train}] Non-finite RMSE ({rmse}); recording as failure.")
                rmse = None
            # `round(None, 4)` and `f"{None:<12.2f}"` are both TypeErrors, so
            # the failure case needs its own value and its own cell -- padded
            # to a number cell's width so the table still lines up.
            row[mname] = None if rmse is None else round(rmse, 4)
            row_str += f"{'FAIL':<12}" if rmse is None else f"{rmse:<12.2f}"
        results[n_train] = row
        print(row_str)

    out_path = RESULTS_DIR / "ablation_results.json"
    # `results` is keyed by n_train, an int, and json.dumps stringifies int
    # keys silently -- a consumer gets "50", not 50.
    #
    # allow_nan=False is an assertion here, not a filter: every failure is
    # already `None` by this point, so a non-finite value reaching this call
    # means some other path produced one, and failing loudly beats writing a
    # bare NaN literal no strict parser will read back.
    #
    # Serialised before the file is opened, because `open(..., "w")`
    # truncates on entry and the encoder streams as it walks: a rejected
    # payload would otherwise leave a partial file where the previous run's
    # ablation sat. This is deliberately NOT a switch to write_result_json --
    # a failed model is an expected outcome here and must still produce a
    # file, recorded as `null` above.
    encoded = json.dumps(results, indent=2, allow_nan=False)
    with open(out_path, "w") as f:
        f.write(encoded)
    print(f"\nSaved -> {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm", action="store_true")
    args = parser.parse_args()
    run_ablation(include_lstm=args.lstm)

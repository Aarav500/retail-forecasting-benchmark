"""Training-window-size ablation on D-Mart Food, using the ported baselines.

Ported from `code/ablation.py`.
"""
import argparse
import json
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
                rmse = float("nan")
            row[mname] = round(rmse, 4)
            row_str += f"{rmse:<12.2f}"
        results[n_train] = row
        print(row_str)

    out_path = RESULTS_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm", action="store_true")
    args = parser.parse_args()
    run_ablation(include_lstm=args.lstm)

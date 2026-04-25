"""
ablation.py — Training window size ablation study.

Evaluates ARIMA, XGBoost, LSTM, and Hybrid across four training window
sizes with a fixed test set on the D-Mart Food category.

Usage:
    python code/ablation.py
    python code/ablation.py --lstm   # include LSTM (~80s extra)
"""
import warnings; warnings.filterwarnings('ignore')
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
from experiment import run_arima, run_xgboost, run_lstm, run_hybrid

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_ablation(include_lstm=False):
    RESULTS_DIR.mkdir(exist_ok=True)
    df     = pd.read_csv(DATA_DIR / "real_food.csv", parse_dates=["ds"])
    series = df["y"].values.astype(float)
    test_n = 37
    train_sizes = [50, 80, 110, 144]

    model_fns = {
        "ARIMA":   lambda tr, te: run_arima(tr, te)[0],
        "XGBoost": lambda tr, te: run_xgboost(tr, te)[0],
        "Hybrid":  lambda tr, te: run_hybrid(tr, te)[0],
    }
    if include_lstm:
        model_fns["LSTM"] = lambda tr, te: run_lstm(tr, te)[0]

    results = {}
    print("=== Ablation: Training Window Size (D-Mart Food) ===")
    print(f"{'n_train':<10}" + "".join(f"{m:<12}" for m in model_fns))
    print("-" * (10 + 12 * len(model_fns)))

    for n_train in train_sizes:
        train = series[:n_train]
        test  = series[n_train: n_train + test_n]
        row, row_str = {"n_train": n_train}, f"{n_train:<10}"
        for mname, fn in model_fns.items():
            try:
                preds = fn(train, test)
                rmse  = np.sqrt(mean_squared_error(test, preds[:len(test)]))
            except Exception as e:
                print(f"  [{mname} n={n_train}] Error: {e}")
                rmse = float("nan")
            row[mname] = round(rmse, 4)
            row_str += f"{rmse:<12.2f}"
        results[n_train] = row
        print(row_str)

    out_path = RESULTS_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm", action="store_true")
    args = parser.parse_args()
    run_ablation(include_lstm=args.lstm)

"""Point-forecast accuracy metrics.

`compute_metrics` is ported from `code/experiment.py:compute_metrics`.
`crps`/`coverage` are real stubs: no currently implemented model produces
a predictive distribution, so there is nothing valid to compute yet —
these are planned for the foundation-model phase of the NeurIPS
execution plan.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(actual, predicted, model_name: str, train_time: float, pred_time: float) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    residuals = actual - predicted
    return {
        "model": model_name,
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "mape": round(float(mape), 4),
        "mean_residual": round(float(np.mean(residuals)), 4),
        "residual_std": round(float(np.std(residuals)), 4),
        "train_time": round(float(train_time), 4),
        "pred_time": round(float(pred_time), 4),
        "residuals": residuals.tolist(),
    }


def crps(actual, dist) -> float:
    raise NotImplementedError(
        "CRPS requires a predictive distribution, which no currently "
        "implemented model produces. Planned for the foundation-model "
        "phase of the NeurIPS execution plan."
    )


def coverage(actual, dist, level: float) -> float:
    raise NotImplementedError(
        "Prediction-interval coverage requires a predictive distribution, "
        "which no currently implemented model produces. Planned for the "
        "foundation-model phase of the NeurIPS execution plan."
    )

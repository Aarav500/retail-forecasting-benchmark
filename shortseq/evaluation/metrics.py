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
    """Note: `mape` is NaN when every value in `actual` is zero (e.g. an all-zero
    intermittent-demand window, as can occur in M5) — callers should handle NaN."""
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


def without_residuals(metrics: dict) -> dict:
    """Drop per-timestep residuals from a metrics dict, returning a copy.

    Every result-writing script strips residuals before serialising: they
    are one float per test point and dominate the output file size. The
    scripts each expose a `--keep-residuals` flag that skips this call,
    because Diebold-Mariano testing needs the per-timestep error series
    (it compares two of them point-by-point) and RMSE cannot reconstruct
    one.

    Lives here, beside the `compute_metrics` return value that creates the
    key, so the four call sites share one spelling of it rather than four
    independent string literals that can drift apart via a typo.
    """
    return {k: v for k, v in metrics.items() if k != "residuals"}


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

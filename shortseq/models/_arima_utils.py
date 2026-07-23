"""Shared rolling-forecast loop for pmdarima-based models (ARIMA, SARIMA).

Ported from the update/predict loop inside `code/experiment.py:run_arima`.
"""
import time

import numpy as np


def rolling_arima_forecast(model, train_values, test_values):
    """Roll a fitted pmdarima model forward one step at a time.

    After each prediction, the true test value is appended to history and
    the model is updated with it before the next prediction — the paper's
    strict no-look-ahead rolling one-step-ahead protocol.
    """
    t0 = time.time()
    history = list(train_values)
    forecasts = []
    for _ in range(len(test_values)):
        model.update(history[-1:])
        fc = model.predict(n_periods=1)[0]
        forecasts.append(fc)
        history.append(test_values[len(forecasts) - 1])
    pred_time = time.time() - t0
    return np.array(forecasts), pred_time

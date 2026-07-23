"""ARIMA baseline — auto-ARIMA with AIC-based order selection (non-seasonal).

Ported from `code/experiment.py:run_arima` (seasonal=False branch).
"""
import time

import pandas as pd
import pmdarima as pm

from ._arima_utils import rolling_arima_forecast
from .base import BaseForecaster, Forecast


class ARIMAForecaster(BaseForecaster):
    def __init__(self, max_p: int = 7, max_q: int = 7):
        super().__init__()
        self.max_p = max_p
        self.max_q = max_q
        self._model = None
        self._train_values: list[float] = []

    def fit(self, train: pd.Series) -> "ARIMAForecaster":
        t0 = time.time()
        values = train.values.astype(float)
        try:
            self._model = pm.auto_arima(
                values, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore",
                max_p=self.max_p, max_q=self.max_q,
            )
        except Exception:
            self._model = pm.auto_arima(
                values, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore",
            )
        self.name = f"ARIMA{self._model.order}"
        self._train_values = list(values)
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        test_values = test.values.astype(float)
        forecasts, pred_time = rolling_arima_forecast(self._model, self._train_values, test_values)
        self.pred_time_ = pred_time
        return Forecast(point=forecasts)

"""SARIMA baseline — seasonal auto-ARIMA.

Ported from `code/experiment.py:run_arima` (seasonal=True branch).
"""
import time

import pandas as pd
import pmdarima as pm

from ._arima_utils import rolling_arima_forecast
from .base import BaseForecaster, Forecast


class SARIMAForecaster(BaseForecaster):
    def __init__(self, max_p: int = 5, max_q: int = 5, max_P: int = 2, max_Q: int = 2):
        super().__init__()
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self._model = None
        self._train_values: list[float] = []

    def fit(self, train: pd.Series) -> "SARIMAForecaster":
        t0 = time.time()
        values = train.values.astype(float)
        m = 7 if len(values) > 50 else 4
        try:
            self._model = pm.auto_arima(
                values, seasonal=True, m=m, stepwise=True,
                suppress_warnings=True, error_action="ignore",
                max_p=self.max_p, max_q=self.max_q,
                max_P=self.max_P, max_Q=self.max_Q,
            )
            self.name = f"SARIMA{self._model.order}x{self._model.seasonal_order}"
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

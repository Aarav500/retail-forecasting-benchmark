"""Prophet baseline — additive decomposition with automatic seasonality.

Ported from `code/experiment.py:run_prophet`. Unlike the other baselines,
Prophet forecasts the whole test horizon in a single batch call (no
step-by-step feedback) — this matches the original benchmark exactly;
`predict_rolling` here just means "produce one point per test
timestamp", not that the model is updated between steps.
"""
import time

import numpy as np
import pandas as pd
from prophet import Prophet

from .base import BaseForecaster, Forecast


class ProphetForecaster(BaseForecaster):
    def __init__(self, freq: str = "D"):
        super().__init__()
        self.freq = freq
        self.name = "Prophet"
        self._model = None

    def fit(self, train: pd.Series) -> "ProphetForecaster":
        t0 = time.time()
        train_df = pd.DataFrame({"ds": train.index, "y": train.values})
        self._model = Prophet(
            yearly_seasonality=True, weekly_seasonality=True,
            daily_seasonality=False, seasonality_mode="additive",
            interval_width=0.95,
        )
        self._model.fit(train_df)
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        future = self._model.make_future_dataframe(periods=len(test), freq=self.freq)
        forecast = self._model.predict(future)
        preds = forecast["yhat"].values[-len(test):]
        self.pred_time_ = time.time() - t0
        return Forecast(point=np.array(preds))

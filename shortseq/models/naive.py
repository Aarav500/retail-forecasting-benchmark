"""Seasonal-naive reference baseline — forecasts each point as the value
one season-length ago.

Named as the standard benchmark reference in the NeurIPS execution
plan's baseline table; not present in the TMLR-submitted code.
"""
import time

import numpy as np
import pandas as pd

from .base import BaseForecaster, Forecast


class SeasonalNaiveForecaster(BaseForecaster):
    def __init__(self, season_length: int = 7):
        super().__init__()
        self.season_length = season_length
        self.name = f"SeasonalNaive(m={season_length})"
        self._train_values: list[float] = []

    def fit(self, train: pd.Series) -> "SeasonalNaiveForecaster":
        t0 = time.time()
        self._train_values = list(train.values.astype(float))
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        history = list(self._train_values)
        test_values = test.values.astype(float)
        m = self.season_length
        forecasts = []
        for i in range(len(test_values)):
            pred = history[-m] if len(history) >= m else float(np.mean(history))
            forecasts.append(pred)
            history.append(test_values[i])
        self.pred_time_ = time.time() - t0
        return Forecast(point=np.array(forecasts))

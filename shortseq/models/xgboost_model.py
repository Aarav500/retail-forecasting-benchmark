"""XGBoost baseline — gradient-boosted trees over a lagged feature window.

Ported from `code/experiment.py:run_xgboost` / `create_features_xgb`.
"""
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseForecaster, Forecast


def create_lag_features(series: np.ndarray, lags: int = 14, horizon: int = 1):
    """Build (X, y) lag-feature pairs for supervised regression on a series."""
    X, y = [], []
    for i in range(lags, len(series) - horizon + 1):
        X.append(series[i - lags:i])
        y.append(series[i + horizon - 1])
    return np.array(X), np.array(y)


class XGBoostForecaster(BaseForecaster):
    def __init__(self, lags: int = 14, n_estimators: int = 200, max_depth: int = 5,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42):
        super().__init__()
        self.lags = lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.name = "XGBoost"
        self._model = None
        self._train_values: list[float] = []

    def fit(self, train: pd.Series) -> "XGBoostForecaster":
        t0 = time.time()
        values = train.values.astype(float)
        X_train, y_train = create_lag_features(values, self.lags)
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state, verbosity=0,
        )
        self._model.fit(X_train, y_train)
        self._train_values = list(values)
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        history = list(self._train_values)
        test_values = test.values.astype(float)
        forecasts = []
        for i in range(len(test_values)):
            # NOTE: the `else` branch below is structurally unreachable and kept only
            # because it mirrors the original script. `fit()` calls create_lag_features
            # on `train`, which raises ValueError itself if len(train) < self.lags
            # (empty 1-D array), so a model only ever exists when len(train) >= lags;
            # `history` starts at len(train) and only grows in this loop, so
            # `len(history) >= self.lags` always holds here. Not a working safety net
            # for short series — a true small-series guard would need to live in fit().
            if len(history) >= self.lags:
                x = np.array(history[-self.lags:]).reshape(1, -1)
                pred = self._model.predict(x)[0]
            else:
                pred = np.mean(history)
            forecasts.append(pred)
            history.append(test_values[i])
        self.pred_time_ = time.time() - t0
        return Forecast(point=np.array(forecasts))

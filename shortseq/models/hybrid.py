"""Hybrid ARIMA+XGBoost baseline.

Ported from `code/experiment.py:run_hybrid`. Fits ARIMA, then an XGBoost
regressor on ARIMA's in-sample residuals; forecasts are the ARIMA
forecast plus the XGBoost residual correction.
"""
import copy
import time

import numpy as np
import pandas as pd
import pmdarima as pm
import xgboost as xgb

from .base import BaseForecaster, Forecast
from .xgboost_model import create_lag_features


class HybridForecaster(BaseForecaster):
    def __init__(self, lags: int = 14, arima_max_p: int = 5, arima_max_q: int = 5,
                 xgb_n_estimators: int = 100, xgb_max_depth: int = 4,
                 xgb_learning_rate: float = 0.1, random_state: int = 42):
        super().__init__()
        self.lags = lags
        self.arima_max_p = arima_max_p
        self.arima_max_q = arima_max_q
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.random_state = random_state
        self.name = "Hybrid(ARIMA+XGB)"
        self._arima_model = None
        self._xgb_model = None
        self._has_xgb = False
        self._train_values: list[float] = []
        self._res_history: list[float] = []

    def fit(self, train: pd.Series) -> "HybridForecaster":
        t0 = time.time()
        values = train.values.astype(float)
        try:
            self._arima_model = pm.auto_arima(
                values, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore",
                max_p=self.arima_max_p, max_q=self.arima_max_q,
            )
        except Exception:
            self._arima_model = pm.auto_arima(
                values, seasonal=False, information_criterion="aic",
                suppress_warnings=True,
            )

        arima_train_preds = self._arima_model.predict_in_sample()
        arima_residuals = values - arima_train_preds

        if len(arima_residuals) > self.lags + 5:
            X_res, y_res = create_lag_features(arima_residuals, self.lags)
            self._xgb_model = xgb.XGBRegressor(
                n_estimators=self.xgb_n_estimators, max_depth=self.xgb_max_depth,
                learning_rate=self.xgb_learning_rate, random_state=self.random_state,
                verbosity=0,
            )
            self._xgb_model.fit(X_res, y_res)
            self._has_xgb = True

        self._train_values = list(values)
        self._res_history = list(arima_residuals)
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        # Deep-copy before rolling: pmdarima's .update() mutates the model
        # in place, so rolling on self._arima_model directly would make
        # predict_rolling non-idempotent across repeated calls (see Task 6's
        # ARIMA/SARIMA fix for the same issue).
        arima_model = copy.deepcopy(self._arima_model)
        history = list(self._train_values)
        res_history = list(self._res_history)
        test_values = test.values.astype(float)
        forecasts = []

        for i in range(len(test_values)):
            arima_model.update(history[-1:])
            arima_fc = arima_model.predict(n_periods=1)[0]

            if self._has_xgb and len(res_history) >= self.lags:
                x_res = np.array(res_history[-self.lags:]).reshape(1, -1)
                res_correction = self._xgb_model.predict(x_res)[0]
            else:
                res_correction = 0.0

            final_fc = arima_fc + res_correction
            forecasts.append(final_fc)

            actual_val = test_values[i]
            history.append(actual_val)
            new_residual = actual_val - arima_fc
            res_history.append(new_residual)

        self.pred_time_ = time.time() - t0
        return Forecast(point=np.array(forecasts))

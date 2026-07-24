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
    # pandas 3.x removed the deprecated 'M' (month-end) offset alias that
    # make_future_dataframe forwards straight into pd.date_range. ShortSeq's
    # only freq="M" datasets (M4, see datasets/m4.py) use a month-*start*
    # synthetic index, so the correct replacement is 'MS' -- not the
    # mechanical 'M'->'ME' pandas' own error suggests, which produces
    # month-*end* dates that fail the contiguity check in predict_rolling.
    _FREQ_ALIASES = {"M": "MS"}

    def __init__(self, freq: str = "D", yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True, daily_seasonality: bool = False,
                 seasonality_mode: str = "additive", interval_width: float = 0.95):
        super().__init__()
        self.freq = self._FREQ_ALIASES.get(freq, freq)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.interval_width = interval_width
        self.name = "Prophet"
        self._model = None

    def fit(self, train: pd.Series) -> "ProphetForecaster":
        t0 = time.time()
        train_df = pd.DataFrame({"ds": train.index, "y": train.values})
        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            interval_width=self.interval_width,
        )
        self._model.fit(train_df)
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        future = self._model.make_future_dataframe(periods=len(test), freq=self.freq)

        # Defensive check: make_future_dataframe extrapolates future dates
        # purely from train's last date + freq, ignoring test.index entirely.
        # If test were ever not the immediate, gap-free continuation of train
        # at this frequency, yhat would silently line up against the wrong
        # calendar dates with no error — unlike the other 6 (index-blind)
        # baselines, Prophet's correctness here depends on calendar
        # arithmetic, not just value ordering. Every current call site does
        # a contiguous iloc[:i]/iloc[i:] split, so this is latent, not
        # triggered today, but it's cheap insurance.
        expected_dates = future["ds"].values[-len(test):]
        if not (expected_dates == test.index.values).all():
            raise ValueError(
                "test is not the contiguous continuation of train at the given freq; "
                "Prophet's predict_rolling extrapolates future dates from train's end, "
                "so a gap or misaligned split would silently score against the wrong dates."
            )

        forecast = self._model.predict(future)
        preds = forecast["yhat"].values[-len(test):]
        self.pred_time_ = time.time() - t0
        return Forecast(point=np.array(preds))

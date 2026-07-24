import numpy as np
import pandas as pd
import pytest

from shortseq.datasets.dmart import load_dmart
from shortseq.models.hybrid import HybridForecaster
from shortseq.models.naive import SeasonalNaiveForecaster


def _split(series, frac=0.8):
    i = int(len(series) * frac)
    return series.iloc[:i], series.iloc[i:]


def test_hybrid_forecaster_fits_and_predicts_real_dmart_food():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = HybridForecaster().fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None
    assert model.pred_time_ is not None
    assert model.name == "Hybrid(ARIMA+XGB)"


def test_seasonal_naive_forecaster_reproduces_last_season_values():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = SeasonalNaiveForecaster(season_length=7).fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    # First 7 rolling predictions are exactly the last 7 training values, in order.
    assert np.array_equal(forecast.point[:7], train.values[-7:])
    assert model.pred_time_ is not None
    assert model.name == "SeasonalNaive(m=7)"


def test_hybrid_predict_rolling_is_idempotent_across_repeated_calls():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = HybridForecaster().fit(train)
    fc1 = model.predict_rolling(test)
    fc2 = model.predict_rolling(test)
    assert np.array_equal(fc1.point, fc2.point)


def test_seasonal_naive_falls_back_to_mean_when_train_shorter_than_season():
    series = pd.Series([1.0, 2.0, 3.0])
    model = SeasonalNaiveForecaster(season_length=10).fit(series)
    forecast = model.predict_rolling(pd.Series([4.0]))
    assert forecast.point[0] == pytest.approx(2.0)  # mean of [1, 2, 3]

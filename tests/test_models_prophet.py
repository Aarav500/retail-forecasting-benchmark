import numpy as np
import pytest

from shortseq.datasets.dmart import load_dmart
from shortseq.models.prophet_model import ProphetForecaster


def _split(series, frac=0.8):
    i = int(len(series) * frac)
    return series.iloc[:i], series.iloc[i:]


def test_prophet_forecaster_fits_and_predicts_real_dmart_food():
    ds = load_dmart()["dmart_food"]
    train, test = _split(ds.series)
    model = ProphetForecaster(freq=ds.freq).fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None
    assert model.pred_time_ is not None
    assert model.name == "Prophet"


def test_prophet_predict_rolling_is_idempotent_across_repeated_calls():
    ds = load_dmart()["dmart_food"]
    train, test = _split(ds.series)
    model = ProphetForecaster(freq=ds.freq).fit(train)
    fc1 = model.predict_rolling(test)
    fc2 = model.predict_rolling(test)
    assert np.array_equal(fc1.point, fc2.point)


def test_prophet_forecaster_accepts_yaml_hyperparameters():
    ds = load_dmart()["dmart_food"]
    train, test = _split(ds.series)
    model = ProphetForecaster(
        freq=ds.freq, yearly_seasonality=False, weekly_seasonality=False,
        daily_seasonality=True, seasonality_mode="multiplicative",
        interval_width=0.8,
    ).fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)


def test_prophet_predict_rolling_rejects_non_contiguous_test():
    ds = load_dmart()["dmart_food"]
    train, test = _split(ds.series)
    non_contiguous_test = test.iloc[1:]  # drops the first date, opening a gap
    model = ProphetForecaster(freq=ds.freq).fit(train)
    with pytest.raises(ValueError):
        model.predict_rolling(non_contiguous_test)

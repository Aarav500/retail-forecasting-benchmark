import numpy as np

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


def test_seasonal_naive_forecaster_reproduces_last_season_values():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = SeasonalNaiveForecaster(season_length=7).fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    # First 7 rolling predictions are exactly the last 7 training values, in order.
    assert np.array_equal(forecast.point[:7], train.values[-7:])

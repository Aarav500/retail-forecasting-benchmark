import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")

from shortseq.datasets.dmart import load_dmart
from shortseq.models.foundation.chronos import ChronosForecaster


def _split(series, frac=0.8):
    i = int(len(series) * frac)
    return series.iloc[:i], series.iloc[i:]


def test_chronos_forecaster_fits_and_predicts_real_dmart_food():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = ChronosForecaster(size="small").fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None
    assert model.pred_time_ is not None
    assert model.name == "Chronos-small"
    # dist should carry one sample-path list per test point
    assert len(forecast.dist) == len(test)


def test_chronos_predict_rolling_is_idempotent_across_repeated_calls():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = ChronosForecaster(size="small").fit(train)
    fc1 = model.predict_rolling(test)
    fc2 = model.predict_rolling(test)
    assert np.array_equal(fc1.point, fc2.point)

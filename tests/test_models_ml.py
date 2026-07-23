from shortseq.datasets.dmart import load_dmart
from shortseq.models.xgboost_model import XGBoostForecaster, create_lag_features
from shortseq.models.lstm_model import LSTMForecaster
import numpy as np


def _split(series, frac=0.8):
    i = int(len(series) * frac)
    return series.iloc[:i], series.iloc[i:]


def test_create_lag_features_shapes():
    series = np.arange(20, dtype=float)
    X, y = create_lag_features(series, lags=5, horizon=1)
    assert X.shape == (15, 5)
    assert y.shape == (15,)
    assert list(X[0]) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert y[0] == 5.0


def test_xgboost_forecaster_fits_and_predicts_real_dmart_food():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = XGBoostForecaster().fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None


def test_lstm_forecaster_fits_and_predicts_real_dmart_food():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = LSTMForecaster(epochs=5).fit(train)  # few epochs — this is a smoke test, not a quality test
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None


def test_xgboost_predict_rolling_is_idempotent_across_repeated_calls():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = XGBoostForecaster().fit(train)
    fc1 = model.predict_rolling(test)
    fc2 = model.predict_rolling(test)
    assert np.array_equal(fc1.point, fc2.point)


def test_lstm_predict_rolling_is_idempotent_across_repeated_calls():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = LSTMForecaster(epochs=5).fit(train)
    fc1 = model.predict_rolling(test)
    fc2 = model.predict_rolling(test)
    assert np.array_equal(fc1.point, fc2.point)

from shortseq.datasets.dmart import load_dmart
from shortseq.models.arima import ARIMAForecaster
from shortseq.models.sarima import SARIMAForecaster


def _split(series, frac=0.8):
    i = int(len(series) * frac)
    return series.iloc[:i], series.iloc[i:]


def test_arima_forecaster_fits_and_predicts_real_dmart_food():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = ARIMAForecaster().fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None
    assert model.pred_time_ is not None
    assert model.name.startswith("ARIMA")


def test_sarima_forecaster_fits_and_predicts_real_dmart_food():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = SARIMAForecaster().fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None

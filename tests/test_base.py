import numpy as np
import pandas as pd
import pytest

from shortseq.models.base import BaseForecaster, Forecast
from shortseq.datasets.base import SeriesDataset, compute_cv, compute_ac1, compute_zero_frac, make_dataset


def test_forecast_holds_point_and_optional_dist():
    fc = Forecast(point=np.array([1.0, 2.0, 3.0]))
    assert fc.dist is None
    assert list(fc.point) == [1.0, 2.0, 3.0]


def test_base_forecaster_is_abstract():
    with pytest.raises(TypeError):
        BaseForecaster()


def test_compute_cv():
    series = pd.Series([10.0, 20.0, 30.0])
    assert compute_cv(series) == pytest.approx(series.std() / series.mean())


def test_compute_zero_frac():
    series = pd.Series([0.0, 1.0, 0.0, 2.0])
    assert compute_zero_frac(series) == pytest.approx(0.5)


def test_compute_ac1_of_constant_series_is_not_nan_or_errors_gracefully():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ac1 = compute_ac1(series)
    assert isinstance(ac1, float)


def test_make_dataset_builds_series_dataset_with_stats():
    series = pd.Series([1.0, 2.0, 3.0, 0.0, 5.0], index=pd.date_range("2020-01-01", periods=5, freq="D"))
    ds = make_dataset("toy", series, freq="D", real=True)
    assert isinstance(ds, SeriesDataset)
    assert ds.name == "toy"
    assert ds.n == 5
    assert ds.real is True
    assert ds.zero_frac == pytest.approx(0.2)
    assert ds.cv == pytest.approx(compute_cv(series))
    assert ds.ac1 == pytest.approx(compute_ac1(series))

import pytest

from shortseq.models.foundation.timesfm import TimesFMForecaster
from shortseq.models.foundation.moirai import MoiraiForecaster
from shortseq.models.foundation.moment import MomentForecaster
from shortseq.models.foundation.timer import TimerForecaster
from shortseq.models.foundation.ttm import TTMForecaster
from shortseq.models.foundation.lag_llama import LagLlamaForecaster
from shortseq.models.foundation.forecastpfn import ForecastPFNForecaster

STUB_CLASSES = [
    TimesFMForecaster, MoiraiForecaster, MomentForecaster,
    TimerForecaster, TTMForecaster, LagLlamaForecaster, ForecastPFNForecaster,
]


@pytest.mark.parametrize("cls", STUB_CLASSES)
def test_foundation_stub_raises_not_implemented_on_fit(cls):
    model = cls()
    with pytest.raises(NotImplementedError):
        model.fit(None)


@pytest.mark.parametrize("cls", STUB_CLASSES)
def test_foundation_stub_raises_not_implemented_on_predict(cls):
    model = cls()
    with pytest.raises(NotImplementedError):
        model.predict_rolling(None)


def test_foundation_stub_default_names():
    assert TimesFMForecaster().name == "TimesFM-200m"
    assert MoiraiForecaster().name == "Moirai-small"
    assert MomentForecaster().name == "Moment-small"
    assert TimerForecaster().name == "Timer-base"
    assert TTMForecaster().name == "TTM-512-96"
    assert LagLlamaForecaster().name == "Lag-Llama"
    assert ForecastPFNForecaster().name == "ForecastPFN"

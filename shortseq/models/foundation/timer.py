"""Timer (THU) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "Timer is not yet installed/wrapped — see the NeurIPS execution "
    "plan, Week 1 (foundation model installation)."
)


class TimerForecaster(BaseForecaster):
    """Zero-shot Timer forecaster. Sizes: base(~50M), large(~670M) per
    public reports; not independently verified against the actual model
    release."""

    def __init__(self, size: str = "base"):
        super().__init__()
        self.size = size
        self.name = f"Timer-{size}"

    def fit(self, train: pd.Series) -> "TimerForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)

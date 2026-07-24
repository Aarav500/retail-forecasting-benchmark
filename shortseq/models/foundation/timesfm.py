"""TimesFM (Google) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "TimesFM is not yet installed/wrapped — see the NeurIPS execution "
    "plan, Week 1 (foundation model installation)."
)


class TimesFMForecaster(BaseForecaster):
    """Zero-shot TimesFM forecaster. 200M-parameter patched model per the
    execution plan."""

    def __init__(self, size: str = "200m"):
        super().__init__()
        self.size = size
        self.name = f"TimesFM-{size}"

    def fit(self, train: pd.Series) -> "TimesFMForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)

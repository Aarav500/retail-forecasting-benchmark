"""TTM / Tiny Time Mixers (IBM) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "TTM is not yet installed/wrapped — see the NeurIPS execution plan, "
    "Week 1 (foundation model installation)."
)


class TTMForecaster(BaseForecaster):
    """Zero-shot Tiny Time Mixers forecaster. Context-horizon configs per
    the execution plan: 512-96, 1024-96, 1536-96."""

    def __init__(self, config: str = "512-96"):
        super().__init__()
        self.config = config
        self.name = f"TTM-{config}"

    def fit(self, train: pd.Series) -> "TTMForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)

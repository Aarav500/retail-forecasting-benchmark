"""Moirai (Salesforce) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "Moirai is not yet installed/wrapped — see the NeurIPS execution "
    "plan, Week 1 (foundation model installation)."
)


class MoiraiForecaster(BaseForecaster):
    """Zero-shot Moirai forecaster. Sizes per the execution plan:
    small(14M), base(91M), large(311M)."""

    def __init__(self, size: str = "small"):
        super().__init__()
        self.size = size
        self.name = f"Moirai-{size}"

    def fit(self, train: pd.Series) -> "MoiraiForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)

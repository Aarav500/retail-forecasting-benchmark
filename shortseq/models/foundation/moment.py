"""Moment (CMU) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "Moment is not yet installed/wrapped — see the NeurIPS execution "
    "plan, Week 1 (foundation model installation)."
)


class MomentForecaster(BaseForecaster):
    """Zero-shot Moment forecaster. Sizes per the execution plan:
    small(40M), base(125M), large(385M)."""

    def __init__(self, size: str = "small"):
        super().__init__()
        self.size = size
        self.name = f"Moment-{size}"

    def fit(self, train: pd.Series) -> "MomentForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)

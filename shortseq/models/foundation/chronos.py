"""Chronos (Amazon) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "Chronos is not yet installed/wrapped — see the NeurIPS execution "
    "plan, Week 1 (foundation model installation)."
)


class ChronosForecaster(BaseForecaster):
    """Zero-shot Chronos forecaster. Sizes per the execution plan:
    tiny(8M), mini(20M), small(46M), base(200M), large(710M)."""

    def __init__(self, size: str = "small"):
        super().__init__()
        self.size = size
        self.name = f"Chronos-{size}"

    def fit(self, train: pd.Series) -> "ChronosForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)

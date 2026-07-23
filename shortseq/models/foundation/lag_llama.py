"""Lag-Llama (ServiceNow) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "Lag-Llama is not yet installed/wrapped — see the NeurIPS execution "
    "plan, Week 1 (foundation model installation)."
)


class LagLlamaForecaster(BaseForecaster):
    """Zero-shot Lag-Llama forecaster (6M parameters, single size per the
    execution plan)."""

    def __init__(self):
        super().__init__()
        self.name = "Lag-Llama"

    def fit(self, train: pd.Series) -> "LagLlamaForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)

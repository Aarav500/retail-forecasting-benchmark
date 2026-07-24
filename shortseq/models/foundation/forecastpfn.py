"""ForecastPFN (prior-data fitted network) — stub. Not yet implemented;
see shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "ForecastPFN is not yet installed/wrapped — see the NeurIPS "
    "execution plan, Week 1 (foundation model installation)."
)


class ForecastPFNForecaster(BaseForecaster):
    """Zero-shot ForecastPFN forecaster (~25M parameters per public
    reports, single size; not independently verified against the actual
    model release)."""

    def __init__(self):
        super().__init__()
        self.name = "ForecastPFN"

    def fit(self, train: pd.Series) -> "ForecastPFNForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)

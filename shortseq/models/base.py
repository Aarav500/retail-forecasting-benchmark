"""Common model interface for every ShortSeq baseline and (later) foundation model.

Follows the rolling one-step-ahead protocol used throughout the TMLR
benchmark: `fit` trains on the training series, then `predict_rolling`
forecasts each point in the test series one step at a time, feeding the
true value back into the model's history before the next step (no
look-ahead).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Forecast:
    """Point forecast plus an optional predictive distribution.

    `dist` is reserved for probabilistic models (e.g. foundation models)
    that produce more than a point estimate; unset for classical/ML
    baselines, none of which currently produce one.

    `point` is excluded from the generated `__eq__` (compare=False):
    it's a numpy array, and the default dataclass equality does a
    tuple-wise `==` over fields, which raises `ValueError: The truth
    value of an array ... is ambiguous` for array-valued fields instead
    of returning True/False.
    """
    point: np.ndarray = field(compare=False)
    dist: Optional[object] = None


class BaseForecaster(ABC):
    def __init__(self) -> None:
        self.name: str = self.__class__.__name__
        self.train_time_: Optional[float] = None
        self.pred_time_: Optional[float] = None

    @abstractmethod
    def fit(self, train: pd.Series) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict_rolling(self, test: pd.Series) -> Forecast:
        ...

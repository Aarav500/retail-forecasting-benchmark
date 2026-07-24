"""Chronos (Amazon) — zero-shot foundation model forecaster.

Uses the official `chronos-forecasting` package. Zero-shot: `fit()` only
loads the pretrained model and stores context; no gradient training
happens. `predict_rolling` mirrors the classical baselines' one-step
rolling protocol (predict 1 step, reveal the true value, extend
context, repeat) rather than a single batch call, so DM tests stay
comparable across all models. `Forecast.dist` is populated with the
raw sampled trajectories per step (reserved for this since Task 3 —
CRPS/coverage stubs can consume it later).
"""
import time

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

from ..base import BaseForecaster, Forecast

_MODEL_IDS = {
    "tiny": "amazon/chronos-t5-tiny",
    "mini": "amazon/chronos-t5-mini",
    "small": "amazon/chronos-t5-small",
    "base": "amazon/chronos-t5-base",
    "large": "amazon/chronos-t5-large",
}


class ChronosForecaster(BaseForecaster):
    def __init__(self, size: str = "small", num_samples: int = 20):
        super().__init__()
        if size not in _MODEL_IDS:
            raise ValueError(f"Unknown Chronos size {size!r}; choose from {list(_MODEL_IDS)}")
        self.size = size
        self.num_samples = num_samples
        self.name = f"Chronos-{size}"
        self._pipeline = None
        self._context: list = []

    def fit(self, train: pd.Series) -> "ChronosForecaster":
        t0 = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipeline = ChronosPipeline.from_pretrained(
            _MODEL_IDS[self.size],
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        self._context = list(train.values.astype(float))
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        context = list(self._context)
        test_values = test.values.astype(float)
        points = []
        dist = []
        for i in range(len(test_values)):
            context_tensor = torch.tensor(context, dtype=torch.float32)
            samples_tensor = self._pipeline.predict(
                context=context_tensor,
                prediction_length=1,
                num_samples=self.num_samples,
            )
            samples = samples_tensor[0, :, 0].float().cpu().numpy()
            points.append(float(np.median(samples)))
            dist.append(samples.tolist())
            context.append(test_values[i])  # true value revealed, no look-ahead
        self.pred_time_ = time.time() - t0
        return Forecast(point=np.array(points), dist=dist)

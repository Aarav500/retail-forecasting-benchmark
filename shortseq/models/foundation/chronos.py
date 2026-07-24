"""Chronos (Amazon) — zero-shot foundation model forecaster.

Uses the official `chronos-forecasting` package. Zero-shot: `fit()` only
loads the pretrained model and stores context; no gradient training
happens. `predict_rolling` mirrors the classical baselines' one-step
rolling protocol (predict 1 step, reveal the true value, extend
context, repeat) rather than a single batch call, so DM tests stay
comparable across all models. `Forecast.dist` is populated with the
raw sampled trajectories per step (reserved for this since Task 3 —
CRPS/coverage stubs can consume it later).

`ChronosPipeline.predict()` samples stochastically (temperature/top-k/
top-p) with no seed of its own, so `predict_rolling` seeds `torch` once
per call before its rolling loop to match the determinism convention
used by the other baselines (`random_seed` on LSTM/XGBoost, etc.) and
to make repeated calls with the same inputs reproducible.

The `context`/`inputs` keyword name for `ChronosPipeline.predict()` has
varied across package versions (older examples used `context=`; the
current API — see the `chronos-forecasting` docs — uses `inputs=`).
`requirements-gpu.txt` pins `chronos-forecasting>=1.4.0` with no upper
bound, so the installed version on the GPU box isn't fixed at wrapper-
authoring time; the tensor is passed positionally below so this works
regardless of which keyword name the installed version expects.
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
    def __init__(self, size: str = "small", num_samples: int = 20, seed: int = 42):
        super().__init__()
        if size not in _MODEL_IDS:
            raise ValueError(f"Unknown Chronos size {size!r}; choose from {list(_MODEL_IDS)}")
        self.size = size
        self.num_samples = num_samples
        self.seed = seed
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
        # Seed once per call so two calls with the same trained model and
        # the same `test` series draw identical sample paths (stochastic
        # sampling otherwise has no reproducibility guarantee).
        torch.manual_seed(self.seed)
        context_tensor = torch.tensor(self._context, dtype=torch.float32)
        test_values = test.values.astype(float)
        points = []
        dist = []
        for i in range(len(test_values)):
            # Positional arg: `chronos-forecasting` has renamed this
            # parameter across versions (`context=` -> `inputs=`); passing
            # positionally works regardless of which name is installed.
            samples_tensor = self._pipeline.predict(
                context_tensor,
                prediction_length=1,
                num_samples=self.num_samples,
                limit_prediction_length=True,
            )
            samples = samples_tensor[0, :, 0].float().cpu().numpy()
            points.append(float(np.median(samples)))
            dist.append(samples.tolist())
            # True value revealed, no look-ahead. Append via `cat` instead
            # of rebuilding the tensor from the full Python list each step
            # (avoids O(n^2) rebuilds across a long rolling loop).
            next_val = torch.tensor([test_values[i]], dtype=torch.float32)
            context_tensor = torch.cat([context_tensor, next_val])
        self.pred_time_ = time.time() - t0
        return Forecast(point=np.array(points), dist=dist)

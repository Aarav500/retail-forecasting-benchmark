"""Foundation-model wrappers — not yet implemented.

The NeurIPS execution plan's Week 1 calls for installing and wrapping
Chronos, TimesFM, Moirai, Moment, Timer, TTM, Lag-Llama, and ForecastPFN,
which needs GPU access and new heavy dependencies (torch,
huggingface_hub downloads, etc.) not set up in this scaffolding pass.
Each module here defines a BaseForecaster subclass with the right name
and constructor signature so the registry and tests can already
reference every model by name; `fit`/`predict_rolling` raise
NotImplementedError until the real wrapper is built.
"""

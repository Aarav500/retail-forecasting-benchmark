"""Residuals must be retainable on request.

Per-timestep residuals were stripped unconditionally at write time, which
foreclosed Diebold-Mariano testing on the size-scaling campaign: a DM test
compares two models' error series point-by-point, and RMSE alone cannot
reconstruct that. These tests pin the opt-in behaviour so the same dead
end is not hit again.

The first four tests are structural: two read signatures, two read
`main()`'s source text. They are kept because they state the intended
shape of the API directly, but they are NOT sufficient on their own — a
source-text assertion passes even if the flag is declared in argparse and
then never threaded through to the call site, which is the most likely way
this feature breaks. Everything below `_series` therefore exercises the
real functions: that the metrics dict retains or drops `residuals`
according to the argument, and that the CLI flag genuinely reaches
`run_one`/`run_dataset`.
"""
import inspect
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import experiments.scripts.run_baselines as rb
import experiments.scripts.run_size_scaling as rss
from shortseq.models.naive import SeasonalNaiveForecaster


def test_size_scaling_run_one_accepts_keep_residuals():
    sig = inspect.signature(rss.run_one)
    assert "keep_residuals" in sig.parameters
    assert sig.parameters["keep_residuals"].default is False


def test_baselines_run_dataset_accepts_keep_residuals():
    sig = inspect.signature(rb.run_dataset)
    assert "keep_residuals" in sig.parameters
    assert sig.parameters["keep_residuals"].default is False


def test_size_scaling_cli_exposes_keep_residuals_flag():
    src = inspect.getsource(rss.main)
    assert "--keep-residuals" in src


def test_baselines_cli_exposes_keep_residuals_flag():
    src = inspect.getsource(rb.main)
    assert "--keep-residuals" in src


N_POINTS = 40
N_TEST = N_POINTS - int(N_POINTS * 0.8)  # both scripts split 0.8 by default


def _series(n: int = N_POINTS) -> pd.Series:
    """A short, strictly positive daily series.

    Positive because `compute_metrics` divides by `actual` for MAPE;
    a zero would make the metric NaN and json.dump would then emit a
    bare `NaN` literal, failing for a reason unrelated to residuals.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(100.0 + np.arange(n, dtype=float), index=idx)


class _StubForecast:
    def __init__(self, point):
        self.point = point
        self.dist = None


class _StubChronos:
    """Stand-in for ChronosForecaster, which run_one imports lazily.

    The real class imports torch and chronos-forecasting, neither of
    which is installed on a CPU-only machine (see
    tests/test_chronos_model.py's importorskip). Substituting the model
    keeps run_one itself — the code under test — completely real; only
    the GPU dependency is replaced. Predictions are `actual + 1`, so
    every residual is exactly -1.0 and RMSE is exactly 1.0.
    """

    def __init__(self, size: str):
        self.name = f"Chronos-{size}"
        self.train_time_ = 0.0
        self.pred_time_ = 0.0

    def fit(self, train):
        return self

    def predict_rolling(self, test):
        return _StubForecast(np.asarray(test.values, dtype=float) + 1.0)


def _stub_size_scaling(monkeypatch, tmp_path):
    module = types.ModuleType("shortseq.models.foundation.chronos")
    module.ChronosForecaster = _StubChronos
    monkeypatch.setitem(sys.modules, "shortseq.models.foundation.chronos", module)
    monkeypatch.setattr(rss, "RESULTS_DIR", tmp_path)
    # No committed baseline under tmp_path, so load_arima_rmse returns None.
    monkeypatch.setattr(rss, "BASELINE_DIR", tmp_path)
    return SimpleNamespace(series=_series(), n=N_POINTS)


def _written(tmp_path, filename):
    with open(tmp_path / filename) as f:
        return json.load(f)


def test_run_one_strips_residuals_by_default(tmp_path, monkeypatch):
    dataset = _stub_size_scaling(monkeypatch, tmp_path)

    output = rss.run_one("toy", dataset, "tiny")

    assert output["failed_models"] == {}  # otherwise metrics is empty and this passes vacuously
    assert output["metrics"]["Chronos-tiny"]["rmse"] == pytest.approx(1.0)
    assert "residuals" not in output["metrics"]["Chronos-tiny"]
    on_disk = _written(tmp_path, "toy_chronos_tiny.json")
    assert "residuals" not in on_disk["metrics"]["Chronos-tiny"]


def test_run_one_retains_residuals_when_requested(tmp_path, monkeypatch):
    dataset = _stub_size_scaling(monkeypatch, tmp_path)

    output = rss.run_one("toy", dataset, "tiny", keep_residuals=True)

    assert output["failed_models"] == {}
    # The point of the flag: the per-timestep error series, one value per
    # test point, which is exactly what a DM test consumes.
    assert output["metrics"]["Chronos-tiny"]["residuals"] == [-1.0] * N_TEST
    on_disk = _written(tmp_path, "toy_chronos_tiny.json")
    assert on_disk["metrics"]["Chronos-tiny"]["residuals"] == [-1.0] * N_TEST
    # Retaining residuals must not drop anything else.
    assert on_disk["metrics"]["Chronos-tiny"]["rmse"] == pytest.approx(1.0)


def _stub_baselines(monkeypatch, tmp_path):
    """Run run_dataset for real, but against one cheap deterministic model.

    The seven-model sweep over real D-Mart data is already covered by
    tests/test_experiment_scripts.py; re-fitting Prophet/LSTM/Hybrid here
    would add minutes to exercise a single dict comprehension.
    """
    monkeypatch.setattr(
        rb, "build_models",
        lambda config, freq, season_length: {
            "Naive": SeasonalNaiveForecaster(season_length=season_length)
        },
    )
    monkeypatch.setattr(rb, "RESULTS_DIR", tmp_path)
    with open(rb.CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(series=_series(), freq="D"), config


def test_run_dataset_strips_residuals_by_default(tmp_path, monkeypatch):
    dataset, config = _stub_baselines(monkeypatch, tmp_path)

    output = rb.run_dataset("toy", dataset, config)

    assert output["failed_models"] == {}
    assert "residuals" not in output["metrics"]["Naive"]
    assert "residuals" not in _written(tmp_path, "toy_results.json")["metrics"]["Naive"]


def test_run_dataset_retains_residuals_when_requested(tmp_path, monkeypatch):
    dataset, config = _stub_baselines(monkeypatch, tmp_path)

    output = rb.run_dataset("toy", dataset, config, keep_residuals=True)

    assert output["failed_models"] == {}
    residuals = output["metrics"]["Naive"]["residuals"]
    assert len(residuals) == output["n_test"]
    assert _written(tmp_path, "toy_results.json")["metrics"]["Naive"]["residuals"] == residuals
    assert "rmse" in output["metrics"]["Naive"]


def _record_calls(calls):
    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return {}
    return fake


def _keep_residuals_arg(signature, args, kwargs) -> bool:
    """What `keep_residuals` actually was for one recorded call.

    Bound against the real signature so the assertion holds whether the
    caller passes the flag positionally or by keyword, and resolves to
    the declared default when the caller does not pass it at all — which
    is the failure being guarded against.
    """
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments["keep_residuals"]


@pytest.mark.parametrize("flag, expected", [([], False), (["--keep-residuals"], True)])
def test_size_scaling_cli_flag_reaches_run_one(tmp_path, monkeypatch, flag, expected):
    # Declaring the argparse flag and never threading it is invisible to a
    # `"--keep-residuals" in getsource(main)` assertion, and would leave a
    # future DM campaign writing residual-free files anyway.
    signature = inspect.signature(rss.run_one)
    calls = []
    monkeypatch.setattr(rss, "RESULTS_DIR", tmp_path)  # nothing cached, so nothing is skipped
    monkeypatch.setattr(rss, "load_all", lambda: {"toy": SimpleNamespace(series=_series(), n=N_POINTS)})
    monkeypatch.setattr(rss, "run_one", _record_calls(calls))
    monkeypatch.setattr(sys, "argv", ["run_size_scaling.py", "--sizes", "tiny", *flag])

    rss.main()

    assert len(calls) == 1
    assert _keep_residuals_arg(signature, *calls[0]) is expected


@pytest.mark.parametrize("flag, expected", [([], False), (["--keep-residuals"], True)])
def test_baselines_cli_flag_reaches_run_dataset(tmp_path, monkeypatch, flag, expected):
    signature = inspect.signature(rb.run_dataset)
    calls = []
    monkeypatch.setattr(rb, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(rb, "load_all", lambda: {"toy": SimpleNamespace(series=_series(), n=N_POINTS, freq="D")})
    monkeypatch.setattr(rb, "run_dataset", _record_calls(calls))
    monkeypatch.setattr(sys, "argv", ["run_baselines.py", *flag])

    rb.main()

    assert len(calls) == 1
    assert _keep_residuals_arg(signature, *calls[0]) is expected

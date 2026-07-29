"""A payload strict JSON cannot represent must cost one result, not the sweep.

The result-writing scripts pass `allow_nan=False`, so a non-finite metric
raises at write time instead of silently emitting a bare `NaN` literal.
That write sits outside the per-model `try/except`, and the `main()` loops
were unguarded, so the `ValueError` propagated out of the loop and ended
the run. The cost is not uniform: run_size_scaling.py and run_arima_only.py
skip existing files and resume, but run_baselines.py (35 datasets x 7
models) and run_chronos.py have no skip-existing at all, so an abort at
dataset 20 discarded datasets 1-19 as well.

Both properties have to hold at once, which is what these tests pin:
no silent invalid JSON, AND no lost sweep. Concretely, per script --
the error names the dataset and the offending metric, the failing
dataset leaves no file behind, the sweep reaches the next dataset, the
failure is repeated in the final summary where it cannot scroll past,
and the successful write is byte-for-byte what it was before the guard.

The non-finite value is produced the way it actually occurs in this
codebase, not injected: `compute_metrics`' docstring names an all-zero
`actual` window (intermittent demand, as in M5) as making MAPE NaN,
because the `actual != 0` mask is then empty and `np.mean([])` is NaN.
`_zero_series` builds exactly that window and every layer above it --
the model, `compute_metrics`, `run_dataset`/`run_one` -- runs for real.
(numpy emits a "Mean of empty slice" RuntimeWarning on that path; it is
part of the real behaviour being reproduced, not test noise.)
"""
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
from shortseq.results_io import (
    ResultWriteError,
    non_finite_paths,
    print_write_failures,
    write_result_json,
)

N_POINTS = 40


def _good_series(n: int = N_POINTS) -> pd.Series:
    """A strictly positive daily series: every metric is finite."""
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(100.0 + np.arange(n, dtype=float), index=idx)


def _zero_series(n: int = N_POINTS) -> pd.Series:
    """An all-zero daily series, the documented real NaN trigger.

    Every other metric stays finite -- RMSE and MAE are 0.0, and sklearn
    is perfectly happy with zeros -- so this reproduces the awkward case
    the guard exists for: a payload that is *almost* all valid, with one
    NaN buried at `metrics.<model>.mape`.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(np.zeros(n, dtype=float), index=idx)


def _pre_fix_bytes(tmp_path, payload) -> bytes:
    """Exactly what 6a6b4cf's write path emitted, byte for byte.

    Reproduced by running the old call -- `json.dump` straight into a
    text-mode handle with indent=2, allow_nan=False -- rather than
    compared against a hand-written literal, so the assertion covers
    newline translation and float repr on whatever platform runs it.
    """
    reference_dir = tmp_path / "_reference"
    reference_dir.mkdir(exist_ok=True)
    reference = reference_dir / "pre_fix.json"
    with open(reference, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    return reference.read_bytes()


def _after_summary_banner(captured: str) -> str:
    """Everything the run printed after the final write-failure banner.

    Used to distinguish "reported inline, then scrolled away" from
    "reported inline AND repeated in the closing summary" -- an inline
    print alone satisfies a naive substring assertion on the whole
    transcript, and is precisely the failure mode the summary exists to
    prevent on a run that prints for another half hour afterwards.
    """
    marker = "RESULT FILE(S) NOT WRITTEN"
    assert marker in captured, f"no closing write-failure banner in:\n{captured}"
    return captured.split(marker, 1)[1]


# --- shortseq/results_io.py --------------------------------------------------


def test_non_finite_paths_names_the_model_and_the_metric():
    payload = {"dataset": "toy", "metrics": {"Naive": {"rmse": 0.0, "mape": float("nan")}}}

    assert non_finite_paths(payload) == ["metrics.Naive.mape = nan"]


def test_non_finite_paths_indexes_into_a_residual_array():
    # Reachable under --keep-residuals, where the payload carries one
    # float per test point and "somewhere in metrics" is not an answer.
    payload = {"metrics": {"ARIMA": {"residuals": [1.0, float("nan"), 3.0]}}}

    assert non_finite_paths(payload) == ["metrics.ARIMA.residuals[1] = nan"]


def test_non_finite_paths_reports_infinities_and_every_offender():
    payload = {"a": float("inf"), "b": {"c": float("-inf")}, "d": 1.0, "e": None}

    assert non_finite_paths(payload) == ["a = inf", "b.c = -inf"]


def test_non_finite_paths_is_empty_for_a_serialisable_payload():
    assert non_finite_paths({"metrics": {"Naive": {"rmse": 1.0}}, "n": 3, "ok": True}) == []


def test_write_result_json_leaves_no_file_when_the_payload_is_rejected(tmp_path):
    path = tmp_path / "toy_results.json"

    with pytest.raises(ResultWriteError) as excinfo:
        write_result_json(path, {"metrics": {"Naive": {"mape": float("nan")}}})

    assert not path.exists()
    assert "metrics.Naive.mape = nan" in str(excinfo.value)
    assert excinfo.value.path == path


def test_write_result_json_does_not_truncate_an_existing_file_on_failure(tmp_path):
    """The reason the payload is serialised BEFORE the file is opened.

    `json.dump` into an open handle streams as it walks the payload, so
    it can emit bytes and then raise -- and `open(..., "w")` has already
    truncated whatever was there. Both scripts without skip-existing
    rewrite the same path on every invocation, so the previous good
    result is exactly what would be destroyed.
    """
    path = tmp_path / "toy_results.json"
    write_result_json(path, {"metrics": {"Naive": {"rmse": 1.0}}})
    before = path.read_bytes()

    with pytest.raises(ResultWriteError):
        write_result_json(path, {"metrics": {"Naive": {"rmse": 2.0, "mape": float("nan")}}})

    assert path.read_bytes() == before


def test_write_result_json_reports_the_raw_cause_when_nothing_is_non_finite(tmp_path):
    """A non-serialisable value must not be blamed on a NaN that isn't there."""
    path = tmp_path / "toy_results.json"

    with pytest.raises(ResultWriteError) as excinfo:
        write_result_json(path, {"metrics": {"Naive": {"rmse": {1, 2}}}})

    assert not path.exists()
    assert excinfo.value.offenders == []
    assert "TypeError" in str(excinfo.value)


def test_print_write_failures_is_silent_when_nothing_failed(capsys):
    print_write_failures({})

    assert capsys.readouterr().out == ""


# --- run_baselines.py --------------------------------------------------------


def _stub_baselines(monkeypatch, tmp_path):
    """One cheap real model instead of the seven-model sweep.

    SeasonalNaive is a genuine forecaster, so `compute_metrics` and
    `run_dataset` are both entirely real; only the model *set* is
    narrowed, exactly as tests/test_keep_residuals.py does. Re-fitting
    Prophet/LSTM/Hybrid would add minutes to exercise one write call.
    """
    monkeypatch.setattr(
        rb, "build_models",
        lambda config, freq, season_length: {
            "Naive": SeasonalNaiveForecaster(season_length=season_length)
        },
    )
    monkeypatch.setattr(rb, "RESULTS_DIR", tmp_path)
    with open(rb.CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _baselines_dataset(series) -> SimpleNamespace:
    return SimpleNamespace(series=series, n=len(series), freq="D")


def test_baselines_all_zero_window_really_does_make_mape_nan(tmp_path, monkeypatch):
    """The trigger is real, not stipulated.

    If this ever stops holding, the sweep tests below would be exercising
    an imaginary failure mode and must be rewritten rather than patched.
    """
    config = _stub_baselines(monkeypatch, tmp_path)

    with pytest.raises(ResultWriteError) as excinfo:
        rb.run_dataset("toy_zero", _baselines_dataset(_zero_series()), config)

    assert excinfo.value.offenders == ["metrics.Naive.mape = nan"]
    assert "toy_zero_results.json" in str(excinfo.value)


def test_baselines_happy_path_write_is_byte_identical_to_pre_fix(tmp_path, monkeypatch):
    config = _stub_baselines(monkeypatch, tmp_path)

    output = rb.run_dataset("toy", _baselines_dataset(_good_series()), config)

    assert output["failed_models"] == {}  # otherwise metrics is empty and this is vacuous
    assert (tmp_path / "toy_results.json").read_bytes() == _pre_fix_bytes(tmp_path, output)


def _run_baselines_sweep(tmp_path, monkeypatch, capsys):
    """A two-dataset sweep whose FIRST dataset cannot be written.

    Ordering matters: the unwritable dataset comes first, so "the good
    file exists" is only true if the loop survived the failure.
    """
    _stub_baselines(monkeypatch, tmp_path)  # main() re-reads the real config itself
    monkeypatch.setattr(rb, "load_all", lambda: {
        "toy_zero": _baselines_dataset(_zero_series()),
        "toy_good": _baselines_dataset(_good_series()),
    })
    monkeypatch.setattr(sys, "argv", ["run_baselines.py"])

    rb.main()

    return capsys.readouterr().out


def test_baselines_sweep_continues_past_an_unwritable_dataset(tmp_path, monkeypatch, capsys):
    out = _run_baselines_sweep(tmp_path, monkeypatch, capsys)

    # The dataset after the failure ran and was written.
    assert (tmp_path / "toy_good_results.json").exists()
    assert json.loads((tmp_path / "toy_good_results.json").read_text())["dataset"] == "toy_good"
    # And the failure was announced loudly enough to act on: which dataset,
    # which model, which metric.
    assert "toy_zero" in out
    assert "metrics.Naive.mape = nan" in out


def test_baselines_leaves_no_file_for_the_unwritable_dataset(tmp_path, monkeypatch, capsys):
    _run_baselines_sweep(tmp_path, monkeypatch, capsys)

    # Not a truncated or half-encoded file: no file at all.
    assert not (tmp_path / "toy_zero_results.json").exists()
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["toy_good_results.json"]


def test_baselines_repeats_the_failure_in_the_final_summary(tmp_path, monkeypatch, capsys):
    out = _run_baselines_sweep(tmp_path, monkeypatch, capsys)

    tail = _after_summary_banner(out)
    assert "toy_zero" in tail
    assert "metrics.Naive.mape = nan" in tail


# --- run_size_scaling.py -----------------------------------------------------


class _StubForecast:
    def __init__(self, point):
        self.point = point


class _StubChronos:
    """Stand-in for ChronosForecaster, which run_one imports lazily.

    The real class needs torch and chronos-forecasting, neither installed
    on a CPU-only machine. Substituting it leaves run_one -- the code
    under test -- completely real. Predictions are `actual + 1`, so on
    the all-zero window RMSE is a finite 1.0 while MAPE is NaN.
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
    monkeypatch.setattr(rss, "BASELINE_DIR", tmp_path)  # no baseline: arima_rmse is None


def _scaling_dataset(series) -> SimpleNamespace:
    return SimpleNamespace(series=series, n=len(series))


def test_size_scaling_all_zero_window_really_does_make_mape_nan(tmp_path, monkeypatch):
    _stub_size_scaling(monkeypatch, tmp_path)

    with pytest.raises(ResultWriteError) as excinfo:
        rss.run_one("toy_zero", _scaling_dataset(_zero_series()), "tiny")

    assert excinfo.value.offenders == ["metrics.Chronos-tiny.mape = nan"]
    assert "toy_zero_chronos_tiny.json" in str(excinfo.value)


def test_size_scaling_happy_path_write_is_byte_identical_to_pre_fix(tmp_path, monkeypatch):
    _stub_size_scaling(monkeypatch, tmp_path)

    output = rss.run_one("toy", _scaling_dataset(_good_series()), "tiny")

    assert output["failed_models"] == {}
    written = (tmp_path / "toy_chronos_tiny.json").read_bytes()
    assert written == _pre_fix_bytes(tmp_path, output)


def _run_size_scaling_sweep(tmp_path, monkeypatch, capsys):
    _stub_size_scaling(monkeypatch, tmp_path)
    monkeypatch.setattr(rss, "load_all", lambda: {
        "toy_zero": _scaling_dataset(_zero_series()),
        "toy_good": _scaling_dataset(_good_series()),
    })
    monkeypatch.setattr(sys, "argv", ["run_size_scaling.py", "--sizes", "tiny"])

    rss.main()

    return capsys.readouterr().out


def test_size_scaling_sweep_continues_past_an_unwritable_run(tmp_path, monkeypatch, capsys):
    out = _run_size_scaling_sweep(tmp_path, monkeypatch, capsys)

    assert (tmp_path / "toy_good_chronos_tiny.json").exists()
    assert "toy_zero" in out
    assert "metrics.Chronos-tiny.mape = nan" in out


def test_size_scaling_leaves_no_file_for_the_unwritable_run(tmp_path, monkeypatch, capsys):
    _run_size_scaling_sweep(tmp_path, monkeypatch, capsys)

    assert not (tmp_path / "toy_zero_chronos_tiny.json").exists()
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["toy_good_chronos_tiny.json"]


def test_size_scaling_repeats_the_failure_in_the_final_summary(tmp_path, monkeypatch, capsys):
    """summarize() reads files off disk, so it cannot see this failure.

    A run that wrote nothing contributes to neither a stratum nor the
    per-size `failed` count -- print_summary shows "1 series" and no
    failures at all. The closing block is the only place the missing
    result is visible, which is why it must come after print_summary.
    """
    out = _run_size_scaling_sweep(tmp_path, monkeypatch, capsys)

    assert "SIZE-SCALING SUMMARY" in out
    tail = _after_summary_banner(out)
    assert "toy_zero" in tail
    assert "metrics.Chronos-tiny.mape = nan" in tail
    # The banner really is after the stratified summary, not before it.
    assert out.index("SIZE-SCALING SUMMARY") < out.index("RESULT FILE(S) NOT WRITTEN")

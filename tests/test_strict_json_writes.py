"""Every result file this repo writes must be valid strict JSON -- at no cost to a run.

The result-writing scripts pass `allow_nan=False`, so a non-finite metric
raises at write time instead of silently emitting a bare `NaN` literal.
That write sits outside the per-model `try/except`, and the `main()` loops
were unguarded, so the `ValueError` propagated out of the loop and ended
the run. The cost is not uniform: run_size_scaling.py and run_arima_only.py
skip existing files and resume, but run_baselines.py (35 datasets x 7
models) and run_chronos.py have no skip-existing at all, so an abort at
dataset 20 discarded datasets 1-19 as well.

Both properties have to hold at once, which is what these tests pin: no
silent invalid JSON, AND no lost sweep. Concretely, for each of the four
sweep scripts -- run_baselines.py, run_size_scaling.py, run_chronos.py and
run_arima_only.py -- the error names the dataset and the offending metric,
the failing dataset leaves no file behind, the sweep reaches the next
dataset, and the failure is repeated in the final summary where it cannot
scroll past. For the two converted first (run_baselines.py and
run_size_scaling.py) the successful write is additionally pinned
byte-for-byte against the pre-fix `json.dump`.

Two further writers need different treatment from those four and from each
other, and are covered in the last two sections:

  - run_ablation.py is the only script that produces a failure marker BY
    DESIGN, so refusing to write would discard every model that succeeded
    in the same run. It records a failed model as JSON `null` instead. The
    tests pin the representation, the console formatting that a `None`
    would otherwise crash, the guard that turns an unexpected non-finite
    RMSE into that same marker, and the `allow_nan=False` backstop behind
    it -- which, like every other writer in the repo, serialises before it
    opens the file, so a rejected payload cannot truncate the previous
    run's artifact.
  - patch_scaling_baselines.py rewrites 1,840 already-committed files in a
    loop, so the "then there is no file at all" objection does not apply --
    serialising before the open is precisely what keeps an expensive
    original intact when its rewrite is rejected. The tests pin that, and
    that one rejected file neither aborts the pass nor is counted as
    patched.

The non-finite value is produced the way it actually occurs in this
codebase, not injected: `compute_metrics`' docstring names an all-zero
`actual` window (intermittent demand, as in M5) as making MAPE NaN,
because the `actual != 0` mask is then empty and `np.mean([])` is NaN.
`_zero_series` builds exactly that window and every layer above it --
the model, `compute_metrics`, `run_dataset`/`run_one` -- runs for real.
(numpy emits a "Mean of empty slice" RuntimeWarning on that path; it is
part of the real behaviour being reproduced, not test noise.)

That route does not exist in run_ablation.py, which computes RMSE itself
via sklearn, and sklearn rejects non-finite *input*. Its real route to a
non-finite RMSE is overflow: predictions that are finite but enormous, as
a diverging LSTM or Hybrid produces, make `mean_squared_error` return
`inf` without raising anything -- exactly the "no exception fired and the
number is still meaningless" case its guard exists for. Exactly one
assertion in this module injects rather than triggers -- the one pinning
that `allow_nan=False` is still armed behind that guard, which by
construction nothing can reach while the guard works -- and it says so.
"""
import importlib
import json
import re
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import experiments.scripts.patch_scaling_baselines as psb
import experiments.scripts.run_ablation as ab
import experiments.scripts.run_arima_only as rao
import experiments.scripts.run_baselines as rb
import experiments.scripts.run_size_scaling as rss
from shortseq.evaluation.metrics import compute_metrics, without_residuals
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


def _dataset(series) -> SimpleNamespace:
    """The shape `load_all()` hands the scripts that read `dataset.freq`.

    run_size_scaling.py's `run_one` never reads `freq`, so it keeps its own
    narrower `_scaling_dataset` rather than being given a field its code
    path does not touch.
    """
    return SimpleNamespace(series=series, n=len(series), freq="D")


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


def test_baselines_all_zero_window_really_does_make_mape_nan(tmp_path, monkeypatch):
    """The trigger is real, not stipulated.

    If this ever stops holding, the sweep tests below would be exercising
    an imaginary failure mode and must be rewritten rather than patched.
    """
    config = _stub_baselines(monkeypatch, tmp_path)

    with pytest.raises(ResultWriteError) as excinfo:
        rb.run_dataset("toy_zero", _dataset(_zero_series()), config)

    assert excinfo.value.offenders == ["metrics.Naive.mape = nan"]
    assert "toy_zero_results.json" in str(excinfo.value)


def test_baselines_happy_path_write_is_byte_identical_to_pre_fix(tmp_path, monkeypatch):
    config = _stub_baselines(monkeypatch, tmp_path)

    output = rb.run_dataset("toy", _dataset(_good_series()), config)

    assert output["failed_models"] == {}  # otherwise metrics is empty and this is vacuous
    assert (tmp_path / "toy_results.json").read_bytes() == _pre_fix_bytes(tmp_path, output)


def _run_baselines_sweep(tmp_path, monkeypatch, capsys):
    """A two-dataset sweep whose FIRST dataset cannot be written.

    Ordering matters: the unwritable dataset comes first, so "the good
    file exists" is only true if the loop survived the failure.
    """
    _stub_baselines(monkeypatch, tmp_path)  # main() re-reads the real config itself
    monkeypatch.setattr(rb, "load_all", lambda: {
        "toy_zero": _dataset(_zero_series()),
        "toy_good": _dataset(_good_series()),
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


# --- run_chronos.py ----------------------------------------------------------


class _StubModel:
    """Deterministic stand-in for any forecaster, shared by the sections below.

    Predicts `actual + offset`, so on `_zero_series` RMSE is a finite
    `offset` while MAPE is NaN -- the almost-entirely-valid payload the
    guard exists for. Substituting the model leaves the `run_dataset` under
    test entirely real; only the fit cost is replaced, and for run_chronos.py
    the Chronos half is not installable on a CPU-only machine at all.
    """

    def __init__(self, name: str, offset: float = 1.0):
        self.name = name
        self.offset = offset
        self.train_time_ = 0.0
        self.pred_time_ = 0.0

    def fit(self, train):
        return self

    def predict_rolling(self, test):
        return _StubForecast(np.asarray(test.values, dtype=float) + self.offset)


def _import_run_chronos(monkeypatch):
    """Import run_chronos with its Chronos dependency stubbed out.

    run_size_scaling.py imports ChronosForecaster lazily inside run_one
    precisely so the module stays importable on a CPU-only machine;
    run_chronos.py imports it at module scope, so the module cannot be
    imported at all where torch/chronos-forecasting are absent. Injecting a
    stub module into sys.modules before the import keeps run_dataset and
    main() -- the code under test -- entirely real. run_chronos is dropped
    from sys.modules first so a genuine import cached by another test cannot
    be handed back instead; monkeypatch restores both entries afterwards.
    (Same harness as tests/test_keep_residuals.py, deliberately.)
    """
    module = types.ModuleType("shortseq.models.foundation.chronos")
    module.ChronosForecaster = _StubModel
    monkeypatch.setitem(sys.modules, "shortseq.models.foundation.chronos", module)
    monkeypatch.delitem(sys.modules, "experiments.scripts.run_chronos", raising=False)
    return importlib.import_module("experiments.scripts.run_chronos")


def _stub_chronos_script(monkeypatch, tmp_path):
    """run_chronos with both of its models replaced by cheap deterministic ones.

    Different offsets so the two `metrics` entries stay distinguishable and
    the inline DM test has a non-degenerate pair to compare.
    """
    rc = _import_run_chronos(monkeypatch)
    monkeypatch.setattr(rc, "ARIMAForecaster", lambda: _StubModel("ARIMA", 1.0))
    monkeypatch.setattr(rc, "ChronosForecaster", lambda size: _StubModel(f"Chronos-{size}", 2.0))
    monkeypatch.setattr(rc, "RESULTS_DIR", tmp_path)
    return rc


def test_chronos_all_zero_window_really_does_make_mape_nan(tmp_path, monkeypatch):
    """The trigger is real, not stipulated.

    Both models land on the same window, so both metrics blocks go
    non-finite together -- the error has to name each of them.
    """
    rc = _stub_chronos_script(monkeypatch, tmp_path)

    with pytest.raises(ResultWriteError) as excinfo:
        rc.run_dataset("toy_zero", _dataset(_zero_series()), "tiny")

    assert excinfo.value.offenders == [
        "metrics.ARIMA.mape = nan",
        "metrics.Chronos-tiny.mape = nan",
    ]
    assert "toy_zero_chronos_tiny_results.json" in str(excinfo.value)


def _run_chronos_sweep(tmp_path, monkeypatch, capsys):
    """A two-dataset sweep whose FIRST dataset cannot be written.

    Ordering matters: main() iterates the datasets dict in order, so "the
    good file exists" is only true if the loop survived the failure.
    """
    rc = _stub_chronos_script(monkeypatch, tmp_path)
    monkeypatch.setattr(rc, "load_all", lambda: {
        "toy_zero": _dataset(_zero_series()),
        "toy_good": _dataset(_good_series()),
    })
    monkeypatch.setattr(sys, "argv", ["run_chronos.py", "--size", "tiny"])

    rc.main()

    return capsys.readouterr().out


def test_chronos_sweep_continues_past_an_unwritable_dataset(tmp_path, monkeypatch, capsys):
    out = _run_chronos_sweep(tmp_path, monkeypatch, capsys)

    # This script has no skip-existing, so an abort here would have thrown
    # away every dataset already completed -- on GPU time.
    written = tmp_path / "toy_good_chronos_tiny_results.json"
    assert written.exists()
    assert json.loads(written.read_text())["dataset"] == "toy_good"
    assert "toy_zero" in out
    assert "metrics.ARIMA.mape = nan" in out


def test_chronos_leaves_no_file_for_the_unwritable_dataset(tmp_path, monkeypatch, capsys):
    _run_chronos_sweep(tmp_path, monkeypatch, capsys)

    # Not a truncated or half-encoded file: no file at all.
    assert not (tmp_path / "toy_zero_chronos_tiny_results.json").exists()
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["toy_good_chronos_tiny_results.json"]


def test_chronos_repeats_the_failure_in_the_final_summary(tmp_path, monkeypatch, capsys):
    out = _run_chronos_sweep(tmp_path, monkeypatch, capsys)

    tail = _after_summary_banner(out)
    assert "toy_zero" in tail
    assert "metrics.ARIMA.mape = nan" in tail
    # After the closing "Done." line, not buried above it.
    assert out.index("Done.") < out.index("RESULT FILE(S) NOT WRITTEN")


# --- run_arima_only.py -------------------------------------------------------


def _stub_arima_only(monkeypatch, tmp_path):
    monkeypatch.setattr(rao, "ARIMAForecaster", lambda: _StubModel("ARIMA"))
    monkeypatch.setattr(rao, "RESULTS_DIR", tmp_path)


def test_arima_only_all_zero_window_really_does_make_mape_nan(tmp_path, monkeypatch):
    _stub_arima_only(monkeypatch, tmp_path)

    with pytest.raises(ResultWriteError) as excinfo:
        rao.run_dataset("toy_zero", _dataset(_zero_series()))

    assert excinfo.value.offenders == ["metrics.ARIMA.mape = nan"]
    assert "toy_zero_results.json" in str(excinfo.value)


def _run_arima_only_sweep(tmp_path, monkeypatch, capsys):
    """A two-dataset backfill whose FIRST dataset cannot be written.

    main() sorts `todo` shortest-first, so the unwritable series is made one
    point shorter to put it first by the script's own ordering rule rather
    than by relying on sort stability.
    """
    _stub_arima_only(monkeypatch, tmp_path)
    monkeypatch.setattr(rao, "load_all", lambda: {
        "toy_zero": _dataset(_zero_series(N_POINTS - 1)),
        "toy_good": _dataset(_good_series()),
    })
    monkeypatch.setattr(sys, "argv", ["run_arima_only.py"])

    rao.main()

    return capsys.readouterr().out


def test_arima_only_sweep_continues_past_an_unwritable_dataset(tmp_path, monkeypatch, capsys):
    out = _run_arima_only_sweep(tmp_path, monkeypatch, capsys)

    written = tmp_path / "toy_good_results.json"
    assert written.exists()
    assert json.loads(written.read_text())["dataset"] == "toy_good"
    assert "toy_zero" in out
    assert "metrics.ARIMA.mape = nan" in out


def test_arima_only_leaves_no_file_for_the_unwritable_dataset(tmp_path, monkeypatch, capsys):
    _run_arima_only_sweep(tmp_path, monkeypatch, capsys)

    # A partial file here would be worse than none: the backfill skips any
    # dataset whose result file already exists, so a truncated leftover
    # would be treated as done and never re-run.
    assert not (tmp_path / "toy_zero_results.json").exists()
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["toy_good_results.json"]


def test_arima_only_repeats_the_failure_in_the_final_summary(tmp_path, monkeypatch, capsys):
    out = _run_arima_only_sweep(tmp_path, monkeypatch, capsys)

    tail = _after_summary_banner(out)
    assert "toy_zero" in tail
    assert "metrics.ARIMA.mape = nan" in tail
    assert out.index("Done.") < out.index("RESULT FILE(S) NOT WRITTEN")


# --- run_ablation.py ---------------------------------------------------------
#
# A different problem from the six write paths above, and it wants the
# opposite answer. This script produces a failure marker BY DESIGN -- a
# model that raises has no RMSE -- and the run still has to report the
# models that succeeded on the same windows, so "write nothing at all" is
# the wrong trade here. The marker therefore has to be something strict
# JSON can represent, and everything downstream of it (`round`, the
# `<12.2f` console cell) has to survive it.

# train_sizes tops out at 144 and test_n is 37, so the series must be at
# least 181 points long for the last window to be full.
N_ABLATION_POINTS = 200


def _ablation_series(n: int = N_ABLATION_POINTS) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(100.0 + np.arange(n, dtype=float), index=idx)


class _FailingModel:
    """A model that raises during fit, as ARIMA does on a degenerate window."""

    def fit(self, train):
        raise RuntimeError("fit did not converge")


class _DivergingModel:
    """Finite predictions so large that squaring them overflows to inf.

    The realistic no-exception route to a non-finite RMSE here.
    `mean_squared_error` validates its *inputs* for finiteness and 1e200
    passes that check, but (1e200)**2 is past float64's range, so the mean
    is `inf` and `np.sqrt(inf)` is `inf` -- nothing raises anywhere. A
    diverging LSTM or Hybrid arm is how a real run gets there.
    """

    def fit(self, train):
        return self

    def predict_rolling(self, test):
        return _StubForecast(np.full(len(test), 1e200))


def _stub_ablation(monkeypatch, tmp_path, **overrides):
    """run_ablation with its models and its loader replaced, nothing else.

    The window loop, the real sklearn RMSE, the console formatting and the
    write are all the code under test. Each override replaces one builder
    by model name, e.g. `XGBoost=_FailingModel`.
    """
    monkeypatch.setattr(ab, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ab, "load_dmart", lambda: {
        "dmart_food": SimpleNamespace(series=_ablation_series())
    })
    builders = {
        "ARIMA": lambda: _StubModel("ARIMA", 1.0),
        "XGBoost": lambda: _StubModel("XGBoost", 2.0),
        "Hybrid": lambda: _StubModel("Hybrid", 3.0),
    }
    builders.update(overrides)
    for model_name, builder in builders.items():
        monkeypatch.setattr(ab, f"{model_name}Forecaster", builder)


def _strict_loads(path):
    """json.loads, minus the non-standard NaN/Infinity extensions.

    Python's decoder accepts a bare `NaN` literal by default, so a plain
    round-trip would pass on exactly the file this script used to write.
    `parse_constant` is the hook that makes the read actually strict.
    """
    def reject(name):
        raise AssertionError(f"non-strict JSON constant in output: {name}")

    return json.loads(path.read_text(), parse_constant=reject)


def test_ablation_successful_run_writes_strict_parseable_json(tmp_path, monkeypatch, capsys):
    _stub_ablation(monkeypatch, tmp_path)

    ab.run_ablation()

    data = _strict_loads(tmp_path / "ablation_results.json")
    # `results` is keyed by n_train, an int, and json.dump stringifies int
    # keys silently -- a reader gets "50", not 50. Pinned so the comment in
    # the script is checkable rather than merely claimed.
    assert sorted(data) == ["110", "144", "50", "80"]
    for key, row in data.items():
        assert row["n_train"] == int(key)
        assert row["ARIMA"] == pytest.approx(1.0)
        assert row["XGBoost"] == pytest.approx(2.0)
        assert row["Hybrid"] == pytest.approx(3.0)


def test_ablation_records_a_failed_model_as_null_without_losing_the_others(tmp_path, monkeypatch, capsys):
    _stub_ablation(monkeypatch, tmp_path, XGBoost=_FailingModel)

    results = ab.run_ablation()

    out = capsys.readouterr().out
    # A model that raised is reported as an error and NOT also as a diverging
    # one. The two land on the same JSON marker on purpose, so the console is
    # the only place they stay distinguishable -- and only one of them means
    # there is a traceback to go and read.
    assert "Error: fit did not converge" in out
    assert "Non-finite" not in out
    data = _strict_loads(tmp_path / "ablation_results.json")
    assert len(data) == 4  # every window still ran
    for n_train in (50, 80, 110, 144):
        # null, not 0.0 (which reads as a perfect forecast) and not an
        # absent key (which reads as "not attempted").
        assert results[n_train]["XGBoost"] is None
        assert data[str(n_train)]["XGBoost"] is None
        # The models that succeeded on the same window are still reported.
        assert data[str(n_train)]["ARIMA"] == pytest.approx(1.0)
        assert data[str(n_train)]["Hybrid"] == pytest.approx(3.0)


def test_ablation_console_table_still_lines_up_when_a_model_fails(tmp_path, monkeypatch, capsys):
    """`f"{None:<12.2f}"` is a TypeError and a bare "FAIL" is 8 columns short.

    The table is the only thing an operator watches during the run, so the
    failure cell has to occupy the same width as a number.
    """
    _stub_ablation(monkeypatch, tmp_path, XGBoost=_FailingModel)

    ab.run_ablation()

    lines = capsys.readouterr().out.splitlines()
    rule = next(line for line in lines if line and set(line) == {"-"})
    rows = [line for line in lines if line.split(" ")[0] in ("50", "80", "110", "144")]
    assert len(rows) == 4
    for row in rows:
        assert len(row) == len(rule)


@pytest.mark.filterwarnings("ignore:overflow encountered")
def test_ablation_non_finite_rmse_is_recorded_as_a_failure_with_its_own_warning(tmp_path, monkeypatch, capsys):
    _stub_ablation(monkeypatch, tmp_path, Hybrid=_DivergingModel)

    results = ab.run_ablation()

    out = capsys.readouterr().out
    for n_train in (50, 80, 110, 144):
        # Named, so the operator can tell which arm diverged and where.
        assert f"[Hybrid n={n_train}]" in out
        assert results[n_train]["Hybrid"] is None
    # Distinct from the message a model that raised gets: nothing raised here,
    # and "the model errored" would send an operator looking for a traceback
    # that does not exist.
    assert "Error:" not in out
    assert "non-finite" in out.lower()
    data = _strict_loads(tmp_path / "ablation_results.json")
    assert data["144"]["Hybrid"] is None
    assert data["144"]["ARIMA"] == pytest.approx(1.0)


@pytest.mark.filterwarnings("ignore:overflow encountered")
def test_ablation_allow_nan_false_is_still_armed_behind_the_guard(tmp_path, monkeypatch, capsys):
    """The one payload-level injection in this module, and why it has to be one.

    The guard above converts every non-finite RMSE to `None`, so by
    construction nothing non-finite can reach the encoder while the guard
    works -- there is no realistic input that exercises `allow_nan=False`
    behind it. It is a backstop against a future edit that opens a fourth
    route to the payload, so the only honest way to test it is to disable
    the guard and check the backstop still fires. `math` is rebound on the
    run_ablation module, not on the real math module, so the blast radius
    is one name in one namespace.

    Two properties, and the second is the one this write path actually
    adds: the backstop raises, AND the previous run's artifact is still
    there afterwards.
    """
    _stub_ablation(monkeypatch, tmp_path, Hybrid=_DivergingModel)
    path = tmp_path / "ablation_results.json"
    # A previous run's results. This script rewrites the same path on every
    # invocation, so this file is precisely what a truncate-then-encode
    # write destroys -- `open(..., "w")` truncates on entry and the encoder
    # only reaches the offending value part-way through the walk.
    path.write_text(json.dumps({"50": {"n_train": 50, "ARIMA": 1.0}}, indent=2))
    before = path.read_bytes()
    monkeypatch.setattr(ab, "math", SimpleNamespace(isfinite=lambda value: True))

    with pytest.raises(ValueError, match="not JSON compliant"):
        ab.run_ablation()

    assert path.read_bytes() == before
    # And in particular no bare `Infinity` literal reached the file, which
    # is what the encoder writes by default and what no strict JSON parser
    # will read back.
    assert "Infinity" not in path.read_text()


# --- patch_scaling_baselines.py ----------------------------------------------
#
# The opposite case to run_ablation.py: this rewrites files that already
# exist, 1,840 of them in one loop, so "then there is no file" costs
# nothing and truncate-then-encode costs a GPU-hour. The failure it has to
# survive is not a model failing, it is an input file it cannot re-serialise.


def _metrics_for(series: pd.Series) -> dict:
    """Real `compute_metrics` output for a window, residuals stripped.

    Fed `_zero_series`, `mape` is NaN by the route documented at the top of
    this module; fed `_good_series`, every metric is finite.
    """
    actual = np.asarray(series.values, dtype=float)
    return without_residuals(compute_metrics(actual, actual + 1.0, "Chronos-tiny", 0.0, 0.0))


def _legacy_scaling_file(path, dataset: str, series: pd.Series) -> None:
    """A scaling result file exactly as it was written before 6a6b4cf.

    `json.dump` with the default `allow_nan=True` -- the call every script
    in this repo inherited until 6a6b4cf -- emits a bare `NaN` literal, and
    `json.load` reads it straight back as `float('nan')` without complaint.
    So a corpus written by an older checkout hands this script a payload it
    cannot re-serialise strictly, through no fault of the payload it is
    adding. That is the realistic failure, and it is reproduced rather than
    stipulated: nothing here types in a NaN.
    """
    with open(path, "w") as f:
        json.dump({
            "dataset": dataset,
            "chronos_size": "tiny",
            "arima_rmse": None,
            "metrics": {"Chronos-tiny": _metrics_for(series)},
        }, f, indent=2)


def _stub_patch_scaling(monkeypatch, tmp_path, *extra_argv):
    """Two scaling files, the FIRST of which cannot be re-serialised.

    The loop walks `sorted(glob(...))`, so `a_bad` is reached before
    `b_good` and "the good file was patched" is only true if the pass
    survived the failure.
    """
    scaling = tmp_path / "scaling"
    baselines = tmp_path / "baselines"
    scaling.mkdir()
    baselines.mkdir()
    monkeypatch.setattr(psb, "SCALING_DIR", scaling)
    monkeypatch.setattr(psb, "BASELINE_DIR", baselines)
    _legacy_scaling_file(scaling / "a_bad_chronos_tiny.json", "a_bad", _zero_series())
    _legacy_scaling_file(scaling / "b_good_chronos_tiny.json", "b_good", _good_series())
    write_result_json(baselines / "a_bad_results.json", {"metrics": {"ARIMA": {"rmse": 11.0}}})
    write_result_json(baselines / "b_good_results.json", {"metrics": {"ARIMA": {"rmse": 22.0}}})
    monkeypatch.setattr(sys, "argv", ["patch_scaling_baselines.py", *extra_argv])
    return scaling


def test_patch_leaves_the_expensive_original_intact_when_its_rewrite_is_rejected(tmp_path, monkeypatch, capsys):
    """The whole reason this script is worth converting.

    `open(path, "w")` truncates on entry and `json.dump` streams as it
    walks, so a rejected payload used to leave a partial file where a
    committed scaling result had been -- one that cost real GPU time and
    that no re-run of this script can reconstruct.
    """
    scaling = _stub_patch_scaling(monkeypatch, tmp_path)
    before = (scaling / "a_bad_chronos_tiny.json").read_bytes()

    psb.main()

    assert (scaling / "a_bad_chronos_tiny.json").read_bytes() == before


def test_patch_continues_past_a_file_it_cannot_rewrite(tmp_path, monkeypatch, capsys):
    scaling = _stub_patch_scaling(monkeypatch, tmp_path)

    psb.main()

    assert json.loads((scaling / "b_good_chronos_tiny.json").read_text())["arima_rmse"] == 22.0
    out = capsys.readouterr().out
    assert "a_bad_chronos_tiny.json" in out
    assert "metrics.Chronos-tiny.mape = nan" in out


def test_patch_rewrites_only_the_arima_rmse_field(tmp_path, monkeypatch, capsys):
    scaling = _stub_patch_scaling(monkeypatch, tmp_path)
    before = json.loads((scaling / "b_good_chronos_tiny.json").read_text())

    psb.main()

    after = json.loads((scaling / "b_good_chronos_tiny.json").read_text())
    assert after == {**before, "arima_rmse": 22.0}


def test_patch_does_not_count_a_rejected_file_as_filled(tmp_path, monkeypatch, capsys):
    """The tally is what an operator acts on, so it must not include a no-op.

    One of the two files was patched; counting the rejected one as filled
    would report the corpus as fully backfilled when a third of it is not.
    """
    _stub_patch_scaling(monkeypatch, tmp_path)

    psb.main()

    assert "filled: 1" in capsys.readouterr().out


def test_patch_tally_accounts_for_every_file_the_pass_looked_at(tmp_path, monkeypatch, capsys):
    """A rejected file must be counted somewhere, not merely reported below.

    With four counters over this two-file corpus the tally summed to 1,
    and the missing file appeared only in the failure block underneath.
    Over 1,840 inputs that reads as a smaller corpus rather than as a
    rejection. What is pinned is the arithmetic, not the wording: every
    `label: <count>` line the pass prints ahead of the failure banner is
    summed, so a differently-phrased line still satisfies this.
    """
    scaling = _stub_patch_scaling(monkeypatch, tmp_path)
    n_inputs = len(list(scaling.glob("*_chronos_*.json")))

    psb.main()

    out = capsys.readouterr().out
    # Otherwise this passes vacuously if the fixture ever stops rejecting.
    assert "RESULT FILE(S) NOT WRITTEN" in out
    counts = re.findall(r"^\S[^\n]*: (\d+)$", out.split("RESULT FILE(S)")[0], re.M)
    assert sum(int(count) for count in counts) == n_inputs, out


def test_patch_repeats_every_unwritable_file_in_the_final_summary(tmp_path, monkeypatch, capsys):
    _stub_patch_scaling(monkeypatch, tmp_path)

    psb.main()

    tail = _after_summary_banner(capsys.readouterr().out)
    assert "a_bad_chronos_tiny.json" in tail
    assert "metrics.Chronos-tiny.mape = nan" in tail
    # The shared banner's closing advice is written for the sweeps, where a
    # rejected payload means no file exists. Here one does, and an operator
    # told to "re-run them" would otherwise assume those results were lost.
    assert "UNCHANGED on disk" in tail


def test_patch_dry_run_still_reports_both_files_and_writes_nothing(tmp_path, monkeypatch, capsys):
    """--dry-run must survive the record-and-continue restructure.

    It reports what *would* change, so it counts the rejected file too:
    the rewrite is never attempted, and whether the payload serialises is
    not something a dry run has any business deciding.
    """
    scaling = _stub_patch_scaling(monkeypatch, tmp_path, "--dry-run")
    before = {p.name: p.read_bytes() for p in scaling.glob("*.json")}

    psb.main()

    assert "would fill: 2" in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in scaling.glob("*.json")} == before

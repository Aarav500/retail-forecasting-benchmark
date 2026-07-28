"""Structural tests for experiments/scripts/run_size_scaling.py.

Covers the pure-Python summary/stratification logic on synthetic result
files. The GPU-dependent run_one() is not covered here — it needs torch
and a real model; it gets exercised on the GPU instance instead.

Stratification is the thing most worth pinning: the campaign's whole
point is that short and long series behave differently, so a summary
that silently pooled them would produce a confidently wrong finding.
"""
import json

import pytest

import experiments.scripts.run_size_scaling as rss


def _write_result(tmp_path, dataset, size, n, rmse, arima_rmse):
    """Write one synthetic result file in the schema run_one() emits."""
    payload = {
        "dataset": dataset,
        "size": size,
        "n": n,
        "stratum": "short" if n < rss.SHORT_MAX_N else "long",
        "n_train": int(n * 0.8),
        "n_test": n - int(n * 0.8),
        "arima_rmse": arima_rmse,
        "metrics": (
            {f"Chronos-{size}": {"rmse": rmse}} if rmse is not None else {}
        ),
        "failed_models": (
            {} if rmse is not None else {f"Chronos-{size}": "RuntimeError: CUDA OOM"}
        ),
    }
    path = tmp_path / f"{dataset}_chronos_{size}.json"
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def test_summarize_separates_short_and_long_strata(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "RESULTS_DIR", tmp_path)
    # short series: Chronos worse than ARIMA (ratio 1.5)
    _write_result(tmp_path, "short_a", "small", n=100, rmse=150.0, arima_rmse=100.0)
    _write_result(tmp_path, "short_b", "small", n=150, rmse=150.0, arima_rmse=100.0)
    # long series: Chronos better than ARIMA (ratio 0.5)
    _write_result(tmp_path, "long_a", "small", n=900, rmse=50.0, arima_rmse=100.0)

    summary = rss.summarize()

    assert summary["small"]["short"]["n_series"] == 2
    assert summary["small"]["short"]["mean_ratio"] == pytest.approx(1.5)
    assert summary["small"]["short"]["chronos_wins"] == 0
    assert summary["small"]["short"]["arima_wins"] == 2

    assert summary["small"]["long"]["n_series"] == 1
    assert summary["small"]["long"]["mean_ratio"] == pytest.approx(0.5)
    assert summary["small"]["long"]["chronos_wins"] == 1

    # The pooled mean would be 1.1667 — a number that describes neither
    # stratum. Assert it appears nowhere in the summary.
    assert "mean_ratio" not in summary["small"]
    assert "pooled" not in summary["small"]


def test_summarize_counts_failures_separately_from_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "RESULTS_DIR", tmp_path)
    _write_result(tmp_path, "ok", "large", n=100, rmse=90.0, arima_rmse=100.0)
    _write_result(tmp_path, "oom", "large", n=100, rmse=None, arima_rmse=100.0)

    summary = rss.summarize()

    # A failed (e.g. OOM) run must NOT be silently dropped, and must not
    # be counted as either a win or a loss.
    assert summary["large"]["failed"] == 1
    assert summary["large"]["short"]["n_series"] == 1
    assert summary["large"]["short"]["chronos_wins"] == 1


def test_summarize_counts_missing_baseline_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "RESULTS_DIR", tmp_path)
    _write_result(tmp_path, "nobase", "tiny", n=100, rmse=90.0, arima_rmse=None)

    summary = rss.summarize()

    # No ARIMA reference means no ratio can be computed; that's distinct
    # from both a win and a failure.
    assert summary["tiny"]["no_baseline"] == 1
    assert summary["tiny"]["short"] is None


def test_load_arima_rmse_returns_none_when_baseline_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "BASELINE_DIR", tmp_path)
    assert rss.load_arima_rmse("does_not_exist") is None


def test_load_arima_rmse_reads_committed_baseline_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "BASELINE_DIR", tmp_path)
    payload = {
        "dataset": "toy",
        "metrics": {"ARIMA": {"rmse": 123.45}, "LSTM": {"rmse": 200.0}},
        "dm_tests": {},
        "failed_models": {},
    }
    with open(tmp_path / "toy_results.json", "w") as f:
        json.dump(payload, f)
    assert rss.load_arima_rmse("toy") == pytest.approx(123.45)


def test_short_max_n_matches_phase_b_stratification_decision():
    # Pinned deliberately: the Phase B spec addendum fixes the boundary
    # at n < 200. Changing it silently would invalidate cross-campaign
    # comparisons.
    assert rss.SHORT_MAX_N == 200

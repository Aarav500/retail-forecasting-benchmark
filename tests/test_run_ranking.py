"""Tests for the ranking analysis script.

Focus is the stratification guarantee: short (n<200) and long (n>=200)
series must be ranked separately and never pooled. The two strata have
opposite-signed effects at the small end (tiny is 0.972 short vs 1.092
long), so a pooled ranking would average away the structure being
measured.
"""
import json

import pytest

import experiments.scripts.run_ranking as rr


def _write_scaling(tmp_path, dataset, size, n, rmse, arima_rmse=100.0):
    payload = {
        "dataset": dataset,
        "size": size,
        "n": n,
        "stratum": "short" if n < 200 else "long",
        "n_train": int(n * 0.8),
        "n_test": n - int(n * 0.8),
        "arima_rmse": arima_rmse,
        "metrics": {f"Chronos-{size}": {"rmse": rmse}},
        "failed_models": {},
    }
    with open(tmp_path / f"{dataset}_chronos_{size}.json", "w") as f:
        json.dump(payload, f)


def test_build_matrices_separates_strata(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for size, rmse in [("tiny", 90.0), ("small", 80.0), ("large", 70.0)]:
        _write_scaling(tmp_path, "s1", size, n=100, rmse=rmse)
        _write_scaling(tmp_path, "s2", size, n=150, rmse=rmse)
        _write_scaling(tmp_path, "L1", size, n=900, rmse=rmse)

    matrices = rr.build_matrices()

    assert set(matrices) == {"short", "long"}
    assert list(matrices["short"].index) == ["s1", "s2"]
    assert list(matrices["long"].index) == ["L1"]
    assert "pooled" not in matrices


def test_build_matrices_columns_are_sizes_in_scaling_order(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for size in ["large", "tiny", "small"]:  # deliberately unsorted on disk
        _write_scaling(tmp_path, "s1", size, n=100, rmse=50.0)
    matrices = rr.build_matrices()
    # Columns must follow parameter-count order, not filesystem order,
    # so the report reads tiny -> large.
    assert list(matrices["short"].columns) == ["tiny", "small", "large"]


def test_build_matrices_drops_incomplete_series(tmp_path, monkeypatch):
    # A series missing one size would make the grid incomplete, which the
    # Nemenyi CD formula does not tolerate. It must be dropped loudly,
    # not silently NaN-filled.
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for size in ["tiny", "small", "large"]:
        _write_scaling(tmp_path, "complete", size, n=100, rmse=50.0)
    _write_scaling(tmp_path, "partial", "tiny", n=100, rmse=50.0)

    matrices = rr.build_matrices()

    assert list(matrices["short"].index) == ["complete"]
    assert not matrices["short"].isna().any().any()


def test_analyse_stratum_returns_ranking_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for i in range(15):
        _write_scaling(tmp_path, f"s{i}", "tiny", n=100, rmse=90.0 + i)
        _write_scaling(tmp_path, f"s{i}", "small", n=100, rmse=80.0 + i)
        _write_scaling(tmp_path, f"s{i}", "large", n=100, rmse=70.0 + i)

    matrices = rr.build_matrices()
    payload = rr.analyse(matrices["short"])

    assert payload["n_series"] == 15
    assert payload["n_models"] == 3
    assert payload["mean_ranks"]["large"] < payload["mean_ranks"]["tiny"]
    assert "critical_difference" in payload

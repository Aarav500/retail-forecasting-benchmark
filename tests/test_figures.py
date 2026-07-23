from pathlib import Path

import pandas as pd

from shortseq.datasets.base import SeriesDataset
from shortseq.visualization.figures import (
    fig_master_heatmap, fig_regime_scatter, fig_real_residuals, fig_dm_summary,
)

FAKE_RESULTS = {
    "dmart_food": {
        "metrics": {
            "ARIMA": {"rmse": 100.0, "residual_std": 10.0},
            "Hybrid": {"rmse": 90.0, "residual_std": 9.0},
        },
        "dm_tests": {"Hybrid": {"p_value": 0.01}},
    },
    "walmart": {
        "metrics": {
            "ARIMA": {"rmse": 200.0, "residual_std": 20.0},
            "Hybrid": {"rmse": 150.0, "residual_std": 15.0},
        },
        "dm_tests": {"Hybrid": {"p_value": 0.2}},
    },
}
FAKE_DATASETS = {
    "dmart_food": SeriesDataset(
        name="dmart_food", series=pd.Series([1.0, 2.0, 3.0]), freq="D",
        n=3, cv=0.02, ac1=-0.1, zero_frac=0.0, real=True,
    ),
    "walmart": SeriesDataset(
        name="walmart", series=pd.Series([1.0, 2.0, 3.0]), freq="W",
        n=3, cv=0.3, ac1=0.5, zero_frac=0.0, real=False,
    ),
}


def test_fig_master_heatmap_writes_output_files(tmp_path):
    fig_master_heatmap(FAKE_RESULTS, tmp_path)
    assert (tmp_path / "fig1_master_heatmap.png").exists()
    assert (tmp_path / "fig1_master_heatmap.pdf").exists()


def test_fig_regime_scatter_writes_output_files(tmp_path):
    fig_regime_scatter(FAKE_RESULTS, FAKE_DATASETS, tmp_path)
    assert (tmp_path / "fig_regime_scatter.png").exists()


def test_fig_real_residuals_writes_output_files(tmp_path):
    fig_real_residuals(FAKE_RESULTS, ["dmart_food", "walmart"], tmp_path)
    assert (tmp_path / "fig_real_residuals.png").exists()


def test_fig_dm_summary_writes_output_files(tmp_path):
    fig_dm_summary(FAKE_RESULTS, ["dmart_food", "walmart"], tmp_path)
    assert (tmp_path / "fig_dm_summary.png").exists()

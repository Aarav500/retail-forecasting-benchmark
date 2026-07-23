"""Reproducible figure generation from saved results + the dataset registry.

Ported from `code/figures_v2.py`. See this plan's Task 12 scope note for
which functions were dropped and why.
"""
from pathlib import Path

import matplotlib


def _backend_already_configured() -> bool:
    """True if a backend has already been picked (by the caller, an rc
    file, or $MPLBACKEND) -- i.e. whether it is safe to default to "Agg"
    below without clobbering someone else's choice.

    NOTE: checking `"matplotlib.pyplot" in sys.modules` is NOT a valid
    proxy for this (verified: it produces a false negative). The normal
    calling convention is `matplotlib.use("TkAgg")` *before*
    `import matplotlib.pyplot`, and `matplotlib.use()` deliberately does
    not import pyplot when it isn't already imported (see its docstring/
    source) precisely to avoid forcing premature backend resolution --
    so pyplot can be absent from `sys.modules` even though the caller has
    already configured a backend. Instead we read the rcParam directly
    via the public `auto_select=False` accessor (matplotlib>=3.10), which
    returns the configured backend without triggering matplotlib's lazy
    auto-detection, or `None` if nothing has been configured yet.
    """
    try:
        return matplotlib.get_backend(auto_select=False) is not None
    except TypeError:
        # matplotlib < 3.10 lacks the `auto_select` kwarg. Fall back to
        # the private accessor `matplotlib.use()` itself relies on
        # internally for this exact "already set?" check.
        return matplotlib.rcParams._get_backend_or_none() is not None


# Default to a non-interactive backend only if nothing has configured a
# backend yet: this module only ever writes figures to disk, and the
# platform-default GUI backend (e.g. TkAgg) is not guaranteed to be
# usable/stable in headless environments (CI, worker processes) where no
# display is attached. Guarded so we never clobber a caller's own prior
# `matplotlib.use(...)` choice -- libraries should not call `.use()`
# unconditionally (see matplotlib's own guidance for library code).
if not _backend_already_configured():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shortseq.datasets.base import SeriesDataset

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})
COLORS = {
    "ARIMA": "#2196F3", "SARIMA": "#4CAF50", "Prophet": "#FF9800",
    "XGBoost": "#9C27B0", "LSTM": "#F44336", "Hybrid": "#795548", "Naive": "#607D8B",
}


def fig_master_heatmap(results: dict, figures_dir: Path):
    models = ["ARIMA", "SARIMA", "Prophet", "XGBoost", "LSTM", "Hybrid", "Naive"]
    datasets = list(results.keys())

    data = np.full((len(models), len(datasets)), np.nan)
    for j, d in enumerate(datasets):
        for i, m in enumerate(models):
            v = results.get(d, {}).get("metrics", {}).get(m, {}).get("rmse")
            if v is not None:
                data[i, j] = v

    with np.errstate(invalid="ignore"):
        col_max = np.nanmax(data, axis=0)
        norm = data / col_max

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.4), 5))
    im = ax.imshow(norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(datasets)):
            v = data[i, j]
            if not np.isnan(v):
                txt = f"{v:.0f}" if v > 100 else f"{v:.2f}"
                color = "black" if np.isnan(norm[i, j]) or norm[i, j] < 0.65 else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Normalised RMSE", fraction=0.046, pad=0.04)
    ax.set_title("Model Performance Across Demand Regimes\n(Normalised RMSE — lower is better)",
                 fontweight="bold", pad=20)
    plt.tight_layout()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "fig1_master_heatmap.pdf", bbox_inches="tight")
    plt.savefig(figures_dir / "fig1_master_heatmap.png", bbox_inches="tight")
    plt.close()


def fig_regime_scatter(results: dict, datasets: dict[str, SeriesDataset], figures_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cvs, ac1s, ratios_hybrid, labels_plot = [], [], [], []

    for name, ds in datasets.items():
        m = results.get(name, {}).get("metrics", {})
        arima_r = m.get("ARIMA", {}).get("rmse")
        hybrid_r = m.get("Hybrid", {}).get("rmse")
        if not arima_r or not hybrid_r:
            continue
        cvs.append(ds.cv)
        ac1s.append(ds.ac1)
        ratios_hybrid.append(hybrid_r / arima_r)
        labels_plot.append(name)

    ax1 = axes[0]
    ax1.scatter(cvs, ratios_hybrid, s=100, zorder=5, edgecolors="white", lw=1.2)
    for cv, r, lbl in zip(cvs, ratios_hybrid, labels_plot):
        ax1.annotate(lbl, (cv, r), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax1.axhline(1.0, color="#2196F3", lw=2, ls="--", label="ARIMA baseline")
    ax1.set_xlabel("Coefficient of Variation (CV = σ/μ)")
    ax1.set_ylabel("Hybrid RMSE / ARIMA RMSE")
    ax1.set_title("When Does Hybrid Beat ARIMA?\n(Ratio < 1 = Hybrid better)", fontweight="bold")
    ax1.legend(fontsize=9)

    ax2 = axes[1]
    ax2.scatter(ac1s, ratios_hybrid, s=100, zorder=5, edgecolors="white", lw=1.2)
    for ac1, r, lbl in zip(ac1s, ratios_hybrid, labels_plot):
        ax2.annotate(lbl, (ac1, r), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax2.axhline(1.0, color="#2196F3", lw=2, ls="--")
    ax2.set_xlabel("Lag-1 Autocorrelation AC(1)")
    ax2.set_ylabel("Hybrid RMSE / ARIMA RMSE")
    ax2.set_title("AC(1) vs Model Performance", fontweight="bold")

    plt.suptitle("Regime Characterization Across All Loaded Datasets", fontweight="bold", fontsize=12)
    plt.tight_layout()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "fig_regime_scatter.pdf", bbox_inches="tight")
    plt.savefig(figures_dir / "fig_regime_scatter.png", bbox_inches="tight")
    plt.close()


def fig_real_residuals(results: dict, dataset_names: list[str], figures_dir: Path):
    models = ["ARIMA", "SARIMA", "XGBoost", "LSTM", "Hybrid"]
    fig, axes = plt.subplots(1, len(dataset_names), figsize=(4 * len(dataset_names), 5))
    if len(dataset_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, dataset_names):
        m_data = results.get(name, {}).get("metrics", {})
        stds = [m_data.get(m, {}).get("residual_std", 0) for m in models]
        colors = [COLORS.get(m, "#607D8B") for m in models]
        bars = ax.bar(models, stds, color=colors, edgecolor="white")
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("Residual Std Dev")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right")
        for bar, v in zip(bars, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", fontsize=8)

    plt.suptitle("Residual Stability\n(Lower = more stable predictions)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "fig_real_residuals.pdf", bbox_inches="tight")
    plt.savefig(figures_dir / "fig_real_residuals.png", bbox_inches="tight")
    plt.close()


def fig_dm_summary(results: dict, dataset_names: list[str], figures_dir: Path):
    models = ["SARIMA", "Prophet", "XGBoost", "LSTM", "Hybrid"]
    fig, axes = plt.subplots(1, len(dataset_names), figsize=(4 * len(dataset_names), 4.5))
    if len(dataset_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, dataset_names):
        dm = results.get(name, {}).get("dm_tests", {})
        pvals = [dm.get(m, {}).get("p_value", 1.0) for m in models]
        colors = ["#4CAF50" if p < 0.05 else "#FF9800" for p in pvals]
        ax.bar(models, pvals, color=colors, edgecolor="white")
        ax.axhline(0.05, color="#F44336", lw=2, ls="--", label="α=0.05")
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("DM Test p-value")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.legend(fontsize=8)

    plt.suptitle("Diebold-Mariano Significance Tests vs ARIMA Baseline", fontweight="bold", fontsize=13)
    plt.tight_layout()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "fig_dm_summary.pdf", bbox_inches="tight")
    plt.savefig(figures_dir / "fig_dm_summary.png", bbox_inches="tight")
    plt.close()

"""D-Mart real retail data (India, daily) — 4 product-category series.

Ported from the loading logic in `code/real_experiment.py`.
"""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset

CATEGORIES = ["food", "electronics", "clothing", "furniture"]


def load_dmart() -> dict[str, SeriesDataset]:
    """Load the 4 real D-Mart category series (n=181 each, daily)."""
    out: dict[str, SeriesDataset] = {}
    for cat in CATEGORIES:
        df = pd.read_csv(DATA_DIR / f"real_{cat}.csv", parse_dates=["ds"])
        series = df.set_index("ds")["y"].astype(float).sort_index()
        name = f"dmart_{cat}"
        out[name] = make_dataset(name, series, freq="D", real=True)
    return out

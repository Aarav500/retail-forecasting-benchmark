"""Walmart-calibrated weekly data (US) — store 1, matching the paper's
single-series analysis ('Use store 1 as primary series' in
code/experiment.py:generate_walmart_data)."""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset


def load_walmart() -> dict[str, SeriesDataset]:
    """Load the Walmart-calibrated series (n=143, weekly, store 1)."""
    df = pd.read_csv(DATA_DIR / "walmart.csv", parse_dates=["ds"])
    df = df[df["store"] == 1]
    series = df.set_index("ds")["y"].astype(float).sort_index()
    return {"walmart": make_dataset("walmart", series, freq="W", real=False)}

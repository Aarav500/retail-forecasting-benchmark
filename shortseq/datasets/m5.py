"""M5-calibrated intermittent daily demand series."""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset


def load_m5() -> dict[str, SeriesDataset]:
    """Load the M5-calibrated series (n=365, daily, ~32.9% zero-demand)."""
    df = pd.read_csv(DATA_DIR / "m5.csv", parse_dates=["ds"])
    series = df.set_index("ds")["y"].astype(float).sort_index()
    return {"m5": make_dataset("m5", series, freq="D", real=False)}

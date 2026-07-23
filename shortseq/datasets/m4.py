"""M4 Micro Monthly series — 24 real series sampled across 3 CV bins.

`m4_monthly_train.csv` is an R-exported wide table (index_col=0 gives the
series id as the row label; remaining columns are the observations,
dropna-trimmed to each series' actual length). M4 doesn't publish real
calendar dates for these series, so a synthetic monthly index is used
purely so downstream code can rely on a DatetimeIndex — the values
themselves are the real M4 competition data.
"""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset


def load_m4() -> dict[str, SeriesDataset]:
    """Load the 24 sampled M4 Micro Monthly series (n=68-197, monthly)."""
    train_df = pd.read_csv(DATA_DIR / "m4_monthly_train.csv", index_col=0)
    sample = pd.read_csv(DATA_DIR / "m4_sample_ids.csv")

    out: dict[str, SeriesDataset] = {}
    for _, row in sample.iterrows():
        series_id = row["id"]
        values = train_df.loc[series_id].dropna().values.astype(float)
        index = pd.period_range("2000-01", periods=len(values), freq="M").to_timestamp()
        series = pd.Series(values, index=index)
        name = f"m4_{series_id}"
        out[name] = make_dataset(name, series, freq="M", real=True)
    return out

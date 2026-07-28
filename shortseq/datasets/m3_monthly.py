"""M3 Monthly series — 100 real series sampled from the M3 competition set.

Prepared by `experiments/scripts/prepare_datasets.py` (fixed seed 42) from
the `datasetsforecast` distribution of M3; the sampled ids are frozen in
`data/derived/m3_monthly_ids.csv` and one `ds,y` CSV per series lives in
`data/derived/m3_monthly/`.

Unlike M4, M3 publishes a starting year/month per series, so these `ds`
values are real dates rather than a synthetic index — normalised to
month-*start* by the preparation script, matching the month-start
convention the repo's freq="M" handling assumes (see `_FREQ_ALIASES` in
`shortseq/models/prophet_model.py`). A minority of upstream M3 series
carry a 1900-01 placeholder start rather than a genuine one; only the
spacing of the index matters to the forecasters, but the absolute dates
of those series should not be read as meaningful.
"""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset

DERIVED_DIR = DATA_DIR / "derived"
SOURCE_DIR = DERIVED_DIR / "m3_monthly"


def load_m3_monthly() -> dict[str, SeriesDataset]:
    """Load the 100 sampled M3 Monthly series (n=69-144, monthly)."""
    ids = pd.read_csv(DERIVED_DIR / "m3_monthly_ids.csv")["id"].astype(str)

    out: dict[str, SeriesDataset] = {}
    for series_id in ids:
        df = pd.read_csv(SOURCE_DIR / f"{series_id}.csv", parse_dates=["ds"])
        series = df.set_index("ds")["y"].astype(float).sort_index()
        name = f"m3m_{series_id}"
        out[name] = make_dataset(name, series, freq="M", real=True)
    return out

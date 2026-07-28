"""Rossmann real retail data (Germany, daily) — 50 sampled store series.

Prepared by `experiments/scripts/prepare_datasets.py` (fixed seed 42) from
the Kaggle `rossmann-store-sales` competition dump; the sampled store ids
are frozen in `data/derived/rossmann_ids.csv` and one `ds,y` CSV per store
lives in `data/derived/rossmann/`. `y` is daily `Sales` for that store over
2013-01-01..2015-07-31 (942 days).

Two shaping decisions worth knowing when reading results off these series:

**Closed-day zeros are kept.** Rossmann stores are shut on most Sundays,
which the source records as a real row with `Open=0, Sales=0` (a further 54
rows across the whole file are open days with genuinely zero sales). Those
rows are retained, so `zero_frac` sits around 0.17 for a typical store.
Dropping them was the alternative, and was rejected twice over: it would
leave a six-days-a-week index that is neither daily nor weekly and that
every index-blind baseline would silently mis-space, and it would throw away
precisely the zero-inflated demand this benchmark is built to study — the
same property that makes the M5 source (`m5.py`, ~32.9% zero days)
interesting. A closed Sunday is observed zero demand, not missing data.

**Only stores with a gap-free daily index are eligible.** 180 of the 1115
stores were closed for refurbishment for roughly the second half of 2014 and
are absent from the file for that window (758 rows rather than 942). Unlike
closed Sundays, that IS missing data, so those stores are excluded rather
than zero-filled; 935 remain in the sampling pool. Every series here
therefore has a contiguous daily DatetimeIndex.
"""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset

DERIVED_DIR = DATA_DIR / "derived"
SOURCE_DIR = DERIVED_DIR / "rossmann"


def load_rossmann() -> dict[str, SeriesDataset]:
    """Load the 50 sampled Rossmann store series (n=941-942, daily)."""
    ids = pd.read_csv(DERIVED_DIR / "rossmann_ids.csv")["id"].astype(str)

    out: dict[str, SeriesDataset] = {}
    for store_id in ids:
        df = pd.read_csv(SOURCE_DIR / f"{store_id}.csv", parse_dates=["ds"])
        series = df.set_index("ds")["y"].astype(float).sort_index()
        name = f"rossmann_store_{store_id}"
        out[name] = make_dataset(name, series, freq="D", real=True)
    return out

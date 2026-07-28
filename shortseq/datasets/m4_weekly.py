"""M4 Weekly series — 50 real series sampled from the M4 competition set.

Prepared by `experiments/scripts/prepare_datasets.py` (fixed seed 42) from
the `datasetsforecast` distribution of M4; the sampled ids are frozen in
`data/derived/m4_weekly_ids.csv` and one `step,y` CSV per series lives in
`data/derived/m4_weekly/`.

M4 doesn't publish real calendar dates for these series — upstream the
time column is a plain 1,2,3,... step counter — so, exactly as in the
existing M4 Micro Monthly loader (`m4.py`), a synthetic weekly index is
used purely so downstream code can rely on a DatetimeIndex. The values
themselves are the real M4 competition data.
"""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset

DERIVED_DIR = DATA_DIR / "derived"
SOURCE_DIR = DERIVED_DIR / "m4_weekly"


def load_m4_weekly() -> dict[str, SeriesDataset]:
    """Load the 50 sampled M4 Weekly series (n=93-2296, weekly).

    Note the wide length spread: M4 Weekly runs 93-2610 observations
    upstream, so this sample is only partly "short" by ShortSeq's usual
    standards. The design spec asks for an unfiltered random draw, so no
    length cap is applied here.
    """
    ids = pd.read_csv(DERIVED_DIR / "m4_weekly_ids.csv")["id"].astype(str)

    out: dict[str, SeriesDataset] = {}
    for series_id in ids:
        df = pd.read_csv(SOURCE_DIR / f"{series_id}.csv")
        # Sort by the upstream step counter rather than trusting CSV row
        # order, so the synthetic index below is chronological by
        # construction (same guard as the other loaders' `.sort_index()`).
        values = df.sort_values("step")["y"].astype(float).values
        # Sunday-anchored, matching pandas' default 'W' == 'W-SUN' and the
        # repo's other weekly sources (UCI, Walmart). This is not cosmetic:
        # Prophet's predict_rolling extrapolates future dates with
        # date_range(..., freq="W"), so a Monday-anchored index (what
        # period_range('...', freq='W').to_timestamp() would give) fails its
        # contiguity check. Same class of gotcha as `_FREQ_ALIASES` in
        # shortseq/models/prophet_model.py. 2000-01-02 was a Sunday.
        index = pd.date_range("2000-01-02", periods=len(values), freq="W")
        name = f"m4w_{series_id}"
        out[name] = make_dataset(name, pd.Series(values, index=index), freq="W", real=True)
    return out

"""Walmart real retail data (US, weekly) — 50 sampled store-dept series.

Prepared by `experiments/scripts/prepare_datasets.py` (fixed seed 42) from
the Kaggle `walmart-recruiting-store-sales-forecasting` competition dump;
the sampled `<store>_<dept>` ids are frozen in
`data/derived/walmart_real_ids.csv` and one `ds,y` CSV per pair lives in
`data/derived/walmart_real/`. `y` is `Weekly_Sales` for that store-department
over 2010-02-05..2012-10-26 (143 weeks of full history).

Keyed `walmart_real_<store>_<dept>`, NOT `walmart_*`: the bare `walmart` key
already belongs to the older single calibrated series in `walmart.py`, which
is `real=False`. These are the real competition numbers and must not be
conflated with it.

**Eligibility.** Only store-dept pairs with a gap-free weekly index of at
least 100 weeks are in the sampling pool (2671 of 3331). The remainder are
departments that opened or closed mid-window, some with a single
observation; sampling unfiltered would yield degenerate 1-3 point "series".

**Friday-anchored dates.** These are the source's genuine week-ending
Fridays and are left untouched. That makes this the only weekly source in
the repo that is not Sunday-anchored (UCI, the calibrated `walmart`, and M4
Weekly's synthetic index are all `W-SUN`). Shifting them two days to match
would have been cosmetic consistency bought with corrupted calendar data, so
it was not done. The one known consequence: `ProphetForecaster` builds its
future dates with `make_future_dataframe(..., freq="W")`, which pandas
resolves to the `W-SUN` anchor, so its contiguity check rejects these
series. Prophet therefore needs a `W-FRI` freq for this source; that is a
model-side concern, deliberately left to the wiring/benchmark tasks rather
than papered over here. The other baselines are index-blind and unaffected.

Negative `y` values appear in a few pairs (a week whose returns exceeded its
sales). They are real observations and are kept; no series here has a
non-positive mean.
"""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset

DERIVED_DIR = DATA_DIR / "derived"
SOURCE_DIR = DERIVED_DIR / "walmart_real"


def load_walmart_real() -> dict[str, SeriesDataset]:
    """Load the 50 sampled Walmart store-dept series (n=100-143, weekly)."""
    ids = pd.read_csv(DERIVED_DIR / "walmart_real_ids.csv")["id"].astype(str)

    out: dict[str, SeriesDataset] = {}
    for pair_id in ids:
        df = pd.read_csv(SOURCE_DIR / f"{pair_id}.csv", parse_dates=["ds"])
        series = df.set_index("ds")["y"].astype(float).sort_index()
        name = f"walmart_real_{pair_id}"
        out[name] = make_dataset(name, series, freq="W", real=True)
    return out

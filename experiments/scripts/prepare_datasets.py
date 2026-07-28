"""One-time raw -> derived conversion for the Phase B dataset sources.

Large raw inputs (the `datasetsforecast` downloads and the local Kaggle
dumps under `data/<competition>/`) are gitignored. This script reads them, samples
the subset the design calls for, and writes small per-series CSVs into
`data/derived/`, which ARE committed. The loaders in `shortseq/datasets/`
read only from `data/derived/`, so a fresh clone reproduces the exact
benchmark without re-downloading anything.

Sampling uses a fixed seed (42, the project's existing convention) and is
performed once, here. The selected ids are written to a per-source
`*_ids.csv` manifest so the sample is frozen in the repo rather than
re-drawn on every load.

Usage::

    python experiments/scripts/prepare_datasets.py            # all sources
    python experiments/scripts/prepare_datasets.py --only m4_weekly

See `docs/superpowers/specs/2026-07-28-new-datasets-design.md`.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"

SEED = 42
N_M4_WEEKLY = 50
N_M3_MONTHLY = 100
N_ROSSMANN = 50
N_WALMART_REAL = 50

# Minimum usable history for a Walmart store-dept pair. The competition's
# full history is 143 weeks, but 660 of the 3331 pairs are departments that
# opened, closed or churned mid-window — some with a single observation.
# 100 weeks keeps the sample comfortably longer than any evaluation horizon
# while still admitting the handful of genuinely shorter-lived departments.
MIN_WALMART_WEEKS = 100


def _sample_ids(all_ids, k: int) -> list[str]:
    """Draw `k` series ids reproducibly.

    The pool is sorted first so the draw depends only on the *set* of ids
    the upstream package returns, not on its row ordering, and the result
    is sorted so the manifest (and therefore every loader's iteration
    order) is stable.
    """
    pool = sorted(set(map(str, all_ids)))
    if k > len(pool):
        raise ValueError(f"asked for {k} series but only {len(pool)} available")
    rng = np.random.default_rng(SEED)
    return sorted(rng.choice(pool, size=k, replace=False).tolist())


def _write_manifest(source: str, ids: list[str]) -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids}).to_csv(DERIVED_DIR / f"{source}_ids.csv", index=False)


def prepare_m4_weekly() -> int:
    """Sample 50 M4 Weekly series into `data/derived/m4_weekly/`.

    M4 does not publish calendar dates for its series: the upstream `ds`
    column is a plain 1,2,3,... step counter. The derived CSVs therefore
    keep that honest `step,y` shape and the loader builds the synthetic
    weekly DatetimeIndex, exactly as the existing M4 Micro Monthly loader
    (`shortseq/datasets/m4.py`) does — no fabricated dates are committed.
    """
    from datasetsforecast.m4 import M4  # optional dep; only needed to prepare

    df, *_ = M4.load(directory=str(DATA_DIR / "_m4_cache"), group="Weekly")
    ids = _sample_ids(df["unique_id"].unique(), N_M4_WEEKLY)

    out_dir = DERIVED_DIR / "m4_weekly"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df[df["unique_id"].astype(str).isin(ids)]
    for series_id, group in df.groupby(df["unique_id"].astype(str)):
        out = pd.DataFrame(
            {"step": group["ds"].astype(int), "y": group["y"].astype(float)}
        ).sort_values("step")
        out.to_csv(out_dir / f"{series_id}.csv", index=False)

    _write_manifest("m4_weekly", ids)
    return len(ids)


def prepare_m3_monthly() -> int:
    """Sample 100 M3 Monthly series into `data/derived/m3_monthly/`.

    Unlike M4, M3 does publish a starting year/month per series, and
    `datasetsforecast` turns it into real month-*end* timestamps. Those
    dates are kept (only the calendar month carries information) but are
    normalised to month-*start*, which is the convention the rest of the
    repo's freq="M" handling assumes — see the `_FREQ_ALIASES = {"M": "MS"}`
    comment in `shortseq/models/prophet_model.py`.

    Caveat kept in the open: a minority of M3 series carry a 1900-01
    placeholder start rather than a genuine one. Only the *spacing* of the
    index matters to the forecasters, so this is harmless here, but the
    absolute dates of those series should not be read as meaningful.
    """
    from datasetsforecast.m3 import M3  # optional dep; only needed to prepare

    df, *_ = M3.load(directory=str(DATA_DIR / "_m3_cache"), group="Monthly")
    ids = _sample_ids(df["unique_id"].unique(), N_M3_MONTHLY)

    out_dir = DERIVED_DIR / "m3_monthly"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df[df["unique_id"].astype(str).isin(ids)]
    for series_id, group in df.groupby(df["unique_id"].astype(str)):
        ds = pd.to_datetime(group["ds"]).dt.to_period("M").dt.to_timestamp()
        out = pd.DataFrame({"ds": ds, "y": group["y"].astype(float)}).sort_values("ds")
        out.to_csv(out_dir / f"{series_id}.csv", index=False)

    _write_manifest("m3_monthly", ids)
    return len(ids)


def _contiguous_ids(index_counts: pd.Series, spans: pd.Series) -> pd.Index:
    """Ids whose observation count equals their calendar span.

    Equality means every period between first and last observation is
    present, i.e. a gap-free index. Cheaper and clearer than diffing every
    id's index individually.
    """
    return index_counts.index[index_counts == spans]


def prepare_rossmann() -> int:
    """Sample 50 Rossmann stores into `data/derived/rossmann/`.

    Source: the Kaggle `rossmann-store-sales` train.csv (real German drug
    store daily sales, 2013-01-01..2015-07-31, 1115 stores). One series per
    store: `Sales` over `Date`.

    Two shaping decisions, both deliberate:

    1. **Closed-day zero rows are KEPT.** Rossmann stores are shut on most
       Sundays, which upstream is a real row with `Open=0, Sales=0` (plus 54
       rows that are open with genuinely zero sales). Dropping them would
       leave a 6-days-a-week index that is neither daily nor weekly, and
       would discard exactly the zero-inflated demand this benchmark exists
       to study — cf. the M5 loader, whose selling point is its ~32.9%
       zero-demand days. Keeping them yields a true daily index at ~17%
       zeros.

    2. **The pool is restricted to stores with a gap-free daily index.** 180
       of the 1115 stores were closed for refurbishment for roughly the
       second half of 2014 and are simply absent from the file for that
       window (758 rows instead of 942). Those are excluded rather than
       zero-filled: the gap is missing data, not observed zero demand, and
       fabricating six months of zeros would be a much bigger lie than
       dropping the store. 935 stores remain eligible.
    """
    raw = DATA_DIR / "rossmann-store-sales" / "train.csv"
    df = pd.read_csv(
        raw, usecols=["Store", "Date", "Sales"], parse_dates=["Date"]
    )

    by_store = df.groupby("Store")["Date"]
    counts = by_store.size()
    spans = (by_store.max() - by_store.min()).dt.days + 1
    eligible = _contiguous_ids(counts, spans)
    ids = _sample_ids(eligible, N_ROSSMANN)

    out_dir = DERIVED_DIR / "rossmann"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df[df["Store"].astype(str).isin(ids)]
    for store_id, group in df.groupby(df["Store"].astype(str)):
        out = pd.DataFrame(
            {"ds": group["Date"], "y": group["Sales"].astype(float)}
        ).sort_values("ds")
        out.to_csv(out_dir / f"{store_id}.csv", index=False)

    _write_manifest("rossmann", ids)
    return len(ids)


def prepare_walmart_real() -> int:
    """Sample 50 Walmart store-dept pairs into `data/derived/walmart_real/`.

    Source: the Kaggle `walmart-recruiting-store-sales-forecasting`
    train.csv.zip (real US weekly sales, 45 stores x 81 departments,
    2010-02-05..2012-10-26). `pandas.read_csv` reads the .zip directly, so
    the archive is never unpacked to disk.

    One series per `(Store, Dept)` pair, keyed `<store>_<dept>`. Shaping
    decisions:

    1. **Eligible pairs need a gap-free weekly index of >= 100 weeks.** Only
       2660 of the 3331 pairs carry the full 143-week history; the rest are
       departments that opened or closed mid-window, some with a single
       observation. Sampling unfiltered would produce degenerate 1-3 point
       "series". 100 weeks (`MIN_WALMART_WEEKS`) leaves 2671 eligible pairs.

    2. **The real Friday dates are kept as-is.** These are genuine
       week-ending Fridays, unlike every other weekly source in this repo,
       which is Sunday-anchored. They are NOT shifted to Sundays: that would
       corrupt real calendar data for cosmetic consistency. See the loader
       docstring for the one downstream consequence.

    Negative `Weekly_Sales` values (net returns in a week) are left in — they
    are real observations, and no eligible pair has a non-positive mean.
    """
    raw = (
        DATA_DIR
        / "walmart-recruiting-store-sales-forecasting"
        / "train.csv.zip"
    )
    df = pd.read_csv(
        raw, usecols=["Store", "Dept", "Date", "Weekly_Sales"], parse_dates=["Date"]
    )
    df["pair"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)

    by_pair = df.groupby("pair")["Date"]
    counts = by_pair.size()
    spans = (by_pair.max() - by_pair.min()).dt.days // 7 + 1
    eligible = _contiguous_ids(counts, spans)
    eligible = eligible[counts[eligible] >= MIN_WALMART_WEEKS]
    ids = _sample_ids(eligible, N_WALMART_REAL)

    out_dir = DERIVED_DIR / "walmart_real"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df[df["pair"].isin(ids)]
    for pair_id, group in df.groupby("pair"):
        out = pd.DataFrame(
            {"ds": group["Date"], "y": group["Weekly_Sales"].astype(float)}
        ).sort_values("ds")
        out.to_csv(out_dir / f"{pair_id}.csv", index=False)

    _write_manifest("walmart_real", ids)
    return len(ids)


# A later Phase B task adds the remaining Kaggle-derived source (favorita)
# here; each entry is a zero-argument callable returning the number of series
# it wrote.
PREPARERS = {
    "m4_weekly": prepare_m4_weekly,
    "m3_monthly": prepare_m3_monthly,
    "rossmann": prepare_rossmann,
    "walmart_real": prepare_walmart_real,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(PREPARERS),
        help="prepare only these sources (default: all)",
    )
    args = parser.parse_args()

    selected = args.only or sorted(PREPARERS)
    for source in selected:
        print(f"[prepare] {source} ...", flush=True)
        count = PREPARERS[source]()
        print(f"[prepare] {source}: wrote {count} series", flush=True)


if __name__ == "__main__":
    main()

"""One-time raw -> derived conversion for the Phase B dataset sources.

Large raw inputs (the `datasetsforecast` downloads here, the local Kaggle
dumps in later sources) are gitignored. This script reads them, samples
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


# Later Phase B tasks add the Kaggle-derived sources (rossmann, walmart_real,
# favorita) here; each entry is a zero-argument callable returning the number
# of series it wrote.
PREPARERS = {
    "m4_weekly": prepare_m4_weekly,
    "m3_monthly": prepare_m3_monthly,
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

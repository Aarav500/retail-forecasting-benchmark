"""Tests for the Phase B dataset sources (see docs/superpowers/specs/2026-07-28-new-datasets-design.md).

Mirrors the structure of `tests/test_datasets.py`: per-source checks on
series counts, `n` ranges, the `real` flag and sane summary statistics.
These read only from `data/derived/`, which is committed, so they pass on
a fresh clone without re-downloading anything.
"""
import pandas as pd

from shortseq.datasets.m3_monthly import load_m3_monthly
from shortseq.datasets.m4_weekly import load_m4_weekly


def _assert_sane(ds, freq: str) -> None:
    """Checks that must hold for every real series regardless of source."""
    assert ds.real is True
    assert ds.freq == freq
    assert ds.n == len(ds.series)
    assert isinstance(ds.series.index, pd.DatetimeIndex)
    assert ds.series.index.is_monotonic_increasing
    assert ds.series.index.is_unique
    assert ds.series.notna().all()
    # CV/AC(1) are finite and in their mathematically valid ranges. Both M4
    # and M3 are strictly positive demand-like data, so CV must be positive
    # and finite (a zero CV would mean a constant series).
    assert 0.0 < ds.cv < 10.0
    assert -1.0 <= ds.ac1 <= 1.0


# --- M4 Weekly -------------------------------------------------------------


def test_load_m4_weekly_returns_50_sampled_series():
    datasets = load_m4_weekly()
    assert len(datasets) == 50
    assert all(name.startswith("m4w_") for name in datasets)


def test_load_m4_weekly_series_are_real_weekly_and_sane():
    for ds in load_m4_weekly().values():
        _assert_sane(ds, freq="W")
        # M4 Weekly series run 93-2610 observations in the full competition
        # set; the sample is drawn from that same pool.
        assert 93 <= ds.n <= 2610


def test_load_m4_weekly_index_is_sunday_anchored():
    """Sunday-anchored, i.e. pandas' default 'W' == 'W-SUN'.

    Regression guard: a Monday-anchored synthetic index still has 7-day
    gaps but breaks `ProphetForecaster.predict_rolling`, which extrapolates
    with `date_range(..., freq="W")` and then checks contiguity against the
    test index. The repo's other weekly sources (UCI, Walmart) are Sundays
    too, so this keeps freq="W" meaning one thing benchmark-wide.
    """
    for ds in load_m4_weekly().values():
        assert (ds.series.index.dayofweek == 6).all()
        gaps = ds.series.index.to_series().diff().dropna().unique()
        assert list(gaps) == [pd.Timedelta(days=7)]


def test_load_m4_weekly_sample_is_frozen():
    """Golden ids from the committed seed-42 draw.

    Asserted against literals rather than by re-deriving the sample: the
    upstream id pool isn't committed, so re-deriving would only restate
    the manifest. Literals are what actually catches the sample being
    silently regenerated with a different seed or a different upstream
    release.
    """
    names = list(load_m4_weekly())
    assert names[0] == "m4w_W119"
    assert names[-1] == "m4w_W86"


# --- M3 Monthly ------------------------------------------------------------


def test_load_m3_monthly_returns_100_sampled_series():
    datasets = load_m3_monthly()
    assert len(datasets) == 100
    assert all(name.startswith("m3m_") for name in datasets)


def test_load_m3_monthly_series_are_real_monthly_and_sane():
    for ds in load_m3_monthly().values():
        _assert_sane(ds, freq="M")
        # M3 Monthly runs 66-144 observations across the full competition set.
        assert 66 <= ds.n <= 144


def test_load_m3_monthly_index_is_month_start():
    """Month-START timestamps: `prophet_model._FREQ_ALIASES` maps 'M' -> 'MS'
    on the strength of this convention, so it is asserted, not assumed."""
    for ds in load_m3_monthly().values():
        assert (ds.series.index.day == 1).all()


def test_load_m3_monthly_sample_is_frozen():
    """Golden ids from the committed seed-42 draw (see the M4 Weekly twin)."""
    names = list(load_m3_monthly())
    assert names[0] == "m3m_M1052"
    assert names[-1] == "m3m_M983"


# --- cross-source ----------------------------------------------------------


def test_new_source_keys_do_not_collide():
    m4w = load_m4_weekly()
    m3m = load_m3_monthly()
    assert not set(m4w) & set(m3m)

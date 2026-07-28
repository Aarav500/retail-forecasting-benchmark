"""Tests for the Phase B dataset sources (see docs/superpowers/specs/2026-07-28-new-datasets-design.md).

Mirrors the structure of `tests/test_datasets.py`: per-source checks on
series counts, `n` ranges, the `real` flag and sane summary statistics.
These read only from `data/derived/`, which is committed, so they pass on
a fresh clone without re-downloading anything.
"""
import pandas as pd
import pytest

from shortseq.datasets.favorita import (
    FAVORITA_GAP_DATES,
    load_favorita_family,
    load_favorita_store_item,
)
from shortseq.datasets.m3_monthly import load_m3_monthly
from shortseq.datasets.m4_weekly import load_m4_weekly
from shortseq.datasets.rossmann import load_rossmann
from shortseq.datasets.walmart import load_walmart
from shortseq.datasets.walmart_real import load_walmart_real


def _assert_sane(ds, freq: str) -> None:
    """Checks that must hold for every real series regardless of source."""
    assert ds.real is True
    assert ds.freq == freq
    assert ds.n == len(ds.series)
    assert isinstance(ds.series.index, pd.DatetimeIndex)
    assert ds.series.index.is_monotonic_increasing
    assert ds.series.index.is_unique
    assert ds.series.notna().all()
    # CV/AC(1) are finite and in their mathematically valid ranges. Every
    # source here is demand-like data with a positive mean, so CV must be
    # positive and finite (a zero CV would mean a constant series).
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


# --- Rossmann --------------------------------------------------------------


def test_load_rossmann_returns_50_sampled_stores():
    datasets = load_rossmann()
    assert len(datasets) == 50
    assert all(name.startswith("rossmann_store_") for name in datasets)


def test_load_rossmann_series_are_real_daily_and_sane():
    for ds in load_rossmann().values():
        _assert_sane(ds, freq="D")
        # The sampling pool is restricted to stores with a gap-free daily
        # index, which upstream means the full 2013-01-01..2015-07-31 span
        # (942 days; one store starts a day late).
        assert 941 <= ds.n <= 942
        assert (ds.series >= 0).all()


def test_load_rossmann_index_is_contiguous_daily():
    """A gap-free daily index is the whole point of restricting the pool.

    180 of the 1115 Rossmann stores are missing ~6 months (the 2014
    refurbishment window). Keeping them would give a DatetimeIndex that
    *looks* daily to every index-blind baseline while silently splicing
    July 2014 onto January 2015.
    """
    for ds in load_rossmann().values():
        gaps = ds.series.index.to_series().diff().dropna().unique()
        assert list(gaps) == [pd.Timedelta(days=1)]


def test_load_rossmann_keeps_closed_day_zeros():
    """Closed-Sunday rows are kept as genuine zero-demand observations.

    Rossmann stores mostly close on Sundays (`Open=0`, `Sales=0`). Dropping
    those rows would both destroy the daily index and throw away exactly
    the zero-inflation this benchmark studies, so `zero_frac` must be
    non-trivially positive for the typical store.
    """
    datasets = load_rossmann()
    zero_fracs = [ds.zero_frac for ds in datasets.values()]
    assert max(zero_fracs) > 0.1
    # ~1/7 of days closed for a Sunday-closing store; nothing near all-zero.
    assert all(0.0 <= zf < 0.5 for zf in zero_fracs)


def test_load_rossmann_sample_is_frozen():
    """Golden ids from the committed seed-42 draw (see the M4 Weekly twin)."""
    names = list(load_rossmann())
    assert names[0] == "rossmann_store_1060"
    assert names[-1] == "rossmann_store_981"


# --- Walmart (real Kaggle data) --------------------------------------------


def test_load_walmart_real_returns_50_sampled_store_depts():
    datasets = load_walmart_real()
    assert len(datasets) == 50
    assert all(name.startswith("walmart_real_") for name in datasets)


def test_load_walmart_real_series_are_real_weekly_and_sane():
    for ds in load_walmart_real().values():
        _assert_sane(ds, freq="W")
        # Pool is store-dept pairs with a gap-free weekly index of >= 100
        # weeks; the full competition history is 143 weeks.
        assert 100 <= ds.n <= 143


def test_load_walmart_real_index_is_friday_anchored():
    """Kaggle's Walmart dates are genuine week-*ending Fridays*, not Sundays.

    Pinned deliberately: the repo's other weekly sources (UCI, the
    calibrated `walmart`, M4 Weekly) are Sunday-anchored, so it would be
    easy to "fix" these to Sundays for consistency. These are real calendar
    dates from the competition data and are left alone; the anchor
    difference is a property of the source, not a bug. See the loader
    docstring for the Prophet consequence.
    """
    for ds in load_walmart_real().values():
        assert (ds.series.index.dayofweek == 4).all()
        gaps = ds.series.index.to_series().diff().dropna().unique()
        assert list(gaps) == [pd.Timedelta(days=7)]


def test_load_walmart_real_dates_match_the_competition_span():
    """Real dates, not a synthetic index: the 143-week series must span the
    published 2010-02-05 .. 2012-10-26 competition window exactly."""
    full = [ds for ds in load_walmart_real().values() if ds.n == 143]
    assert full, "expected at least one full-history store-dept pair"
    for ds in full:
        assert ds.series.index[0] == pd.Timestamp("2010-02-05")
        assert ds.series.index[-1] == pd.Timestamp("2012-10-26")


def test_walmart_real_needs_w_fri_for_prophet():
    """Pins the one known consequence of keeping the real Friday dates.

    `ProphetForecaster` extrapolates future dates with
    `make_future_dataframe(..., freq=self.freq)`, and pandas resolves a bare
    "W" to the `W-SUN` anchor — so on Friday-dated series its own contiguity
    check rejects the result. This is asserted in BOTH directions so the
    requirement is documented executably: whoever wires this source into the
    benchmark (or ever considers shifting these dates to Sundays) gets told
    by a test rather than by a mid-run crash. The other baselines are
    index-blind and unaffected.
    """
    from shortseq.models.prophet_model import ProphetForecaster

    series = load_walmart_real()["walmart_real_12_52"].series
    train, test = series.iloc[:-8], series.iloc[-8:]

    with pytest.raises(ValueError, match="contiguous continuation"):
        ProphetForecaster(freq="W").fit(train).predict_rolling(test)

    forecast = ProphetForecaster(freq="W-FRI").fit(train).predict_rolling(test)
    assert len(forecast.point) == len(test)


def test_load_walmart_real_sample_is_frozen():
    """Golden ids from the committed seed-42 draw (see the M4 Weekly twin)."""
    names = list(load_walmart_real())
    assert names[0] == "walmart_real_12_52"
    assert names[-1] == "walmart_real_9_91"


# --- Favorita (one source, two aggregation views) --------------------------
#
# `favorita_store_item_*` and `favorita_family_*` are two aggregations of the
# SAME Kaggle competition dump, not two sources. The tests below deliberately
# assert that relationship (see `test_favorita_views_are_the_same_source`) so
# that a later diversity claim in the paper cannot quietly count Favorita
# twice.


def test_load_favorita_store_item_returns_50_sampled_pairs():
    datasets = load_favorita_store_item()
    assert len(datasets) == 50
    assert all(name.startswith("favorita_store_item_") for name in datasets)


def test_load_favorita_family_returns_all_33_families():
    """All 33 families, not a sample: the family view is a full census.

    Sampling 33 out of 33 would be a no-op, and taking a subset would mean
    choosing which product lines to report on — an unnecessary degree of
    freedom in a benchmark whose selling point is that its sampling is
    frozen and mechanical.
    """
    datasets = load_favorita_family()
    assert len(datasets) == 33
    assert all(name.startswith("favorita_family_") for name in datasets)


def test_load_favorita_store_item_series_are_real_daily_and_sane():
    for ds in load_favorita_store_item().values():
        _assert_sane(ds, freq="D")
        assert ds.n == 1684
        assert (ds.series >= 0).all()


def test_load_favorita_family_series_are_real_daily_and_sane():
    for ds in load_favorita_family().values():
        _assert_sane(ds, freq="D")
        assert ds.n == 1684
        assert (ds.series >= 0).all()


def test_favorita_series_span_the_competition_window():
    """Real calendar dates, 2013-01-01 .. 2017-08-15 (the published span)."""
    for loader in (load_favorita_store_item, load_favorita_family):
        for ds in loader().values():
            assert ds.series.index[0] == pd.Timestamp("2013-01-01")
            assert ds.series.index[-1] == pd.Timestamp("2017-08-15")


def test_favorita_index_is_daily_except_four_christmas_gaps():
    """The only breaks in the daily index are the four Christmas Days.

    Every Favorita store shuts on Navidad and the competition file simply
    omits those rows rather than recording them as zeros (contrast New
    Year's Day, which IS present as explicit zero rows). Those four dates
    are left absent rather than zero-filled — see the loader docstring — so
    the gap structure is pinned here: exactly 1684 observations over a
    1688-day span, with 1-day steps everywhere except four 2-day steps.
    """
    assert len(FAVORITA_GAP_DATES) == 4
    for loader in (load_favorita_store_item, load_favorita_family):
        for ds in loader().values():
            idx = ds.series.index
            gaps = sorted(idx.to_series().diff().dropna().unique())
            assert gaps == [pd.Timedelta(days=1), pd.Timedelta(days=2)]
            missing = pd.date_range(idx[0], idx[-1], freq="D").difference(idx)
            assert list(missing) == list(FAVORITA_GAP_DATES)


def test_favorita_store_item_keeps_zero_demand_days_but_excludes_dead_pairs():
    """Zero-inflation is kept; structural non-carriage is not.

    53 of the 1782 (store, family) pairs are identically zero across the
    whole window (the store simply never carried that product line) and
    another 16 are >=99% zero — those have an undefined CV and are not
    demand series at all. Eligibility therefore requires sales on at least
    half the days. The surviving series still carry substantial zero
    inflation, which is the property this benchmark exists to study.
    """
    zero_fracs = [ds.zero_frac for ds in load_favorita_store_item().values()]
    assert max(zero_fracs) <= 0.5
    assert max(zero_fracs) > 0.1  # genuinely zero-inflated series survive


def test_favorita_store_item_excludes_late_opening_stores():
    """Stores that opened mid-window are not in the pool.

    The 2021 re-release pads every store back to 2013-01-01 with zero rows,
    so eight stores (20, 21, 22, 29, 36, 42, 52, 53) carry hundreds — store
    52 carries 1566 — of leading zeros for a period in which the store did
    not exist. Those zeros are fabricated non-observations, not observed
    zero demand, so the stores are excluded rather than kept or trimmed.
    """
    late = {"20", "21", "22", "29", "36", "42", "52", "53"}
    for name in load_favorita_store_item():
        store = name[len("favorita_store_item_"):].split("_", 1)[0]
        assert store not in late


def test_favorita_views_are_the_same_source():
    """The two views must aggregate to the same totals on the same dates.

    This is the executable form of the "count Favorita once" rule: if the
    store-item and family views were ever re-derived from genuinely
    different data, this test would fail. For every sampled (store, family)
    pair, that store's contribution cannot exceed the all-store family
    total on any date.
    """
    families = load_favorita_family()
    for name, ds in load_favorita_store_item().items():
        family_slug = name[len("favorita_store_item_"):].split("_", 1)[1]
        total = families[f"favorita_family_{family_slug}"].series
        assert list(ds.series.index) == list(total.index)
        assert (ds.series <= total + 1e-6).all()


def test_load_favorita_sample_is_frozen():
    """Golden ids from the committed seed-42 draw (see the M4 Weekly twin)."""
    names = list(load_favorita_store_item())
    assert names[0] == "favorita_store_item_13_beauty"
    assert names[-1] == "favorita_store_item_9_grocery_ii"
    families = list(load_favorita_family())
    assert families[0] == "favorita_family_automotive"
    assert families[-1] == "favorita_family_seafood"


def test_favorita_works_with_prophet_despite_the_christmas_gaps():
    """The gapped index does not break the one index-sensitive baseline.

    `ProphetForecaster.predict_rolling` extrapolates the horizon from the
    train end with `freq="D"` and rejects a non-contiguous continuation.
    All four gaps fall in December of 2013-2016, i.e. strictly inside the
    training portion of any end-of-series split, so the horizon is
    contiguous and Prophet is unaffected. Asserted rather than assumed,
    since leaving the gaps in was a deliberate choice (see the loader
    docstring) and this is the risk it carries.
    """
    from shortseq.models.prophet_model import ProphetForecaster

    series = load_favorita_family()["favorita_family_grocery_i"].series
    train, test = series.iloc[:-8], series.iloc[-8:]
    forecast = ProphetForecaster(freq="D").fit(train).predict_rolling(test)
    assert len(forecast.point) == len(test)


# --- cross-source ----------------------------------------------------------


def test_new_source_keys_do_not_collide():
    sources = [
        load_m4_weekly(),
        load_m3_monthly(),
        load_rossmann(),
        load_walmart_real(),
        load_favorita_store_item(),
        load_favorita_family(),
    ]
    seen: set[str] = set()
    for source in sources:
        assert not seen & set(source)
        seen |= set(source)


def test_walmart_real_does_not_collide_with_calibrated_walmart():
    """The pre-existing calibrated single series owns the bare `walmart` key.

    `walmart_real_*` is prefixed precisely to keep the two apart once both
    are wired into the registry (Task 4), so this is asserted rather than
    assumed.
    """
    calibrated = load_walmart()
    real = load_walmart_real()
    assert set(calibrated) == {"walmart"}
    assert not set(calibrated) & set(real)
    assert "walmart" not in real

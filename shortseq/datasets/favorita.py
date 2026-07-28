"""Favorita real retail data (Ecuador, daily) — ONE source, TWO views.

ONE SOURCE
==========
Everything in this module comes from a single origin: the Kaggle
`store-sales-time-series-forecasting` competition dump, which is itself a
re-release of the older `favorita-grocery-sales-forecasting` data from the
same Ecuadorian grocery chain, Corporación Favorita. The project's original
execution plan listed "Favorita", "Corporación Favorita" and "StoreSales" as
three separate dataset sources; investigation showed they are three
aggregations of that one competition, and this module implements the two
that are useful here as two documented *views*.

**Binding consequence: any claim about the number of dataset sources in this
benchmark must count Favorita ONCE.** Both `load_favorita_store_item` and
`load_favorita_family` read from the same 3,000,888-row `train.csv`; their
series are not independent observations of retail demand, they are the same
transactions sliced two ways (the family view is literally the store-item
view summed over stores). Counting the views separately would inflate the
benchmark's claimed source diversity — a fair and easy target for reviewer
criticism in a paper whose thesis is methodological rigour. The two views
are still worth having: they probe the same demand process at two
aggregation levels, which is exactly the axis along which forecastability
changes (aggregation smooths intermittency), and that contrast is a result,
not a diversity claim.

THE TWO VIEWS
=============
Both are prepared by `experiments/scripts/prepare_datasets.py` (fixed seed
42) and read only from `data/derived/`.

**Store-item view** — `load_favorita_store_item`, keys
`favorita_store_item_<store>_<family_slug>`. 50 sampled (store_nbr, family)
pairs; the draw is frozen in `data/derived/favorita_store_item_ids.csv` and
one `ds,y` CSV per pair lives in `data/derived/favorita_store_item/`.

**Product-family view** — `load_favorita_family`, keys
`favorita_family_<family_slug>`. All 33 product families, each summed across
all 54 stores; manifest `data/derived/favorita_family_ids.csv`, CSVs in
`data/derived/favorita_family/`. No sampling: 33 of 33 is a census, and
picking a subset would be an unforced choice about which product lines the
benchmark reports on.

`<family_slug>` is the upper-case source label lower-cased with every run of
non-alphanumeric characters collapsed to `_` (`BREAD/BAKERY` ->
`bread_bakery`, `LIQUOR,WINE,BEER` -> `liquor_wine_beer`). All 33 slugs are
distinct.

SHAPING DECISIONS
=================
Every series here is n=1684 real daily observations spanning
2013-01-01..2017-08-15. That is LONG by this benchmark's standards — an
order of magnitude past the n < 200 short-series regime the project is
about. Kept deliberately, as the contrast case the spec's stratification
addendum calls for; results over these series must be reported in the long
stratum, never pooled with the short one.

**The four Christmas gaps are left as gaps.** The source has 1684 dates over
a 1688-day span: 25 December 2013, 2014, 2015 and 2016 are absent (the 2017
data stops in August). Every Favorita store shuts for Navidad, and the
competition file omits those rows rather than recording zeros — which is
inconsistent with how it treats the equally chain-wide New Year's Day
closure, where explicit `sales=0` rows ARE present. That inconsistency makes
zero-filling tempting: it would restore a contiguous daily index and merely
apply the source's own convention. It was rejected anyway. Four fabricated
observations is a small lie, but it is still writing data that the source
does not contain, and Task 2 set the precedent that absent data is excluded
rather than invented (`rossmann.py`, on the refurbishment holes). The cost
is that the index is daily-with-four-2-day-steps rather than strictly
contiguous; the four gaps all fall deep inside the training portion of any
end-of-series split, so the one index-sensitive baseline (Prophet, whose
`predict_rolling` checks that the *horizon* continues contiguously) is
unaffected — pinned by `test_favorita_works_with_prophet_despite_the_christmas_gaps`.

**Zero-demand days are kept; structurally dead pairs are not.** As with
Rossmann and M5, an observed zero is real information and this benchmark
exists partly to study zero-inflated demand, so no zero row is dropped.
But 53 of the 1782 (store, family) pairs are identically zero across all
1684 days and 69 are >=99% zero: those are stores that never carried the
product line, and a flat zero column has an undefined CV and is not a
demand series at all. Store-item eligibility therefore requires sales on
more than half the days. The threshold is loose on purpose — surviving
pairs still reach ~50% zero days, above the ~32.9% that makes the M5 source
(`m5.py`) interesting — so it removes non-carriage without removing
intermittency.

**Late-opening stores are excluded from the store-item view.** The 2021
re-release pads every store back to 2013-01-01, so eight stores (20, 21, 22,
29, 36, 42, 52, 53) begin with between 128 and 1566 zero rows covering
dates on which the store did not exist. Those zeros are fabricated absence
dressed as zero demand — indistinguishable, downstream, from a store that
sold nothing — so the stores are dropped, exactly as Task 2 dropped the 180
Rossmann stores with refurbishment holes rather than zero-filling them. 46
stores remain in the pool. Trimming each series to its store's opening date
was the alternative and would have yielded some genuinely short series;
it was rejected because it trades a uniform view for eight ragged ones on a
judgement call about opening dates that the source never states.

The family view applies **no** store filter, and needs none: summing over
stores, a not-yet-opened store contributes exactly 0, which is the
arithmetically correct total for that date. The chain's growth from 46 to 54
stores is a real trend in those series, not an artefact.
"""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset

DERIVED_DIR = DATA_DIR / "derived"
STORE_ITEM_DIR = DERIVED_DIR / "favorita_store_item"
FAMILY_DIR = DERIVED_DIR / "favorita_family"

# Dates absent from the source (Navidad; stores chain-wide shut). Exposed so
# tests and downstream code can assert the gap structure rather than
# rediscover it. 2017-12-25 is past the end of the data.
FAVORITA_GAP_DATES = tuple(
    pd.Timestamp(f"{year}-12-25") for year in (2013, 2014, 2015, 2016)
)


def _load_view(manifest: str, source_dir, prefix: str) -> dict[str, SeriesDataset]:
    """Load one aggregation view from its frozen manifest."""
    ids = pd.read_csv(DERIVED_DIR / manifest)["id"].astype(str)

    out: dict[str, SeriesDataset] = {}
    for series_id in ids:
        df = pd.read_csv(source_dir / f"{series_id}.csv", parse_dates=["ds"])
        series = df.set_index("ds")["y"].astype(float).sort_index()
        name = f"{prefix}{series_id}"
        out[name] = make_dataset(name, series, freq="D", real=True)
    return out


def load_favorita_store_item() -> dict[str, SeriesDataset]:
    """Store-item view: 50 sampled (store, family) pairs (n=1684, daily).

    One of two views of a single source — see the module docstring, and do
    not count it as a source separate from `load_favorita_family`.
    """
    return _load_view(
        "favorita_store_item_ids.csv", STORE_ITEM_DIR, "favorita_store_item_"
    )


def load_favorita_family() -> dict[str, SeriesDataset]:
    """Product-family view: all 33 families, summed over stores (n=1684, daily).

    One of two views of a single source — see the module docstring, and do
    not count it as a source separate from `load_favorita_store_item`.
    """
    return _load_view("favorita_family_ids.csv", FAMILY_DIR, "favorita_family_")

import pytest

from shortseq.datasets.dmart import load_dmart
from shortseq.datasets.uci import load_uci
from shortseq.datasets.walmart import load_walmart
from shortseq.datasets.m5 import load_m5
from shortseq.datasets.m4 import load_m4
from shortseq.datasets.registry import load_all, load_regime, load_source


def test_load_dmart_matches_paper_stats():
    datasets = load_dmart()
    assert set(datasets) == {"dmart_food", "dmart_electronics", "dmart_clothing", "dmart_furniture"}
    food = datasets["dmart_food"]
    assert food.n == 181
    assert food.real is True
    assert 0.015 < food.cv < 0.03


def test_load_uci_matches_paper_stats():
    datasets = load_uci()
    assert len(datasets) == 5
    for ds in datasets.values():
        assert ds.n == 53
        assert ds.real is True


def test_load_walmart_matches_paper_stats():
    datasets = load_walmart()
    walmart = datasets["walmart"]
    assert walmart.n == 143


def test_load_m5_matches_paper_stats():
    datasets = load_m5()
    m5 = datasets["m5"]
    assert m5.n == 365
    assert 0.30 < m5.zero_frac < 0.35


def test_load_m4_returns_24_series():
    datasets = load_m4()
    assert len(datasets) == 24
    for ds in datasets.values():
        assert 60 <= ds.n <= 200


# Every implemented source, keyed by the label its series names carry, with
# the number of series it must contribute. Listing them per-source rather
# than as one total is deliberate — a bare count would still pass if one
# source vanished while another grew, which is exactly the silent registry
# breakage this test exists to catch.
#
# `walmart` and `m5` are whole keys (single-series sources), not prefixes;
# they are matched exactly so that `walmart_real_*` is not swallowed by
# `walmart`. The remaining prefixes are mutually exclusive (note `m4w_`
# does not start with `m4_`).
EXPECTED_SOURCE_SERIES = {
    "dmart_": 4,
    "uci_": 5,
    "walmart": 1,  # the older calibrated single series
    "m5": 1,
    "m4_": 24,
    "m4w_": 50,
    "m3m_": 100,
    "rossmann_store_": 50,
    "walmart_real_": 50,
    "favorita_store_item_": 50,
    "favorita_family_": 33,
}
EXACT_KEY_SOURCES = {"walmart", "m5"}


def _classify(key: str) -> str:
    if key in EXACT_KEY_SOURCES:
        return key
    matches = [
        p
        for p in EXPECTED_SOURCE_SERIES
        if p not in EXACT_KEY_SOURCES and key.startswith(p)
    ]
    assert len(matches) == 1, f"{key!r} matched {matches}"
    return matches[0]


def test_load_all_combines_every_implemented_source():
    datasets = load_all()

    counts: dict[str, int] = {}
    for key in datasets:
        label = _classify(key)
        counts[label] = counts.get(label, 0) + 1

    assert counts == EXPECTED_SOURCE_SERIES
    # 368 = the original 35 plus the 333 added in Phase B. `load_all` itself
    # raises on a duplicate key, so an equal total also means no collisions.
    assert len(datasets) == sum(EXPECTED_SOURCE_SERIES.values()) == 368


def test_load_regime_filters_by_cv():
    datasets = load_regime(cv_range=(0, 0.05))
    assert all(ds.cv <= 0.05 for ds in datasets.values())
    assert "dmart_food" in datasets


def test_load_source_raises_for_planned_but_unimplemented():
    # Tourism is the only source still unimplemented (it needs R/tsibbledata).
    with pytest.raises(NotImplementedError):
        load_source("tourism")


def test_load_source_returns_data_for_newly_implemented_sources():
    assert len(load_source("m4_weekly")) == 50
    assert len(load_source("m3_monthly")) == 100
    assert len(load_source("rossmann")) == 50
    assert len(load_source("walmart_real")) == 50


def test_favorita_is_one_source_with_two_views():
    # The plan listed favorita / corporacion_favorita / storesales as three
    # sources; they are one Kaggle dump, so the legacy names are aliases and
    # the single source key yields both aggregation views.
    favorita = load_source("favorita")
    assert len(favorita) == 50 + 33
    assert sum(k.startswith("favorita_store_item_") for k in favorita) == 50
    assert sum(k.startswith("favorita_family_") for k in favorita) == 33
    assert set(load_source("corporacion_favorita")) == set(favorita)
    assert set(load_source("storesales")) == set(favorita)


def test_real_walmart_is_an_alias_for_walmart_real():
    assert set(load_source("real_walmart")) == set(load_source("walmart_real"))


def test_load_source_raises_for_unknown_name():
    with pytest.raises(ValueError):
        load_source("not_a_real_source")

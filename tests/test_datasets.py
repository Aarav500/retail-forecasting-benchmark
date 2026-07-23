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


def test_load_all_combines_every_implemented_source():
    datasets = load_all()
    assert len(datasets) == 4 + 5 + 1 + 1 + 24


def test_load_regime_filters_by_cv():
    datasets = load_regime(cv_range=(0, 0.05))
    assert all(ds.cv <= 0.05 for ds in datasets.values())
    assert "dmart_food" in datasets


def test_load_source_raises_for_planned_but_unimplemented():
    with pytest.raises(NotImplementedError):
        load_source("favorita")


def test_load_source_raises_for_unknown_name():
    with pytest.raises(ValueError):
        load_source("not_a_real_source")

"""Tests for Friedman + Nemenyi ranking.

The Nemenyi critical difference is the one piece of genuinely new
statistics here, and a wrong constant would silently produce confident
but wrong significance conclusions. The constants below were verified by
three independent derivations and reproduced directly with scipy:
    CD(k=5, N=199, alpha=0.05) = 0.4323813055932977
The classic failure mode is omitting the /sqrt(2) when converting the
studentized range, which inflates CD to 0.611480 -- there is an explicit
named guard against exactly that.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from shortseq.evaluation.ranking import friedman_nemenyi, nemenyi_critical_difference


def test_cd_matches_independently_verified_value():
    assert nemenyi_critical_difference(k=5, n_blocks=199, alpha=0.05) == pytest.approx(
        0.4323813055932977, abs=1e-9
    )


def test_cd_at_alpha_010():
    assert nemenyi_critical_difference(k=5, n_blocks=199, alpha=0.10) == pytest.approx(
        0.3898595, abs=1e-6
    )


def test_cd_is_not_off_by_sqrt2():
    # Named guard against the single most common Nemenyi bug: pairing the
    # RAW studentized range with the 6N denominator. That yields 0.611480.
    cd = nemenyi_critical_difference(k=5, n_blocks=199, alpha=0.05)
    assert abs(cd - 0.611480) > 0.1


def test_k2_collapses_to_normal_quantile():
    # Decisive convention check: at k=2 Nemenyi must reduce to a two-sided
    # normal test. Under the raw-q convention this fails.
    cd = nemenyi_critical_difference(k=2, n_blocks=100, alpha=0.05)
    expected = norm.ppf(0.975) * np.sqrt(2 * 3 / (6 * 100))
    assert cd == pytest.approx(expected, abs=1e-12)


def test_cd_shrinks_as_series_count_grows():
    # More blocks -> tighter critical difference.
    assert nemenyi_critical_difference(5, 1000, 0.05) < nemenyi_critical_difference(5, 199, 0.05)


def test_cd_grows_with_more_models():
    assert nemenyi_critical_difference(8, 199, 0.05) > nemenyi_critical_difference(3, 199, 0.05)


def test_friedman_detects_a_uniformly_better_model():
    # model_a beats the others on every single series -> must be significant.
    rows = []
    for i in range(30):
        rows.append({"model_a": 1.0 + i * 0.01, "model_b": 5.0 + i * 0.01, "model_c": 9.0 + i * 0.01})
    result = friedman_nemenyi(pd.DataFrame(rows))
    assert result["friedman_p_value"] < 0.001
    assert result["friedman_significant"] is True
    # rank 1 = best (lowest error)
    assert result["mean_ranks"]["model_a"] == pytest.approx(1.0)
    assert result["mean_ranks"]["model_c"] == pytest.approx(3.0)


def test_friedman_does_not_fire_on_identical_models():
    # All models identical on every series -> all ties, no real difference.
    rows = [{"m1": 5.0, "m2": 5.0, "m3": 5.0} for _ in range(30)]
    result = friedman_nemenyi(pd.DataFrame(rows))
    assert result["friedman_significant"] is False


def test_mean_ranks_sum_to_k_times_k_plus_1_over_2():
    # Structural invariant of within-series ranking. Catches ranking across
    # the wrong axis, which is an easy and silent mistake.
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(50, 4)), columns=["a", "b", "c", "d"])
    result = friedman_nemenyi(df)
    k = 4
    assert sum(result["mean_ranks"].values()) == pytest.approx(k * (k + 1) / 2)
    for r in result["mean_ranks"].values():
        assert 1.0 <= r <= k


def test_ties_get_average_ranks():
    # Two models tied for best on every series -> both get mean rank 1.5.
    rows = [{"a": 1.0, "b": 1.0, "c": 2.0} for _ in range(20)]
    result = friedman_nemenyi(rows_to_df(rows))
    assert result["mean_ranks"]["a"] == pytest.approx(1.5)
    assert result["mean_ranks"]["b"] == pytest.approx(1.5)
    assert result["mean_ranks"]["c"] == pytest.approx(3.0)


def rows_to_df(rows):
    return pd.DataFrame(rows)


def test_pairwise_flags_significance_against_cd():
    rows = []
    for i in range(40):
        rows.append({"best": 1.0 + i * 0.01, "mid": 5.0 + i * 0.01, "worst": 9.0 + i * 0.01})
    result = friedman_nemenyi(pd.DataFrame(rows))
    pair = result["pairwise"]["best_vs_worst"]
    assert pair["rank_difference"] == pytest.approx(2.0)
    assert pair["significant"] is True


def test_incomplete_grid_is_rejected():
    # The CD formula assumes every model is scored on every series. A NaN
    # means a missing cell, which would silently invalidate the result.
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, np.nan]})
    with pytest.raises(ValueError, match="complete"):
        friedman_nemenyi(df)


def test_requires_at_least_three_models_for_friedman():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="at least 3"):
        friedman_nemenyi(df)

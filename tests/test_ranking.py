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
    # abs=1e-6 rather than tighter: the value comes from
    # studentized_range.ppf, which scipy computes by numerical quadrature, so
    # low-order digits may shift across scipy releases. The bug this guards
    # against (the missing /sqrt(2)) moves the value by 0.18, so 1e-6 gives
    # identical protection while surviving a scipy upgrade.
    assert nemenyi_critical_difference(k=5, n_blocks=199, alpha=0.05) == pytest.approx(
        0.4323813055932977, abs=1e-6
    )


def test_cd_at_alpha_010():
    # A regression constant, recorded from this implementation - not an
    # independent anchor. Demsar (2006) Table 5 gives q_0.10(k=5) = 2.459
    # (already divided by sqrt(2)), so CD = 2.459 * sqrt(5*6 / (6*199)) =
    # 0.38978, which corroborates the runtime value to the table's 4 s.f.
    # It does NOT corroborate it at abs=1e-6: substituting the table-derived
    # value here fails, because the table's own rounding to 4 s.f. moves CD by
    # ~8e-05. The independent check against the table follows below, at a
    # tolerance the table can actually meet.
    assert nemenyi_critical_difference(k=5, n_blocks=199, alpha=0.10) == pytest.approx(
        0.3898595, abs=1e-6
    )
    # Genuinely independent of this implementation: the only inputs are the
    # published q and the CD formula.
    assert nemenyi_critical_difference(k=5, n_blocks=199, alpha=0.10) == pytest.approx(
        2.459 * np.sqrt(5 * 6 / (6 * 199)), abs=1e-4
    )  # Demsar (2006) Table 5, table precision


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
    # abs=1e-9: this compares two different scipy code paths (studentized_range
    # quadrature vs the normal quantile), so an exact match is not guaranteed
    # across releases. Still far tighter than the sqrt(2) error it must catch.
    assert cd == pytest.approx(expected, abs=1e-9)


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


@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_friedman_does_not_fire_on_identical_models():
    # All models identical on every series -> all ties, no real difference.
    # scipy emits a RuntimeWarning here; that is expected and is deliberately
    # NOT suppressed inside ranking.py, which would also hide real signals.
    rows = [{"m1": 5.0, "m2": 5.0, "m3": 5.0} for _ in range(30)]
    result = friedman_nemenyi(pd.DataFrame(rows))
    assert result["friedman_significant"] is False
    # Pin the nan itself, not just the flag: the ranking.py docstring claims
    # this test pins the nan behaviour. Without this line a future scipy
    # returning (0.0, 1.0) would still pass and the docstring would quietly
    # become false.
    assert np.isnan(result["friedman_statistic"])


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
    result = friedman_nemenyi(pd.DataFrame(rows))
    assert result["mean_ranks"]["a"] == pytest.approx(1.5)
    assert result["mean_ranks"]["b"] == pytest.approx(1.5)
    assert result["mean_ranks"]["c"] == pytest.approx(3.0)


def test_pairwise_flags_significance_against_cd():
    rows = []
    for i in range(40):
        rows.append({"best": 1.0 + i * 0.01, "mid": 5.0 + i * 0.01, "worst": 9.0 + i * 0.01})
    result = friedman_nemenyi(pd.DataFrame(rows))
    pair = result["pairwise"]["best_vs_worst"]
    assert pair["rank_difference"] == pytest.approx(2.0)
    assert pair["significant"] is True


def test_pairwise_not_significant_when_below_cd():
    # The False branch. Without this, an implementation that computes the
    # critical difference and then never consults it - e.g. a hardcoded
    # "significant": True - passes the whole suite.
    # Seed 12 rather than an arbitrary one: pure noise, and the largest rank
    # difference is 0.11 * CD, so no pair is anywhere near the threshold and
    # the test cannot flip on a small numerical change.
    rng = np.random.default_rng(12)
    df = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    result = friedman_nemenyi(df)
    assert result["pairwise"]["a_vs_b"]["significant"] is False
    assert result["pairwise"]["a_vs_c"]["significant"] is False
    assert result["pairwise"]["b_vs_c"]["significant"] is False


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


def test_empty_grid_is_rejected_with_an_actionable_message():
    # Without the guard this surfaces as "need at least one array to
    # concatenate" from np.vstack, which does not say what the caller did wrong.
    df = pd.DataFrame({"a": [], "b": [], "c": []})
    with pytest.raises(ValueError, match="at least 1 series"):
        friedman_nemenyi(df)


def test_alpha_is_actually_used_for_the_critical_difference():
    # Constrains ranking.py's alpha argument at the CD call site. With alpha
    # hardcoded to 0.05 there, every existing test still passes because they
    # all use the default. A larger alpha means a less conservative test, so
    # the critical difference must shrink.
    rng = np.random.default_rng(12)
    df = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    cd_05 = friedman_nemenyi(df, alpha=0.05)["critical_difference"]
    cd_10 = friedman_nemenyi(df, alpha=0.10)["critical_difference"]
    assert cd_10 < cd_05
    assert cd_10 == pytest.approx(nemenyi_critical_difference(3, 30, 0.10), abs=1e-9)


def test_alpha_is_actually_used_for_the_omnibus_verdict():
    # Constrains the other alpha use site, the p_value < alpha comparison,
    # which the critical-difference test above does not reach. Seed 65 gives
    # a Friedman p of 0.0718, which straddles the two alphas, so the verdict
    # must flip.
    rng = np.random.default_rng(65)
    df = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    assert 0.05 < friedman_nemenyi(df)["friedman_p_value"] < 0.10
    assert friedman_nemenyi(df, alpha=0.05)["friedman_significant"] is False
    assert friedman_nemenyi(df, alpha=0.10)["friedman_significant"] is True


def test_cd_requires_at_least_two_models():
    with pytest.raises(ValueError, match="at least 2 models"):
        nemenyi_critical_difference(k=1, n_blocks=199, alpha=0.05)


def test_cd_requires_at_least_one_block():
    with pytest.raises(ValueError, match="at least 1 block"):
        nemenyi_critical_difference(k=5, n_blocks=0, alpha=0.05)


@pytest.mark.parametrize("bad_alpha", [5, 0.0, 1.0, -0.05, 95])
def test_cd_rejects_alpha_outside_the_unit_interval(bad_alpha):
    # alpha reaches this function from hyperparams.yaml, so a percent-vs-
    # fraction mix-up is a live risk rather than a hypothetical one. Unguarded,
    # alpha=5 returns nan (so nothing is ever significant) while the omnibus
    # p_value < 5 test fires - a coherent-looking, entirely wrong table.
    with pytest.raises(ValueError, match="alpha must be in"):
        nemenyi_critical_difference(k=5, n_blocks=199, alpha=bad_alpha)
    # The comment above reasons about friedman_nemenyi, so exercise that path
    # too rather than trusting it to reach the helper. The grid is valid and
    # 3 models wide, so the rejection is unambiguously about alpha and not
    # about the shape of the input.
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 1.0], "c": [3.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="alpha must be in"):
        friedman_nemenyi(df, alpha=bad_alpha)

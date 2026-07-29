"""Statistical ranking across many series - Friedman test + Nemenyi post-hoc.

Ranks models within each series (so series whose RMSE differ by orders of
magnitude - n=53 UCI vs n=1684 Favorita - contribute equally), runs the
Friedman omnibus test on those ranks, and computes the Nemenyi critical
difference for pairwise comparisons.

The Nemenyi critical difference is

    CD = q_alpha * sqrt( k * (k + 1) / (6 * N) )

where k is the number of models, N the number of series (blocks), and
q_alpha is the studentized range statistic at infinite degrees of freedom
DIVIDED BY sqrt(2). That division is not optional: pairing the raw
studentized range with the 6*N denominator inflates CD by ~1.414x and
silently produces wrong significance conclusions. `q_alpha` is computed at
runtime rather than read from a hardcoded table, which removes
transcription risk and generalises beyond the usual k<=10 tables.

Reference: Demsar (2006), "Statistical Comparisons of Classifiers over
Multiple Data Sets", JMLR 7:1-30.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range


def nemenyi_critical_difference(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for average ranks.

    k: number of models being compared.
    n_blocks: number of series (blocks), NOT the number of observations.
    """
    if k < 2:
        raise ValueError(f"need at least 2 models, got {k}")
    if n_blocks < 1:
        raise ValueError(f"need at least 1 block, got {n_blocks}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha} - did you pass a percentage?")
    # q_alpha is the studentized range DIVIDED BY sqrt(2). Do not remove this
    # divisor: pairing raw q with the 6*N denominator inflates CD by 1.414x
    # (0.4324 -> 0.6115 at k=5, N=199). See module docstring; Demsar (2006) Table 5
    # publishes q ALREADY divided by sqrt(2). Guarded by test_cd_is_not_off_by_sqrt2.
    q = studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def friedman_nemenyi(results: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Rank models across series and test whether the ordering is real.

    results: rows = series, columns = models, values = a LOWER-IS-BETTER
    metric (e.g. RMSE). Every model must be scored on every series.

    Returns a dict with mean ranks (rank 1 = best), the Friedman statistic
    and p-value, the Nemenyi critical difference, and pairwise comparisons.

    If the Friedman omnibus test is not significant, pairwise results are
    still returned but `friedman_significant` is False - post-hoc
    comparisons after a non-significant omnibus test are not meaningful,
    and the flag makes that impossible to overlook.

    Degenerate input - every model scoring identically on every series -
    yields `nan` for both the statistic and the p-value. That was verified
    empirically against scipy 1.17.1, which returns nan with a
    RuntimeWarning rather than raising. `nan < alpha` evaluates False, so
    `friedman_significant` is correctly False and no special-casing is
    needed. This is pinned by test_friedman_does_not_fire_on_identical_models,
    which will fail loudly if a future scipy raises instead - preferable to
    swallowing the exception here, which would also hide unrelated errors.
    """
    if results.isna().any().any():
        raise ValueError(
            "results grid must be complete - every model scored on every "
            "series. Missing cells invalidate the Nemenyi critical difference."
        )
    k = results.shape[1]
    n = results.shape[0]
    if k < 3:
        raise ValueError(f"Friedman needs at least 3 models, got {k}")
    if n < 1:
        raise ValueError(f"need at least 1 series, got {n}")

    # Rank WITHIN each series, ascending so rank 1 = lowest error = best.
    # Ties get average ranks.
    ranks = np.vstack([rankdata(row) for row in results.to_numpy()])
    mean_ranks = {
        model: float(ranks[:, i].mean()) for i, model in enumerate(results.columns)
    }

    stat, p_value = friedmanchisquare(*[results[c].to_numpy() for c in results.columns])
    significant = bool(p_value < alpha)

    cd = nemenyi_critical_difference(k, n, alpha)

    pairwise = {}
    for a, b in itertools.combinations(results.columns, 2):
        diff = abs(mean_ranks[a] - mean_ranks[b])
        pairwise[f"{a}_vs_{b}"] = {
            "rank_difference": round(diff, 4),
            "significant": bool(diff >= cd),
        }

    return {
        "n_series": int(n),
        "n_models": int(k),
        "alpha": alpha,
        "mean_ranks": mean_ranks,
        "friedman_statistic": float(stat),
        "friedman_p_value": float(p_value),
        "friedman_significant": significant,
        "critical_difference": cd,
        "pairwise": pairwise,
    }

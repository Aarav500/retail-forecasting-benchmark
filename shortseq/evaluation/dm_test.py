"""Diebold-Mariano test for equal predictive accuracy, plus Bonferroni correction.

Ported from `code/experiment.py:diebold_mariano_test`.
"""
import numpy as np
from scipy import stats


def diebold_mariano_test(actual, pred1, pred2, h: int = 1) -> tuple[float, float]:
    """DM test statistic and two-sided p-value. H0: pred1 and pred2 are equally accurate."""
    e1 = np.asarray(actual, dtype=float) - np.asarray(pred1, dtype=float)
    e2 = np.asarray(actual, dtype=float) - np.asarray(pred2, dtype=float)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    d_mean = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for lag in range(1, h):
        gamma_l = np.cov(d[lag:], d[:-lag])[0, 1]
        gamma_sum += (1 - lag / h) * gamma_l
    var_d = (gamma0 + 2 * gamma_sum) / n
    if var_d <= 0:
        return 0.0, 1.0
    dm_stat = d_mean / np.sqrt(var_d)
    p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return round(float(dm_stat), 4), round(float(p_val), 4)


def bonferroni_correct(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Bonferroni-correct a set of p-values from multiple pairwise DM tests."""
    m = len(p_values)
    corrected_alpha = alpha / m if m > 0 else alpha
    return {
        name: {"p_value": p, "significant": p < corrected_alpha}
        for name, p in p_values.items()
    }

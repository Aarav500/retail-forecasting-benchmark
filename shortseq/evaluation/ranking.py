"""Statistical ranking across many models/series — Friedman test + Nemenyi post-hoc.

Not yet implemented: meaningful only with the full 27+ model x 500+
series grid from later phases of the execution plan. Implementing it
now against the current baselines would be unvalidated code with no
real analysis to check it against.
"""
import pandas as pd


def friedman_nemenyi(results: pd.DataFrame):
    """results: rows=series, columns=models, values=metric (e.g. RMSE).
    Returns the Friedman statistic, p-value, and pairwise Nemenyi results.
    """
    raise NotImplementedError(
        "Friedman/Nemenyi ranking is planned for the multi-model "
        "experiment campaign phase of the NeurIPS execution plan, not "
        "this scaffolding pass."
    )

"""Boundary-map analysis — logistic regression P(foundation model beats ARIMA | n, AC1).

Not yet implemented: requires foundation model results across many
(n, AC1) combinations, which don't exist until the foundation-model and
synthetic-data phases of the execution plan are done.
"""
import pandas as pd


def fit_boundary(results: pd.DataFrame):
    """results: one row per (series, model) with columns n, ac1, and a
    boolean fm_beats_arima. Fits
    P(fm_beats_arima) = sigmoid(b1*n + b2*ac1 + b3*n*ac1).
    """
    raise NotImplementedError(
        "Boundary analysis is planned once foundation model results "
        "exist across a range of (n, AC(1)) — see the NeurIPS execution "
        "plan's Boundary Identification Protocol."
    )

import numpy as np
import pytest

from shortseq.evaluation.metrics import compute_metrics, crps, coverage
from shortseq.evaluation.dm_test import diebold_mariano_test, bonferroni_correct


def test_compute_metrics_perfect_forecast():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([10.0, 20.0, 30.0])
    m = compute_metrics(actual, predicted, "Test", train_time=0.1, pred_time=0.2)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["model"] == "Test"


def test_compute_metrics_known_error():
    actual = np.array([10.0, 20.0])
    predicted = np.array([12.0, 18.0])
    m = compute_metrics(actual, predicted, "Test", train_time=0.0, pred_time=0.0)
    assert m["rmse"] == 2.0
    assert m["mae"] == 2.0


def test_crps_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        crps(np.array([1.0]), None)


def test_coverage_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        coverage(np.array([1.0]), None, level=0.8)


def test_dm_test_identical_predictions_gives_zero_stat():
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    dm_stat, p_val = diebold_mariano_test(actual, pred, pred, h=1)
    assert dm_stat == 0.0
    assert p_val == 1.0


def test_bonferroni_correct_scales_alpha():
    p_values = {"A": 0.02, "B": 0.04}
    result = bonferroni_correct(p_values, alpha=0.05)
    assert result["A"]["significant"] is True
    assert result["B"]["significant"] is False

import pandas as pd
import pytest

from shortseq.evaluation.ranking import friedman_nemenyi
from shortseq.analysis.boundary import fit_boundary


def test_friedman_nemenyi_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        friedman_nemenyi(pd.DataFrame())


def test_fit_boundary_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        fit_boundary(pd.DataFrame())

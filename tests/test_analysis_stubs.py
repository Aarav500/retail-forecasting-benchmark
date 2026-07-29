import pandas as pd
import pytest

from shortseq.analysis.boundary import fit_boundary


def test_fit_boundary_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        fit_boundary(pd.DataFrame())

"""Dataset contract shared by every ShortSeq data loader."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class SeriesDataset:
    name: str
    series: pd.Series  # datetime-indexed
    freq: str  # 'D', 'W', or 'M'
    n: int
    cv: float
    ac1: float
    zero_frac: float
    real: bool  # True for real data, False for calibrated/synthetic


def compute_cv(series: pd.Series) -> float:
    return float(series.std() / series.mean())


def compute_ac1(series: pd.Series) -> float:
    return float(acf(series.values, nlags=1, fft=False)[1])


def compute_zero_frac(series: pd.Series) -> float:
    return float((series == 0).mean())


def make_dataset(name: str, series: pd.Series, freq: str, real: bool) -> SeriesDataset:
    return SeriesDataset(
        name=name,
        series=series,
        freq=freq,
        n=len(series),
        cv=compute_cv(series),
        ac1=compute_ac1(series),
        zero_frac=compute_zero_frac(series),
        real=real,
    )

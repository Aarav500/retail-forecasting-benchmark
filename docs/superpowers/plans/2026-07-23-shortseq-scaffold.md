# ShortSeq Package Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the flat TMLR-submission scripts in `code/` into a pip-installable `shortseq/` package with real, tested interfaces (`BaseForecaster`, `SeriesDataset`) that later NeurIPS-campaign work (foundation models, new datasets, new metrics) can plug into — without fabricating any results or untested code.

**Architecture:** `shortseq/{datasets,models,evaluation,analysis,visualization}` package, ported model-by-model and loader-by-loader from `code/experiment.py` and friends, validated by pytest against the real data already in `data/`. Not-yet-implemented pieces (8 foundation models, Friedman/Nemenyi ranking, boundary analysis, CRPS/coverage) get real stub classes/functions that raise `NotImplementedError` with a pointer to the execution plan, so the registry and tests can already reference them by name.

**Tech Stack:** Python 3.11 (venv), statsmodels/pmdarima (ARIMA/SARIMA), Prophet, XGBoost, TensorFlow/Keras (LSTM), scikit-learn, pandas/numpy, matplotlib/seaborn, pytest.

---

## Important context for whoever executes this plan

- Repo root: `C:\Users\aarav\OneDrive\Desktop\Incomplete papers\retail-forecasting-benchmark` (a real git repo, cloned from `github.com/Aarav500/retail-forecasting-benchmark`).
- The current HEAD before this plan started is tagged `tmlr-submission` — the exact TMLR-submitted state is always recoverable via `git checkout tmlr-submission`, regardless of what happens to `code/`.
- **Known bug fixed during this migration:** `code/experiment.py`'s `run_dataset_experiment` (around line 462) computes the Diebold-Mariano test for every non-ARIMA model using `res['residuals'][::-1][:len(test)]` as the second prediction series — that's the model's *residuals, time-reversed*, not its *predictions*. This silently fed the DM test a scrambled quantity instead of real forecast errors, so the DM p-values in the TMLR submission likely don't mean what they claim to (RMSE/MAE numbers are unaffected). This plan's `experiments/scripts/run_baselines.py` (Task 13) fixes this by passing each model's actual predictions into the DM test. This does **not** modify anything in the tagged `tmlr-submission` state — it only affects the new package going forward. The user has already confirmed this fix (not a faithful bug-for-bug port).
- If any `pip install` step fails or any library behaves unexpectedly (e.g. a wheel doesn't exist for Python 3.11 on Windows), **stop and report the exact error** rather than silently swapping library versions or algorithms — the whole point of this migration is that the ported code produces the same numbers as `paper.pdf`, and silently changing a dependency version could change results without anyone noticing.
- **Version risk already observed:** `requirements.txt` has no upper version bounds, so the verified install (Task 1) pulled brand-new majors: pandas 3.0.5, numpy 2.4.6, tensorflow 2.21.0, prophet 1.3.0, pmdarima 2.1.1, statsmodels 0.14.6, scikit-learn 1.9.0 — all considerably newer than whatever produced `paper.pdf`'s numbers. This is a real risk beyond ordinary run-to-run nondeterminism (e.g. pandas 3.0 changed several defaults from 2.x). If RMSE numbers diverge substantially from the paper in Task 15's verification step, check whether a version-specific behavior change is the cause before assuming the port itself is wrong — but do not preemptively downgrade anything unless a concrete divergence shows up.
- Every "run test, verify it fails/passes" step gives an exact command and what to expect. If the actual output doesn't match, do not move on — diagnose before continuing (see `superpowers:systematic-debugging` if unsure how).

---

### Task 1: Environment setup

**Files:** none (environment only)

**Already done and verified while writing this plan** (documented here so the executor doesn't redo it and knows why the venv lives outside the repo):

1. Created a Python 3.11 venv at `.venv` inside the repo. Installing `requirements.txt` there failed with `OSError: [Errno 2] No such file or directory` on a `jupyterlab` static asset path, then (after excluding jupyter) on a TensorFlow C++ header path (`tensorflow/include/external/envoy_api/...`). Root cause: Windows' 260-character path limit, combined with this repo's already-long location (`OneDrive\Desktop\Incomplete papers\retail-forecasting-benchmark\.venv\...`) and both jupyterlab's and TensorFlow's own deeply-nested internal paths.
2. Rather than enabling Windows Long Path support (a system-settings change, and the user chose not to do that), the fix was to put the venv at a short path outside the repo entirely: `C:\venvs\shortseq`. The broken in-repo `.venv` was deleted.
3. `C:\venvs\shortseq` was created and `pip install --upgrade pip` succeeded there.

**If `C:\venvs\shortseq\Scripts\python.exe` already exists, skip to Step 2.** Otherwise, recreate it:

- [ ] **Step 1: Create the virtualenv at the short path**

Run:
```bash
"/c/Users/aarav/AppData/Local/Programs/Python/Python311/python.exe" -m venv "C:/venvs/shortseq"
C:/venvs/shortseq/Scripts/python.exe -m pip install --upgrade pip -q
```
Expected: no errors; `C:/venvs/shortseq/Scripts/python.exe` exists.

- [ ] **Step 2: Install existing requirements plus new dev/runtime deps**

Run (from the repo root):
```bash
cd "/c/Users/aarav/OneDrive/Desktop/Incomplete papers/retail-forecasting-benchmark"
C:/venvs/shortseq/Scripts/python.exe -m pip install -r requirements.txt pyyaml pytest
```
Expected: all packages install successfully (statsmodels, prophet, scikit-learn, xgboost, tensorflow, pandas, numpy, matplotlib, seaborn, pmdarima, scipy, pyyaml, pytest, jupyter, ipykernel). This was already run once at the short path during plan-writing — confirm the result before re-running (see Step 3). If it fails again even at the short path, stop and report the exact error — do not substitute a different package version without checking first.

- [ ] **Step 3: Verify every library actually imports**

Run:
```bash
C:/venvs/shortseq/Scripts/python.exe -c "import statsmodels, prophet, sklearn, xgboost, tensorflow, pandas, numpy, matplotlib, seaborn, pmdarima, scipy, yaml, pytest; print('all imports OK')"
```
Expected: `all imports OK` (TensorFlow may print informational GPU/CPU startup logs first — that's fine as long as the final line prints).

- [ ] **Step 4: Add pyyaml to requirements.txt so it's tracked**

Add a line to `requirements.txt`:
```
pyyaml>=6.0
```

- [ ] **Step 5: Record the venv location for anyone else who works on this repo**

Add a short section near the top of `README.md` (this can be folded into the bigger README rewrite in Task 15, but note it now so it isn't lost): the venv must live outside the repo / outside any deeply-nested or OneDrive-synced path (e.g. `C:\venvs\shortseq`) on Windows, because TensorFlow's and JupyterLab's own package internals exceed the 260-character path limit when combined with a long project path. Linux/macOS contributors aren't affected and can use an in-repo `.venv` as usual.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "chore: add pyyaml to requirements, document short-path venv requirement on Windows"
```

(`.venv/` must NOT be committed — confirm it's covered by `.gitignore`; if not, add it.)

---

### Task 2: Package skeleton + pyproject.toml

**Files:**
- Create: `shortseq/__init__.py`
- Create: `shortseq/datasets/__init__.py`
- Create: `shortseq/models/__init__.py`
- Create: `shortseq/models/foundation/__init__.py`
- Create: `shortseq/evaluation/__init__.py`
- Create: `shortseq/analysis/__init__.py`
- Create: `shortseq/visualization/__init__.py`
- Create: `pyproject.toml`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create the package directories and empty `__init__.py` files**

```bash
mkdir -p shortseq/datasets shortseq/models/foundation shortseq/evaluation shortseq/analysis shortseq/visualization tests
```

`shortseq/__init__.py`:
```python
"""ShortSeq — a benchmark for foundation model evaluation on short time series."""

__version__ = "0.1.0"
```

`shortseq/datasets/__init__.py`, `shortseq/models/__init__.py`, `shortseq/models/foundation/__init__.py` (placeholder — replaced with real content in Task 10), `shortseq/evaluation/__init__.py`, `shortseq/analysis/__init__.py`, `shortseq/visualization/__init__.py`, `tests/__init__.py` — each an empty file for now (they become real docstring modules or stay empty; `foundation/__init__.py` gets its real docstring in Task 10).

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "shortseq"
version = "0.1.0"
description = "Benchmark for foundation model evaluation on short time series"
readme = "README.md"
license = {text = "MIT"}
authors = [{name = "Aarav Shah", email = "ashah264@ucr.edu"}]
requires-python = ">=3.10"
dependencies = [
    "statsmodels>=0.14.0",
    "prophet>=1.1.5",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "tensorflow>=2.13.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "pmdarima>=2.0.4",
    "scipy>=1.11.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4.0"]

[tool.setuptools.packages.find]
include = ["shortseq*"]
```

- [ ] **Step 3: Install the package in editable mode and verify**

```bash
C:/venvs/shortseq/Scripts/python.exe -m pip install -e .
C:/venvs/shortseq/Scripts/python.exe -c "import shortseq; print(shortseq.__version__)"
```
Expected: `0.1.0`

- [ ] **Step 4: Commit**

```bash
git add shortseq pyproject.toml tests/__init__.py
git commit -m "feat: add shortseq package skeleton, pip-installable"
```

---

### Task 3: Core interfaces — `BaseForecaster`, `SeriesDataset`

**Files:**
- Create: `shortseq/models/base.py`
- Create: `shortseq/datasets/base.py`
- Test: `tests/test_base.py`

**Design note (deviation from the brainstorming spec, disclosed):** the spec sketched `BaseForecaster.predict(self, h: int) -> Forecast` as an illustrative batch interface. The actual TMLR baselines (ARIMA/XGBoost/LSTM/Hybrid) all use a *rolling one-step-ahead* protocol — predict one step, then feed the true value back in before predicting the next — which is what the paper's methodology and DM tests depend on. Implementing an unused batch `predict(h)` now would be speculative (YAGNI); it's deferred to when zero-shot foundation models are actually wrapped. `BaseForecaster` therefore defines `fit(train)` and `predict_rolling(test)` instead.

- [ ] **Step 1: Write the failing test**

`tests/test_base.py`:
```python
import numpy as np
import pandas as pd
import pytest

from shortseq.models.base import BaseForecaster, Forecast
from shortseq.datasets.base import SeriesDataset, compute_cv, compute_ac1, compute_zero_frac, make_dataset


def test_forecast_holds_point_and_optional_dist():
    fc = Forecast(point=np.array([1.0, 2.0, 3.0]))
    assert fc.dist is None
    assert list(fc.point) == [1.0, 2.0, 3.0]


def test_base_forecaster_is_abstract():
    with pytest.raises(TypeError):
        BaseForecaster()


def test_compute_cv():
    series = pd.Series([10.0, 20.0, 30.0])
    assert compute_cv(series) == pytest.approx(series.std() / series.mean())


def test_compute_zero_frac():
    series = pd.Series([0.0, 1.0, 0.0, 2.0])
    assert compute_zero_frac(series) == pytest.approx(0.5)


def test_compute_ac1_of_constant_series_is_not_nan_or_errors_gracefully():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ac1 = compute_ac1(series)
    assert isinstance(ac1, float)


def test_make_dataset_builds_series_dataset_with_stats():
    series = pd.Series([1.0, 2.0, 3.0, 0.0, 5.0], index=pd.date_range("2020-01-01", periods=5, freq="D"))
    ds = make_dataset("toy", series, freq="D", real=True)
    assert isinstance(ds, SeriesDataset)
    assert ds.name == "toy"
    assert ds.n == 5
    assert ds.real is True
    assert ds.zero_frac == pytest.approx(0.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shortseq.models.base'` (or similar — the modules don't exist yet).

- [ ] **Step 3: Write `shortseq/models/base.py`**

```python
"""Common model interface for every ShortSeq baseline and (later) foundation model.

Follows the rolling one-step-ahead protocol used throughout the TMLR
benchmark: `fit` trains on the training series, then `predict_rolling`
forecasts each point in the test series one step at a time, feeding the
true value back into the model's history before the next step (no
look-ahead).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Forecast:
    """Point forecast plus an optional predictive distribution.

    `dist` is reserved for probabilistic models (e.g. foundation models)
    that produce more than a point estimate; unset for classical/ML
    baselines, none of which currently produce one.
    """
    point: np.ndarray
    dist: Optional[object] = None


class BaseForecaster(ABC):
    def __init__(self) -> None:
        self.name: str = self.__class__.__name__
        self.train_time_: Optional[float] = None
        self.pred_time_: Optional[float] = None

    @abstractmethod
    def fit(self, train: pd.Series) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict_rolling(self, test: pd.Series) -> Forecast:
        ...
```

- [ ] **Step 4: Write `shortseq/datasets/base.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_base.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add shortseq/models/base.py shortseq/datasets/base.py tests/test_base.py
git commit -m "feat: add BaseForecaster and SeriesDataset core interfaces"
```

---

### Task 4: Dataset loaders (D-Mart, UCI, Walmart, M5, M4, registry)

**Files:**
- Create: `shortseq/datasets/dmart.py`
- Create: `shortseq/datasets/uci.py`
- Create: `shortseq/datasets/walmart.py`
- Create: `shortseq/datasets/m5.py`
- Create: `shortseq/datasets/m4.py`
- Create: `shortseq/datasets/registry.py`
- Test: `tests/test_datasets.py`

- [ ] **Step 1: Write the failing test**

`tests/test_datasets.py`:
```python
import pytest

from shortseq.datasets.dmart import load_dmart
from shortseq.datasets.uci import load_uci
from shortseq.datasets.walmart import load_walmart
from shortseq.datasets.m5 import load_m5
from shortseq.datasets.m4 import load_m4
from shortseq.datasets.registry import load_all, load_regime, load_source


def test_load_dmart_matches_paper_stats():
    datasets = load_dmart()
    assert set(datasets) == {"dmart_food", "dmart_electronics", "dmart_clothing", "dmart_furniture"}
    food = datasets["dmart_food"]
    assert food.n == 181
    assert food.real is True
    assert 0.015 < food.cv < 0.03


def test_load_uci_matches_paper_stats():
    datasets = load_uci()
    assert len(datasets) == 5
    for ds in datasets.values():
        assert ds.n == 53
        assert ds.real is True


def test_load_walmart_matches_paper_stats():
    datasets = load_walmart()
    walmart = datasets["walmart"]
    assert walmart.n == 143


def test_load_m5_matches_paper_stats():
    datasets = load_m5()
    m5 = datasets["m5"]
    assert m5.n == 365
    assert 0.30 < m5.zero_frac < 0.35


def test_load_m4_returns_24_series():
    datasets = load_m4()
    assert len(datasets) == 24
    for ds in datasets.values():
        assert 60 <= ds.n <= 200


def test_load_all_combines_every_implemented_source():
    datasets = load_all()
    assert len(datasets) == 4 + 5 + 1 + 1 + 24


def test_load_regime_filters_by_cv():
    datasets = load_regime(cv_range=(0, 0.05))
    assert all(ds.cv <= 0.05 for ds in datasets.values())
    assert "dmart_food" in datasets


def test_load_source_raises_for_planned_but_unimplemented():
    with pytest.raises(NotImplementedError):
        load_source("favorita")


def test_load_source_raises_for_unknown_name():
    with pytest.raises(ValueError):
        load_source("not_a_real_source")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_datasets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shortseq.datasets.dmart'`.

- [ ] **Step 3: Write `shortseq/datasets/dmart.py`**

```python
"""D-Mart real retail data (India, daily) — 4 product-category series.

Ported from the loading logic in `code/real_experiment.py`.
"""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset

CATEGORIES = ["food", "electronics", "clothing", "furniture"]


def load_dmart() -> dict[str, SeriesDataset]:
    """Load the 4 real D-Mart category series (n=181 each, daily)."""
    out: dict[str, SeriesDataset] = {}
    for cat in CATEGORIES:
        df = pd.read_csv(DATA_DIR / f"real_{cat}.csv", parse_dates=["ds"])
        series = df.set_index("ds")["y"].astype(float)
        name = f"dmart_{cat}"
        out[name] = make_dataset(name, series, freq="D", real=True)
    return out
```

- [ ] **Step 4: Write `shortseq/datasets/uci.py`**

```python
"""UCI Online Retail real data (UK, weekly) — 5 product/aggregate series."""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset

FILES = {
    "uci_jumbo_bag": "uci_jumbo_bag.csv",
    "uci_lunch_bag": "uci_lunch_bag.csv",
    "uci_red_retrospot": "uci_red_retrospot.csv",
    "uci_regency_cakestand": "uci_regency_cakestand.csv",
    "uci_total": "uci_retail_weekly.csv",
}


def load_uci() -> dict[str, SeriesDataset]:
    """Load the 5 real UCI Online Retail series (n=53 each, weekly)."""
    out: dict[str, SeriesDataset] = {}
    for name, fname in FILES.items():
        df = pd.read_csv(DATA_DIR / fname, parse_dates=["ds"])
        series = df.set_index("ds")["y"].astype(float)
        out[name] = make_dataset(name, series, freq="W", real=True)
    return out
```

- [ ] **Step 5: Write `shortseq/datasets/walmart.py`**

```python
"""Walmart-calibrated weekly data (US) — store 1, matching the paper's
single-series analysis ('Use store 1 as primary series' in
code/experiment.py:generate_walmart_data)."""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset


def load_walmart() -> dict[str, SeriesDataset]:
    """Load the Walmart-calibrated series (n=143, weekly, store 1)."""
    df = pd.read_csv(DATA_DIR / "walmart.csv", parse_dates=["ds"])
    df = df[df["store"] == 1]
    series = df.set_index("ds")["y"].astype(float)
    return {"walmart": make_dataset("walmart", series, freq="W", real=False)}
```

- [ ] **Step 6: Write `shortseq/datasets/m5.py`**

```python
"""M5-calibrated intermittent daily demand series."""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset


def load_m5() -> dict[str, SeriesDataset]:
    """Load the M5-calibrated series (n=365, daily, ~32.9% zero-demand)."""
    df = pd.read_csv(DATA_DIR / "m5.csv", parse_dates=["ds"])
    series = df.set_index("ds")["y"].astype(float)
    return {"m5": make_dataset("m5", series, freq="D", real=False)}
```

- [ ] **Step 7: Write `shortseq/datasets/m4.py`**

```python
"""M4 Micro Monthly series — 24 real series sampled across 3 CV bins.

`m4_monthly_train.csv` is an R-exported wide table (index_col=0 gives the
series id as the row label; remaining columns are the observations,
dropna-trimmed to each series' actual length). M4 doesn't publish real
calendar dates for these series, so a synthetic monthly index is used
purely so downstream code can rely on a DatetimeIndex — the values
themselves are the real M4 competition data.
"""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset


def load_m4() -> dict[str, SeriesDataset]:
    """Load the 24 sampled M4 Micro Monthly series (n=68-197, monthly)."""
    train_df = pd.read_csv(DATA_DIR / "m4_monthly_train.csv", index_col=0)
    sample = pd.read_csv(DATA_DIR / "m4_sample_ids.csv")

    out: dict[str, SeriesDataset] = {}
    for _, row in sample.iterrows():
        series_id = row["id"]
        values = train_df.loc[series_id].dropna().values.astype(float)
        index = pd.period_range("2000-01", periods=len(values), freq="M").to_timestamp()
        series = pd.Series(values, index=index)
        name = f"m4_{series_id}"
        out[name] = make_dataset(name, series, freq="M", real=True)
    return out
```

- [ ] **Step 8: Write `shortseq/datasets/registry.py`**

```python
"""Dataset registry — dispatches to individual loaders and filters by regime."""
from .base import SeriesDataset
from .dmart import load_dmart
from .m4 import load_m4
from .m5 import load_m5
from .uci import load_uci
from .walmart import load_walmart

_LOADERS = {
    "dmart": load_dmart,
    "uci": load_uci,
    "walmart": load_walmart,
    "m5": load_m5,
    "m4": load_m4,
}

# Named in the NeurIPS execution plan but not yet acquired/implemented.
_PLANNED_SOURCES = [
    "favorita", "rossmann", "real_walmart", "corporacion_favorita",
    "storesales", "m4_weekly", "m3_monthly", "tourism",
]


def load_all() -> dict[str, SeriesDataset]:
    """Load every currently implemented dataset source."""
    out: dict[str, SeriesDataset] = {}
    for loader in _LOADERS.values():
        out.update(loader())
    return out


def load_regime(
    cv_range: tuple[float, float] | None = None,
    ac1_range: tuple[float, float] | None = None,
) -> dict[str, SeriesDataset]:
    """Load all implemented datasets, filtered to a CV / AC(1) regime."""
    out: dict[str, SeriesDataset] = {}
    for name, ds in load_all().items():
        if cv_range is not None and not (cv_range[0] <= ds.cv <= cv_range[1]):
            continue
        if ac1_range is not None and not (ac1_range[0] <= ds.ac1 <= ac1_range[1]):
            continue
        out[name] = ds
    return out


def load_source(source: str) -> dict[str, SeriesDataset]:
    """Load a single named source; raises for sources not yet implemented."""
    if source in _PLANNED_SOURCES:
        raise NotImplementedError(
            f"Dataset source '{source}' is on the NeurIPS execution plan but "
            "not yet acquired/implemented in this repo."
        )
    if source not in _LOADERS:
        raise ValueError(f"Unknown dataset source: {source!r}")
    return _LOADERS[source]()
```

- [ ] **Step 9: Run test to verify it passes**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_datasets.py -v`
Expected: 9 passed.

- [ ] **Step 10: Commit**

```bash
git add shortseq/datasets tests/test_datasets.py
git commit -m "feat: port dataset loaders (D-Mart, UCI, Walmart, M5, M4) into shortseq.datasets"
```

---

### Task 5: Evaluation — metrics and Diebold-Mariano test

**Files:**
- Create: `shortseq/evaluation/metrics.py`
- Create: `shortseq/evaluation/dm_test.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shortseq.evaluation.metrics'`.

- [ ] **Step 3: Write `shortseq/evaluation/metrics.py`**

```python
"""Point-forecast accuracy metrics.

`compute_metrics` is ported from `code/experiment.py:compute_metrics`.
`crps`/`coverage` are real stubs: no currently implemented model produces
a predictive distribution, so there is nothing valid to compute yet —
these are planned for the foundation-model phase of the NeurIPS
execution plan.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(actual, predicted, model_name: str, train_time: float, pred_time: float) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    residuals = actual - predicted
    return {
        "model": model_name,
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "mape": round(float(mape), 4),
        "mean_residual": round(float(np.mean(residuals)), 4),
        "residual_std": round(float(np.std(residuals)), 4),
        "train_time": round(float(train_time), 4),
        "pred_time": round(float(pred_time), 4),
        "residuals": residuals.tolist(),
    }


def crps(actual, dist) -> float:
    raise NotImplementedError(
        "CRPS requires a predictive distribution, which no currently "
        "implemented model produces. Planned for the foundation-model "
        "phase of the NeurIPS execution plan."
    )


def coverage(actual, dist, level: float) -> float:
    raise NotImplementedError(
        "Prediction-interval coverage requires a predictive distribution, "
        "which no currently implemented model produces. Planned for the "
        "foundation-model phase of the NeurIPS execution plan."
    )
```

- [ ] **Step 4: Write `shortseq/evaluation/dm_test.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_metrics.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add shortseq/evaluation/metrics.py shortseq/evaluation/dm_test.py tests/test_metrics.py
git commit -m "feat: port metrics and Diebold-Mariano test into shortseq.evaluation"
```

---

### Task 6: Classical baselines — ARIMA, SARIMA

**Files:**
- Create: `shortseq/models/_arima_utils.py`
- Create: `shortseq/models/arima.py`
- Create: `shortseq/models/sarima.py`
- Test: `tests/test_models_classical.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models_classical.py`:
```python
from shortseq.datasets.dmart import load_dmart
from shortseq.models.arima import ARIMAForecaster
from shortseq.models.sarima import SARIMAForecaster


def _split(series, frac=0.8):
    i = int(len(series) * frac)
    return series.iloc[:i], series.iloc[i:]


def test_arima_forecaster_fits_and_predicts_real_dmart_food():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = ARIMAForecaster().fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None
    assert model.pred_time_ is not None
    assert model.name.startswith("ARIMA")


def test_sarima_forecaster_fits_and_predicts_real_dmart_food():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = SARIMAForecaster().fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_models_classical.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shortseq.models.arima'`.

- [ ] **Step 3: Write `shortseq/models/_arima_utils.py`**

```python
"""Shared rolling-forecast loop for pmdarima-based models (ARIMA, SARIMA).

Ported from the update/predict loop inside `code/experiment.py:run_arima`.
"""
import time

import numpy as np


def rolling_arima_forecast(model, train_values, test_values):
    """Roll a fitted pmdarima model forward one step at a time.

    After each prediction, the true test value is appended to history and
    the model is updated with it before the next prediction — the paper's
    strict no-look-ahead rolling one-step-ahead protocol.
    """
    t0 = time.time()
    history = list(train_values)
    forecasts = []
    for _ in range(len(test_values)):
        model.update(history[-1:])
        fc = model.predict(n_periods=1)[0]
        forecasts.append(fc)
        history.append(test_values[len(forecasts) - 1])
    pred_time = time.time() - t0
    return np.array(forecasts), pred_time
```

- [ ] **Step 4: Write `shortseq/models/arima.py`**

```python
"""ARIMA baseline — auto-ARIMA with AIC-based order selection (non-seasonal).

Ported from `code/experiment.py:run_arima` (seasonal=False branch).
"""
import time

import pandas as pd
import pmdarima as pm

from ._arima_utils import rolling_arima_forecast
from .base import BaseForecaster, Forecast


class ARIMAForecaster(BaseForecaster):
    def __init__(self, max_p: int = 7, max_q: int = 7):
        super().__init__()
        self.max_p = max_p
        self.max_q = max_q
        self._model = None
        self._train_values: list[float] = []

    def fit(self, train: pd.Series) -> "ARIMAForecaster":
        t0 = time.time()
        values = train.values.astype(float)
        try:
            self._model = pm.auto_arima(
                values, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore",
                max_p=self.max_p, max_q=self.max_q,
            )
        except Exception:
            self._model = pm.auto_arima(
                values, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore",
            )
        self.name = f"ARIMA{self._model.order}"
        self._train_values = list(values)
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        test_values = test.values.astype(float)
        forecasts, pred_time = rolling_arima_forecast(self._model, self._train_values, test_values)
        self.pred_time_ = pred_time
        return Forecast(point=forecasts)
```

- [ ] **Step 5: Write `shortseq/models/sarima.py`**

```python
"""SARIMA baseline — seasonal auto-ARIMA.

Ported from `code/experiment.py:run_arima` (seasonal=True branch).
"""
import time

import pandas as pd
import pmdarima as pm

from ._arima_utils import rolling_arima_forecast
from .base import BaseForecaster, Forecast


class SARIMAForecaster(BaseForecaster):
    def __init__(self, max_p: int = 5, max_q: int = 5, max_P: int = 2, max_Q: int = 2):
        super().__init__()
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self._model = None
        self._train_values: list[float] = []

    def fit(self, train: pd.Series) -> "SARIMAForecaster":
        t0 = time.time()
        values = train.values.astype(float)
        m = 7 if len(values) > 50 else 4
        try:
            self._model = pm.auto_arima(
                values, seasonal=True, m=m, stepwise=True,
                suppress_warnings=True, error_action="ignore",
                max_p=self.max_p, max_q=self.max_q,
                max_P=self.max_P, max_Q=self.max_Q,
            )
            self.name = f"SARIMA{self._model.order}x{self._model.seasonal_order}"
        except Exception:
            self._model = pm.auto_arima(
                values, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore",
            )
            self.name = f"ARIMA{self._model.order}"
        self._train_values = list(values)
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        test_values = test.values.astype(float)
        forecasts, pred_time = rolling_arima_forecast(self._model, self._train_values, test_values)
        self.pred_time_ = pred_time
        return Forecast(point=forecasts)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_models_classical.py -v`
Expected: 2 passed. (May take ~10-30s — auto-ARIMA searches several orders.)

- [ ] **Step 7: Commit**

```bash
git add shortseq/models/_arima_utils.py shortseq/models/arima.py shortseq/models/sarima.py tests/test_models_classical.py
git commit -m "feat: port ARIMA and SARIMA baselines into shortseq.models"
```

---

### Task 7: ML baselines — XGBoost, LSTM

**Files:**
- Create: `shortseq/models/xgboost_model.py`
- Create: `shortseq/models/lstm_model.py`
- Test: `tests/test_models_ml.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models_ml.py`:
```python
from shortseq.datasets.dmart import load_dmart
from shortseq.models.xgboost_model import XGBoostForecaster, create_lag_features
from shortseq.models.lstm_model import LSTMForecaster
import numpy as np


def _split(series, frac=0.8):
    i = int(len(series) * frac)
    return series.iloc[:i], series.iloc[i:]


def test_create_lag_features_shapes():
    series = np.arange(20, dtype=float)
    X, y = create_lag_features(series, lags=5, horizon=1)
    assert X.shape == (15, 5)
    assert y.shape == (15,)
    assert list(X[0]) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert y[0] == 5.0


def test_xgboost_forecaster_fits_and_predicts_real_dmart_food():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = XGBoostForecaster().fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None


def test_lstm_forecaster_fits_and_predicts_real_dmart_food():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = LSTMForecaster(epochs=5).fit(train)  # few epochs — this is a smoke test, not a quality test
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_models_ml.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shortseq.models.xgboost_model'`.

- [ ] **Step 3: Write `shortseq/models/xgboost_model.py`**

```python
"""XGBoost baseline — gradient-boosted trees over a lagged feature window.

Ported from `code/experiment.py:run_xgboost` / `create_features_xgb`.
"""
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseForecaster, Forecast


def create_lag_features(series: np.ndarray, lags: int = 14, horizon: int = 1):
    """Build (X, y) lag-feature pairs for supervised regression on a series."""
    X, y = [], []
    for i in range(lags, len(series) - horizon + 1):
        X.append(series[i - lags:i])
        y.append(series[i + horizon - 1])
    return np.array(X), np.array(y)


class XGBoostForecaster(BaseForecaster):
    def __init__(self, lags: int = 14, n_estimators: int = 200, max_depth: int = 5,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42):
        super().__init__()
        self.lags = lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.name = "XGBoost"
        self._model = None
        self._train_values: list[float] = []

    def fit(self, train: pd.Series) -> "XGBoostForecaster":
        t0 = time.time()
        values = train.values.astype(float)
        X_train, y_train = create_lag_features(values, self.lags)
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state, verbosity=0,
        )
        self._model.fit(X_train, y_train)
        self._train_values = list(values)
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        history = list(self._train_values)
        test_values = test.values.astype(float)
        forecasts = []
        for i in range(len(test_values)):
            if len(history) >= self.lags:
                x = np.array(history[-self.lags:]).reshape(1, -1)
                pred = self._model.predict(x)[0]
            else:
                pred = np.mean(history)
            forecasts.append(pred)
            history.append(test_values[i])
        self.pred_time_ = time.time() - t0
        return Forecast(point=np.array(forecasts))
```

- [ ] **Step 4: Write `shortseq/models/lstm_model.py`**

```python
"""LSTM baseline — 2-layer LSTM over a MinMax-scaled lagged window.

Ported from `code/experiment.py:run_lstm`.
"""
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from .base import BaseForecaster, Forecast


class LSTMForecaster(BaseForecaster):
    def __init__(self, lags: int = 14, units_layer1: int = 64, units_layer2: int = 32,
                 dropout: float = 0.2, epochs: int = 100, batch_size: int = 16,
                 patience: int = 10, validation_split: float = 0.1, random_seed: int = 42):
        super().__init__()
        self.lags = lags
        self.units_layer1 = units_layer1
        self.units_layer2 = units_layer2
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.name = "LSTM"
        self._model = None
        self._scaler = None
        self._train_scaled: list[float] = []

    def fit(self, train: pd.Series) -> "LSTMForecaster":
        tf.random.set_seed(self.random_seed)
        t0 = time.time()
        values = train.values.astype(float)
        self._scaler = MinMaxScaler()
        train_scaled = self._scaler.fit_transform(values.reshape(-1, 1)).flatten()

        X, y = [], []
        for i in range(self.lags, len(train_scaled)):
            X.append(train_scaled[i - self.lags:i])
            y.append(train_scaled[i])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self._model = Sequential([
            LSTM(self.units_layer1, return_sequences=True, input_shape=(self.lags, 1)),
            Dropout(self.dropout),
            LSTM(self.units_layer2),
            Dropout(self.dropout),
            Dense(1),
        ])
        self._model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(patience=self.patience, restore_best_weights=True, verbose=0)
        self._model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                         validation_split=self.validation_split, callbacks=[es], verbose=0)
        self._train_scaled = list(train_scaled)
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        history = list(self._train_scaled)
        test_values = test.values.astype(float)
        forecasts_scaled = []
        for i in range(len(test_values)):
            x = np.array(history[-self.lags:]).reshape(1, self.lags, 1)
            pred_scaled = self._model.predict(x, verbose=0)[0, 0]
            forecasts_scaled.append(pred_scaled)
            test_scaled = self._scaler.transform([[test_values[i]]])[0, 0]
            history.append(test_scaled)

        forecasts = self._scaler.inverse_transform(
            np.array(forecasts_scaled).reshape(-1, 1)
        ).flatten()
        self.pred_time_ = time.time() - t0
        return Forecast(point=forecasts)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_models_ml.py -v`
Expected: 3 passed. (LSTM test may take a minute or two even with `epochs=5`.)

- [ ] **Step 6: Commit**

```bash
git add shortseq/models/xgboost_model.py shortseq/models/lstm_model.py tests/test_models_ml.py
git commit -m "feat: port XGBoost and LSTM baselines into shortseq.models"
```

---

### Task 8: Hybrid and seasonal-naive baselines

**Files:**
- Create: `shortseq/models/hybrid.py`
- Create: `shortseq/models/naive.py`
- Test: `tests/test_models_hybrid_naive.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models_hybrid_naive.py`:
```python
import numpy as np

from shortseq.datasets.dmart import load_dmart
from shortseq.models.hybrid import HybridForecaster
from shortseq.models.naive import SeasonalNaiveForecaster


def _split(series, frac=0.8):
    i = int(len(series) * frac)
    return series.iloc[:i], series.iloc[i:]


def test_hybrid_forecaster_fits_and_predicts_real_dmart_food():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = HybridForecaster().fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None


def test_seasonal_naive_forecaster_reproduces_last_season_values():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = SeasonalNaiveForecaster(season_length=7).fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    # First 7 rolling predictions are exactly the last 7 training values, in order.
    assert np.array_equal(forecast.point[:7], train.values[-7:])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_models_hybrid_naive.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shortseq.models.hybrid'`.

- [ ] **Step 3: Write `shortseq/models/hybrid.py`**

```python
"""Hybrid ARIMA+XGBoost baseline.

Ported from `code/experiment.py:run_hybrid`. Fits ARIMA, then an XGBoost
regressor on ARIMA's in-sample residuals; forecasts are the ARIMA
forecast plus the XGBoost residual correction.
"""
import time

import numpy as np
import pandas as pd
import pmdarima as pm
import xgboost as xgb

from .base import BaseForecaster, Forecast
from .xgboost_model import create_lag_features


class HybridForecaster(BaseForecaster):
    def __init__(self, lags: int = 14, arima_max_p: int = 5, arima_max_q: int = 5,
                 xgb_n_estimators: int = 100, xgb_max_depth: int = 4,
                 xgb_learning_rate: float = 0.1, random_state: int = 42):
        super().__init__()
        self.lags = lags
        self.arima_max_p = arima_max_p
        self.arima_max_q = arima_max_q
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.random_state = random_state
        self.name = "Hybrid(ARIMA+XGB)"
        self._arima_model = None
        self._xgb_model = None
        self._has_xgb = False
        self._train_values: list[float] = []
        self._res_history: list[float] = []

    def fit(self, train: pd.Series) -> "HybridForecaster":
        t0 = time.time()
        values = train.values.astype(float)
        try:
            self._arima_model = pm.auto_arima(
                values, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore",
                max_p=self.arima_max_p, max_q=self.arima_max_q,
            )
        except Exception:
            self._arima_model = pm.auto_arima(
                values, seasonal=False, information_criterion="aic",
                suppress_warnings=True,
            )

        arima_train_preds = self._arima_model.predict_in_sample()
        arima_residuals = values - arima_train_preds

        if len(arima_residuals) > self.lags + 5:
            X_res, y_res = create_lag_features(arima_residuals, self.lags)
            self._xgb_model = xgb.XGBRegressor(
                n_estimators=self.xgb_n_estimators, max_depth=self.xgb_max_depth,
                learning_rate=self.xgb_learning_rate, random_state=self.random_state,
                verbosity=0,
            )
            self._xgb_model.fit(X_res, y_res)
            self._has_xgb = True

        self._train_values = list(values)
        self._res_history = list(arima_residuals)
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        history = list(self._train_values)
        res_history = list(self._res_history)
        test_values = test.values.astype(float)
        forecasts = []

        for i in range(len(test_values)):
            self._arima_model.update(history[-1:])
            arima_fc = self._arima_model.predict(n_periods=1)[0]

            if self._has_xgb and len(res_history) >= self.lags:
                x_res = np.array(res_history[-self.lags:]).reshape(1, -1)
                res_correction = self._xgb_model.predict(x_res)[0]
            else:
                res_correction = 0.0

            final_fc = arima_fc + res_correction
            forecasts.append(final_fc)

            actual_val = test_values[i]
            history.append(actual_val)
            new_residual = actual_val - arima_fc
            res_history.append(new_residual)

        self.pred_time_ = time.time() - t0
        return Forecast(point=np.array(forecasts))
```

- [ ] **Step 4: Write `shortseq/models/naive.py`**

```python
"""Seasonal-naive reference baseline — forecasts each point as the value
one season-length ago.

Named as the standard benchmark reference in the NeurIPS execution
plan's baseline table; not present in the TMLR-submitted code.
"""
import time

import numpy as np
import pandas as pd

from .base import BaseForecaster, Forecast


class SeasonalNaiveForecaster(BaseForecaster):
    def __init__(self, season_length: int = 7):
        super().__init__()
        self.season_length = season_length
        self.name = f"SeasonalNaive(m={season_length})"
        self._train_values: list[float] = []

    def fit(self, train: pd.Series) -> "SeasonalNaiveForecaster":
        t0 = time.time()
        self._train_values = list(train.values.astype(float))
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        history = list(self._train_values)
        test_values = test.values.astype(float)
        m = self.season_length
        forecasts = []
        for i in range(len(test_values)):
            pred = history[-m] if len(history) >= m else float(np.mean(history))
            forecasts.append(pred)
            history.append(test_values[i])
        self.pred_time_ = time.time() - t0
        return Forecast(point=np.array(forecasts))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_models_hybrid_naive.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add shortseq/models/hybrid.py shortseq/models/naive.py tests/test_models_hybrid_naive.py
git commit -m "feat: port Hybrid baseline and add new SeasonalNaive reference model"
```

---

### Task 9: Prophet baseline

**Files:**
- Create: `shortseq/models/prophet_model.py`
- Test: `tests/test_models_prophet.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models_prophet.py`:
```python
from shortseq.datasets.dmart import load_dmart
from shortseq.models.prophet_model import ProphetForecaster


def test_prophet_forecaster_fits_and_predicts_real_dmart_food():
    ds = load_dmart()["dmart_food"]
    i = int(len(ds.series) * 0.8)
    train, test = ds.series.iloc[:i], ds.series.iloc[i:]
    model = ProphetForecaster(freq=ds.freq).fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_models_prophet.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shortseq.models.prophet_model'`.

- [ ] **Step 3: Write `shortseq/models/prophet_model.py`**

```python
"""Prophet baseline — additive decomposition with automatic seasonality.

Ported from `code/experiment.py:run_prophet`. Unlike the other baselines,
Prophet forecasts the whole test horizon in a single batch call (no
step-by-step feedback) — this matches the original benchmark exactly;
`predict_rolling` here just means "produce one point per test
timestamp", not that the model is updated between steps.
"""
import time

import numpy as np
import pandas as pd
from prophet import Prophet

from .base import BaseForecaster, Forecast


class ProphetForecaster(BaseForecaster):
    def __init__(self, freq: str = "D"):
        super().__init__()
        self.freq = freq
        self.name = "Prophet"
        self._model = None

    def fit(self, train: pd.Series) -> "ProphetForecaster":
        t0 = time.time()
        train_df = pd.DataFrame({"ds": train.index, "y": train.values})
        self._model = Prophet(
            yearly_seasonality=True, weekly_seasonality=True,
            daily_seasonality=False, seasonality_mode="additive",
            interval_width=0.95,
        )
        self._model.fit(train_df)
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        future = self._model.make_future_dataframe(periods=len(test), freq=self.freq)
        forecast = self._model.predict(future)
        preds = forecast["yhat"].values[-len(test):]
        self.pred_time_ = time.time() - t0
        return Forecast(point=np.array(preds))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_models_prophet.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add shortseq/models/prophet_model.py tests/test_models_prophet.py
git commit -m "feat: port Prophet baseline into shortseq.models"
```

---

### Task 10: Foundation model stubs

**Files:**
- Create: `shortseq/models/foundation/__init__.py` (replace placeholder from Task 2)
- Create: `shortseq/models/foundation/chronos.py`
- Create: `shortseq/models/foundation/timesfm.py`
- Create: `shortseq/models/foundation/moirai.py`
- Create: `shortseq/models/foundation/moment.py`
- Create: `shortseq/models/foundation/timer.py`
- Create: `shortseq/models/foundation/ttm.py`
- Create: `shortseq/models/foundation/lag_llama.py`
- Create: `shortseq/models/foundation/forecastpfn.py`
- Test: `tests/test_foundation_stubs.py`

None of these are runnable yet — installing/wrapping the 8 foundation models needs GPU access and new heavy dependencies (torch, huggingface_hub downloads, etc.) that are explicitly out of scope for this scaffold (see the design spec's non-goals). Each stub exists so the registry and later code can already reference every model by name and constructor signature.

- [ ] **Step 1: Write the failing test**

`tests/test_foundation_stubs.py`:
```python
import pytest

from shortseq.models.foundation.chronos import ChronosForecaster
from shortseq.models.foundation.timesfm import TimesFMForecaster
from shortseq.models.foundation.moirai import MoiraiForecaster
from shortseq.models.foundation.moment import MomentForecaster
from shortseq.models.foundation.timer import TimerForecaster
from shortseq.models.foundation.ttm import TTMForecaster
from shortseq.models.foundation.lag_llama import LagLlamaForecaster
from shortseq.models.foundation.forecastpfn import ForecastPFNForecaster

STUB_CLASSES = [
    ChronosForecaster, TimesFMForecaster, MoiraiForecaster, MomentForecaster,
    TimerForecaster, TTMForecaster, LagLlamaForecaster, ForecastPFNForecaster,
]


@pytest.mark.parametrize("cls", STUB_CLASSES)
def test_foundation_stub_raises_not_implemented_on_fit(cls):
    model = cls()
    with pytest.raises(NotImplementedError):
        model.fit(None)


@pytest.mark.parametrize("cls", STUB_CLASSES)
def test_foundation_stub_raises_not_implemented_on_predict(cls):
    model = cls()
    with pytest.raises(NotImplementedError):
        model.predict_rolling(None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_foundation_stubs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shortseq.models.foundation.chronos'`.

- [ ] **Step 3: Write `shortseq/models/foundation/__init__.py`**

```python
"""Foundation-model wrappers — not yet implemented.

The NeurIPS execution plan's Week 1 calls for installing and wrapping
Chronos, TimesFM, Moirai, Moment, Timer, TTM, Lag-Llama, and ForecastPFN,
which needs GPU access and new heavy dependencies (torch,
huggingface_hub downloads, etc.) not set up in this scaffolding pass.
Each module here defines a BaseForecaster subclass with the right name
and constructor signature so the registry and tests can already
reference every model by name; `fit`/`predict_rolling` raise
NotImplementedError until the real wrapper is built.
"""
```

- [ ] **Step 4: Write `shortseq/models/foundation/chronos.py`**

```python
"""Chronos (Amazon) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "Chronos is not yet installed/wrapped — see the NeurIPS execution "
    "plan, Week 1 (foundation model installation)."
)


class ChronosForecaster(BaseForecaster):
    """Zero-shot Chronos forecaster. Sizes per the execution plan:
    tiny(8M), mini(20M), small(46M), base(200M), large(710M)."""

    def __init__(self, size: str = "small"):
        super().__init__()
        self.size = size
        self.name = f"Chronos-{size}"

    def fit(self, train: pd.Series) -> "ChronosForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)
```

- [ ] **Step 5: Write `shortseq/models/foundation/timesfm.py`**

```python
"""TimesFM (Google) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "TimesFM is not yet installed/wrapped — see the NeurIPS execution "
    "plan, Week 1 (foundation model installation)."
)


class TimesFMForecaster(BaseForecaster):
    """Zero-shot TimesFM forecaster. 200M-parameter patched model per the
    execution plan."""

    def __init__(self, size: str = "200m"):
        super().__init__()
        self.size = size
        self.name = f"TimesFM-{size}"

    def fit(self, train: pd.Series) -> "TimesFMForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)
```

- [ ] **Step 6: Write `shortseq/models/foundation/moirai.py`**

```python
"""Moirai (Salesforce) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "Moirai is not yet installed/wrapped — see the NeurIPS execution "
    "plan, Week 1 (foundation model installation)."
)


class MoiraiForecaster(BaseForecaster):
    """Zero-shot Moirai forecaster. Sizes per the execution plan:
    small(14M), base(91M), large(311M)."""

    def __init__(self, size: str = "small"):
        super().__init__()
        self.size = size
        self.name = f"Moirai-{size}"

    def fit(self, train: pd.Series) -> "MoiraiForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)
```

- [ ] **Step 7: Write `shortseq/models/foundation/moment.py`**

```python
"""Moment (CMU) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "Moment is not yet installed/wrapped — see the NeurIPS execution "
    "plan, Week 1 (foundation model installation)."
)


class MomentForecaster(BaseForecaster):
    """Zero-shot Moment forecaster. Sizes per the execution plan:
    small(40M), base(125M), large(385M)."""

    def __init__(self, size: str = "small"):
        super().__init__()
        self.size = size
        self.name = f"Moment-{size}"

    def fit(self, train: pd.Series) -> "MomentForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)
```

- [ ] **Step 8: Write `shortseq/models/foundation/timer.py`**

```python
"""Timer (THU) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "Timer is not yet installed/wrapped — see the NeurIPS execution "
    "plan, Week 1 (foundation model installation)."
)


class TimerForecaster(BaseForecaster):
    """Zero-shot Timer forecaster. Sizes per the execution plan:
    base(50M), large(670M)."""

    def __init__(self, size: str = "base"):
        super().__init__()
        self.size = size
        self.name = f"Timer-{size}"

    def fit(self, train: pd.Series) -> "TimerForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)
```

- [ ] **Step 9: Write `shortseq/models/foundation/ttm.py`**

```python
"""TTM / Tiny Time Mixers (IBM) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "TTM is not yet installed/wrapped — see the NeurIPS execution plan, "
    "Week 1 (foundation model installation)."
)


class TTMForecaster(BaseForecaster):
    """Zero-shot Tiny Time Mixers forecaster. Context-horizon configs per
    the execution plan: 512-96, 1024-96, 1536-96."""

    def __init__(self, config: str = "512-96"):
        super().__init__()
        self.config = config
        self.name = f"TTM-{config}"

    def fit(self, train: pd.Series) -> "TTMForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)
```

- [ ] **Step 10: Write `shortseq/models/foundation/lag_llama.py`**

```python
"""Lag-Llama (ServiceNow) — stub. Not yet implemented; see
shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "Lag-Llama is not yet installed/wrapped — see the NeurIPS execution "
    "plan, Week 1 (foundation model installation)."
)


class LagLlamaForecaster(BaseForecaster):
    """Zero-shot Lag-Llama forecaster (6M parameters, single size per the
    execution plan)."""

    def __init__(self):
        super().__init__()
        self.name = "Lag-Llama"

    def fit(self, train: pd.Series) -> "LagLlamaForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)
```

- [ ] **Step 11: Write `shortseq/models/foundation/forecastpfn.py`**

```python
"""ForecastPFN (prior-data fitted network) — stub. Not yet implemented;
see shortseq/models/foundation/__init__.py."""
import pandas as pd

from ..base import BaseForecaster, Forecast

_NOT_IMPLEMENTED = (
    "ForecastPFN is not yet installed/wrapped — see the NeurIPS "
    "execution plan, Week 1 (foundation model installation)."
)


class ForecastPFNForecaster(BaseForecaster):
    """Zero-shot ForecastPFN forecaster (25M parameters, single size per
    the execution plan)."""

    def __init__(self):
        super().__init__()
        self.name = "ForecastPFN"

    def fit(self, train: pd.Series) -> "ForecastPFNForecaster":
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def predict_rolling(self, test: pd.Series) -> Forecast:
        raise NotImplementedError(_NOT_IMPLEMENTED)
```

- [ ] **Step 12: Run test to verify it passes**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_foundation_stubs.py -v`
Expected: 16 passed.

- [ ] **Step 13: Commit**

```bash
git add shortseq/models/foundation tests/test_foundation_stubs.py
git commit -m "feat: add foundation-model stub interfaces (Chronos, TimesFM, Moirai, Moment, Timer, TTM, Lag-Llama, ForecastPFN)"
```

---

### Task 11: New-analysis stubs — ranking and boundary analysis

**Files:**
- Create: `shortseq/evaluation/ranking.py`
- Create: `shortseq/analysis/boundary.py`
- Test: `tests/test_analysis_stubs.py`

- [ ] **Step 1: Write the failing test**

`tests/test_analysis_stubs.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_analysis_stubs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shortseq.evaluation.ranking'`.

- [ ] **Step 3: Write `shortseq/evaluation/ranking.py`**

```python
"""Statistical ranking across many models/series — Friedman test + Nemenyi post-hoc.

Not yet implemented: meaningful only with the full 27+ model x 500+
series grid from later phases of the execution plan. Implementing it
now against the 6 current baselines would be unvalidated code with no
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
```

- [ ] **Step 4: Write `shortseq/analysis/boundary.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_analysis_stubs.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add shortseq/evaluation/ranking.py shortseq/analysis/boundary.py tests/test_analysis_stubs.py
git commit -m "feat: add ranking and boundary-analysis stub interfaces"
```

---

### Task 12: Visualization — reproducible figures

**Files:**
- Create: `shortseq/visualization/figures.py`
- Test: `tests/test_figures.py`

**Scope note:** ported from `code/figures_v2.py`: `fig_master_heatmap`, `fig_regime_scatter`, `fig_real_residuals`, `fig_dm_summary`. Two functions are intentionally NOT ported: `fig_real_forecasts` (re-fits three models live at figure-generation time — expensive, and redundant with what `experiments/scripts/run_baselines.py` already produces) and `fig_failure_all` (duplicates information already in `fig_master_heatmap`). `code/figures_new_datasets.py`'s fig11/fig12 are also not ported — they plot hardcoded literal numbers transcribed from the paper rather than computing from `results`/`datasets`, which isn't appropriate for a package whose whole point is reproducibility; `fig_regime_scatter` below is extended to include UCI computed dynamically instead.

- [ ] **Step 1: Write the failing test**

`tests/test_figures.py`:
```python
from pathlib import Path

import pandas as pd

from shortseq.datasets.base import SeriesDataset
from shortseq.visualization.figures import (
    fig_master_heatmap, fig_regime_scatter, fig_real_residuals, fig_dm_summary,
)

FAKE_RESULTS = {
    "dmart_food": {
        "metrics": {
            "ARIMA": {"rmse": 100.0, "residual_std": 10.0},
            "Hybrid": {"rmse": 90.0, "residual_std": 9.0},
        },
        "dm_tests": {"Hybrid": {"p_value": 0.01}},
    },
    "walmart": {
        "metrics": {
            "ARIMA": {"rmse": 200.0, "residual_std": 20.0},
            "Hybrid": {"rmse": 150.0, "residual_std": 15.0},
        },
        "dm_tests": {"Hybrid": {"p_value": 0.2}},
    },
}
FAKE_DATASETS = {
    "dmart_food": SeriesDataset(
        name="dmart_food", series=pd.Series([1.0, 2.0, 3.0]), freq="D",
        n=3, cv=0.02, ac1=-0.1, zero_frac=0.0, real=True,
    ),
    "walmart": SeriesDataset(
        name="walmart", series=pd.Series([1.0, 2.0, 3.0]), freq="W",
        n=3, cv=0.3, ac1=0.5, zero_frac=0.0, real=False,
    ),
}


def test_fig_master_heatmap_writes_output_files(tmp_path):
    fig_master_heatmap(FAKE_RESULTS, tmp_path)
    assert (tmp_path / "fig1_master_heatmap.png").exists()
    assert (tmp_path / "fig1_master_heatmap.pdf").exists()


def test_fig_regime_scatter_writes_output_files(tmp_path):
    fig_regime_scatter(FAKE_RESULTS, FAKE_DATASETS, tmp_path)
    assert (tmp_path / "fig_regime_scatter.png").exists()


def test_fig_real_residuals_writes_output_files(tmp_path):
    fig_real_residuals(FAKE_RESULTS, ["dmart_food", "walmart"], tmp_path)
    assert (tmp_path / "fig_real_residuals.png").exists()


def test_fig_dm_summary_writes_output_files(tmp_path):
    fig_dm_summary(FAKE_RESULTS, ["dmart_food", "walmart"], tmp_path)
    assert (tmp_path / "fig_dm_summary.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_figures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shortseq.visualization.figures'`.

- [ ] **Step 3: Write `shortseq/visualization/figures.py`**

```python
"""Reproducible figure generation from saved results + the dataset registry.

Ported from `code/figures_v2.py`. See this plan's Task 12 scope note for
which functions were dropped and why.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shortseq.datasets.base import SeriesDataset

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})
COLORS = {
    "ARIMA": "#2196F3", "SARIMA": "#4CAF50", "Prophet": "#FF9800",
    "XGBoost": "#9C27B0", "LSTM": "#F44336", "Hybrid": "#795548", "Naive": "#607D8B",
}


def fig_master_heatmap(results: dict, figures_dir: Path):
    models = ["ARIMA", "SARIMA", "Prophet", "XGBoost", "LSTM", "Hybrid", "Naive"]
    datasets = list(results.keys())

    data = np.full((len(models), len(datasets)), np.nan)
    for j, d in enumerate(datasets):
        for i, m in enumerate(models):
            v = results.get(d, {}).get("metrics", {}).get(m, {}).get("rmse")
            if v is not None:
                data[i, j] = v

    with np.errstate(invalid="ignore"):
        col_max = np.nanmax(data, axis=0)
        norm = data / col_max

    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.4), 5))
    im = ax.imshow(norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(datasets)):
            v = data[i, j]
            if not np.isnan(v):
                txt = f"{v:.0f}" if v > 100 else f"{v:.2f}"
                color = "black" if np.isnan(norm[i, j]) or norm[i, j] < 0.65 else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Normalised RMSE", fraction=0.046, pad=0.04)
    ax.set_title("Model Performance Across Demand Regimes\n(Normalised RMSE — lower is better)",
                 fontweight="bold", pad=20)
    plt.tight_layout()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "fig1_master_heatmap.pdf", bbox_inches="tight")
    plt.savefig(figures_dir / "fig1_master_heatmap.png", bbox_inches="tight")
    plt.close()


def fig_regime_scatter(results: dict, datasets: dict[str, SeriesDataset], figures_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cvs, ac1s, ratios_hybrid, labels_plot = [], [], [], []

    for name, ds in datasets.items():
        m = results.get(name, {}).get("metrics", {})
        arima_r = m.get("ARIMA", {}).get("rmse")
        hybrid_r = m.get("Hybrid", {}).get("rmse")
        if not arima_r or not hybrid_r:
            continue
        cvs.append(ds.cv)
        ac1s.append(ds.ac1)
        ratios_hybrid.append(hybrid_r / arima_r)
        labels_plot.append(name)

    ax1 = axes[0]
    ax1.scatter(cvs, ratios_hybrid, s=100, zorder=5, edgecolors="white", lw=1.2)
    for cv, r, lbl in zip(cvs, ratios_hybrid, labels_plot):
        ax1.annotate(lbl, (cv, r), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax1.axhline(1.0, color="#2196F3", lw=2, ls="--", label="ARIMA baseline")
    ax1.set_xlabel("Coefficient of Variation (CV = σ/μ)")
    ax1.set_ylabel("Hybrid RMSE / ARIMA RMSE")
    ax1.set_title("When Does Hybrid Beat ARIMA?\n(Ratio < 1 = Hybrid better)", fontweight="bold")
    ax1.legend(fontsize=9)

    ax2 = axes[1]
    ax2.scatter(ac1s, ratios_hybrid, s=100, zorder=5, edgecolors="white", lw=1.2)
    for ac1, r, lbl in zip(ac1s, ratios_hybrid, labels_plot):
        ax2.annotate(lbl, (ac1, r), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax2.axhline(1.0, color="#2196F3", lw=2, ls="--")
    ax2.set_xlabel("Lag-1 Autocorrelation AC(1)")
    ax2.set_ylabel("Hybrid RMSE / ARIMA RMSE")
    ax2.set_title("AC(1) vs Model Performance", fontweight="bold")

    plt.suptitle("Regime Characterization Across All Loaded Datasets", fontweight="bold", fontsize=12)
    plt.tight_layout()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "fig_regime_scatter.pdf", bbox_inches="tight")
    plt.savefig(figures_dir / "fig_regime_scatter.png", bbox_inches="tight")
    plt.close()


def fig_real_residuals(results: dict, dataset_names: list[str], figures_dir: Path):
    models = ["ARIMA", "SARIMA", "XGBoost", "LSTM", "Hybrid"]
    fig, axes = plt.subplots(1, len(dataset_names), figsize=(4 * len(dataset_names), 5))
    if len(dataset_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, dataset_names):
        m_data = results.get(name, {}).get("metrics", {})
        stds = [m_data.get(m, {}).get("residual_std", 0) for m in models]
        colors = [COLORS.get(m, "#607D8B") for m in models]
        bars = ax.bar(models, stds, color=colors, edgecolor="white")
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("Residual Std Dev")
        ax.set_xticklabels(models, rotation=35, ha="right")
        for bar, v in zip(bars, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}", ha="center", fontsize=8)

    plt.suptitle("Residual Stability\n(Lower = more stable predictions)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "fig_real_residuals.pdf", bbox_inches="tight")
    plt.savefig(figures_dir / "fig_real_residuals.png", bbox_inches="tight")
    plt.close()


def fig_dm_summary(results: dict, dataset_names: list[str], figures_dir: Path):
    models = ["SARIMA", "Prophet", "XGBoost", "LSTM", "Hybrid"]
    fig, axes = plt.subplots(1, len(dataset_names), figsize=(4 * len(dataset_names), 4.5))
    if len(dataset_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, dataset_names):
        dm = results.get(name, {}).get("dm_tests", {})
        pvals = [dm.get(m, {}).get("p_value", 1.0) for m in models]
        colors = ["#4CAF50" if p < 0.05 else "#FF9800" for p in pvals]
        ax.bar(models, pvals, color=colors, edgecolor="white")
        ax.axhline(0.05, color="#F44336", lw=2, ls="--", label="α=0.05")
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("DM Test p-value")
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.legend(fontsize=8)

    plt.suptitle("Diebold-Mariano Significance Tests vs ARIMA Baseline", fontweight="bold", fontsize=13)
    plt.tight_layout()
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "fig_dm_summary.pdf", bbox_inches="tight")
    plt.savefig(figures_dir / "fig_dm_summary.png", bbox_inches="tight")
    plt.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_figures.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add shortseq/visualization/figures.py tests/test_figures.py
git commit -m "feat: port reproducible figure generation into shortseq.visualization"
```

---

### Task 13: Experiment orchestration scripts

**Files:**
- Move: `configs/hyperparams.yaml` → `experiments/configs/hyperparams.yaml`
- Create: `experiments/scripts/run_baselines.py`
- Create: `experiments/scripts/run_ablation.py`

- [ ] **Step 1: Move the hyperparameters config**

```bash
mkdir -p experiments/configs experiments/scripts
git mv configs/hyperparams.yaml experiments/configs/hyperparams.yaml
```

- [ ] **Step 2: Write `experiments/scripts/run_baselines.py`**

```python
"""CLI to run all baseline models against every implemented ShortSeq dataset.

Generalizes `code/real_experiment.py` and `code/m4_experiment.py`'s role
of "run the baseline set against a dataset and save results" across
every dataset in the registry (D-Mart, UCI, Walmart, M5, M4), using the
ported BaseForecaster models. The M4-specific naive/AR1/seasonal-naive
win-count micro-analysis from the original `m4_experiment.py` is
superseded here by running the full baseline set (including the new
SeasonalNaiveForecaster) uniformly, consistent with the benchmark's
uniform evaluation protocol.

Fixes a bug present in `code/experiment.py:run_dataset_experiment`: the
original DM test call passed each model's *residuals, reversed* as the
comparison series instead of its actual predictions. This version passes
each model's real predictions.
"""
import json
from pathlib import Path

import yaml

from shortseq.datasets.registry import load_all
from shortseq.evaluation.dm_test import diebold_mariano_test
from shortseq.evaluation.metrics import compute_metrics
from shortseq.models.arima import ARIMAForecaster
from shortseq.models.hybrid import HybridForecaster
from shortseq.models.lstm_model import LSTMForecaster
from shortseq.models.naive import SeasonalNaiveForecaster
from shortseq.models.prophet_model import ProphetForecaster
from shortseq.models.sarima import SARIMAForecaster
from shortseq.models.xgboost_model import XGBoostForecaster

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "hyperparams.yaml"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

SEASON_LENGTH_BY_FREQ = {"D": 7, "W": 52, "M": 12}


def build_models(config: dict, freq: str, season_length: int) -> dict:
    return {
        "ARIMA": ARIMAForecaster(max_p=config["arima"]["max_p"], max_q=config["arima"]["max_q"]),
        "SARIMA": SARIMAForecaster(
            max_p=config["sarima"]["max_p"], max_q=config["sarima"]["max_q"],
            max_P=config["sarima"]["max_P"], max_Q=config["sarima"]["max_Q"],
        ),
        "Prophet": ProphetForecaster(freq=freq),
        "XGBoost": XGBoostForecaster(
            lags=config["xgboost"]["lag_window"], n_estimators=config["xgboost"]["n_estimators"],
            max_depth=config["xgboost"]["max_depth"], learning_rate=config["xgboost"]["learning_rate"],
            subsample=config["xgboost"]["subsample"], colsample_bytree=config["xgboost"]["colsample_bytree"],
            random_state=config["xgboost"]["random_state"],
        ),
        "LSTM": LSTMForecaster(
            lags=config["lstm"]["lag_window"], units_layer1=config["lstm"]["units_layer1"],
            units_layer2=config["lstm"]["units_layer2"], dropout=config["lstm"]["dropout"],
            epochs=config["lstm"]["epochs"], batch_size=config["lstm"]["batch_size"],
            patience=config["lstm"]["patience"], validation_split=config["lstm"]["validation_split"],
            random_seed=config["lstm"]["random_seed"],
        ),
        "Hybrid": HybridForecaster(
            lags=config["hybrid"]["xgboost_residual"]["lag_window"],
            arima_max_p=config["hybrid"]["arima"]["max_p"], arima_max_q=config["hybrid"]["arima"]["max_q"],
            xgb_n_estimators=config["hybrid"]["xgboost_residual"]["n_estimators"],
            xgb_max_depth=config["hybrid"]["xgboost_residual"]["max_depth"],
            xgb_learning_rate=config["hybrid"]["xgboost_residual"]["learning_rate"],
            random_state=config["hybrid"]["xgboost_residual"]["random_state"],
        ),
        "Naive": SeasonalNaiveForecaster(season_length=season_length),
    }


def run_dataset(name: str, dataset, config: dict, split: float = 0.8) -> dict:
    series = dataset.series
    split_idx = int(len(series) * split)
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]

    season_length = SEASON_LENGTH_BY_FREQ.get(dataset.freq, 7)
    models = build_models(config, dataset.freq, season_length)

    results = {}
    preds_by_model = {}
    for model_name, model in models.items():
        try:
            model.fit(train)
            forecast = model.predict_rolling(test)
            preds = forecast.point[: len(test)]
            preds_by_model[model_name] = preds
            results[model_name] = compute_metrics(
                test.values, preds, model.name, model.train_time_, model.pred_time_
            )
            print(f"  {model_name}: RMSE={results[model_name]['rmse']:.2f}")
        except Exception as exc:
            print(f"  [{name}/{model_name}] failed: {exc}")

    dm_tests = {}
    if "ARIMA" in preds_by_model:
        arima_preds = preds_by_model["ARIMA"]
        for model_name, preds in preds_by_model.items():
            if model_name == "ARIMA":
                continue
            dm_stat, p_val = diebold_mariano_test(test.values, arima_preds, preds)
            dm_tests[model_name] = {"dm_stat": dm_stat, "p_value": p_val}

    output = {
        "dataset": name,
        "n_train": split_idx,
        "n_test": len(test),
        "metrics": {k: {mk: mv for mk, mv in v.items() if mk != "residuals"} for k, v in results.items()},
        "dm_tests": dm_tests,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}_results.json", "w") as f:
        json.dump(output, f, indent=2)
    return output


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    datasets = load_all()
    print(f"Running baselines on {len(datasets)} datasets...")
    for name, dataset in datasets.items():
        print(f"\n{'=' * 60}\n{name} | n={dataset.n} | freq={dataset.freq}\n{'=' * 60}")
        run_dataset(name, dataset, config)
    print(f"\nDone. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `experiments/scripts/run_ablation.py`**

```python
"""Training-window-size ablation on D-Mart Food, using the ported baselines.

Ported from `code/ablation.py`.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from shortseq.datasets.dmart import load_dmart
from shortseq.models.arima import ARIMAForecaster
from shortseq.models.hybrid import HybridForecaster
from shortseq.models.lstm_model import LSTMForecaster
from shortseq.models.xgboost_model import XGBoostForecaster

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"


def run_ablation(include_lstm: bool = False) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    series = load_dmart()["dmart_food"].series
    test_n = 37
    train_sizes = [50, 80, 110, 144]

    model_builders = {
        "ARIMA": ARIMAForecaster,
        "XGBoost": XGBoostForecaster,
        "Hybrid": HybridForecaster,
    }
    if include_lstm:
        model_builders["LSTM"] = LSTMForecaster

    results = {}
    print("=== Ablation: Training Window Size (D-Mart Food) ===")
    print(f"{'n_train':<10}" + "".join(f"{m:<12}" for m in model_builders))
    print("-" * (10 + 12 * len(model_builders)))

    for n_train in train_sizes:
        train = series.iloc[:n_train]
        test = series.iloc[n_train:n_train + test_n]
        row, row_str = {"n_train": n_train}, f"{n_train:<10}"
        for mname, builder in model_builders.items():
            try:
                model = builder().fit(train)
                preds = model.predict_rolling(test).point
                rmse = float(np.sqrt(mean_squared_error(test.values, preds[:len(test)])))
            except Exception as exc:
                print(f"  [{mname} n={n_train}] Error: {exc}")
                rmse = float("nan")
            row[mname] = round(rmse, 4)
            row_str += f"{rmse:<12.2f}"
        results[n_train] = row
        print(row_str)

    out_path = RESULTS_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm", action="store_true")
    args = parser.parse_args()
    run_ablation(include_lstm=args.lstm)
```

- [ ] **Step 4: Commit**

```bash
git add experiments/configs experiments/scripts
git commit -m "feat: add run_baselines.py and run_ablation.py experiment scripts"
```

---

### Task 14: Repo reorganization and retiring `code/`

**Files:**
- Move: `results/` → `experiments/results/`
- Move: `paper.pdf` → `paper/paper.pdf`
- Move: `figures/` → `paper/figures/`
- Delete: `code/`
- Delete: (now-empty) `configs/`

- [ ] **Step 1: Move results, paper, and figures**

```bash
mkdir -p paper
git mv results experiments/results 2>/dev/null || true
# if experiments/results already has content from Task 13 runs, merge manually instead of git mv
git mv paper.pdf paper/paper.pdf
git mv figures paper/figures
```

If `experiments/results` already exists with files (from running `run_baselines.py`/`run_ablation.py` in earlier tasks), move `results/`'s contents into it individually instead of a directory-level `git mv`, then remove the now-empty `results/` directory.

- [ ] **Step 2: Confirm the migration mapping table from the design spec is fully covered**

Cross-check against `docs/superpowers/specs/2026-07-23-shortseq-scaffold-design.md`'s "Migration mapping" table — every row should now have a corresponding `shortseq/` or `experiments/` file. The `generate_primary_data`/`generate_walmart_data`/`generate_m5_data` synthetic generators are the one deliberate drop (per the design spec) since real/calibrated CSVs already exist in `data/`.

- [ ] **Step 3: Delete `code/` and the now-empty `configs/`**

```bash
git rm -r code
rmdir configs 2>/dev/null || true
```

This is safe: the exact pre-migration state is recoverable at any time via `git checkout tmlr-submission`.

- [ ] **Step 4: Update `.gitignore` if needed**

Ensure `.venv/`, `__pycache__/`, `*.pyc`, and `shortseq.egg-info/` are ignored. Read the current `.gitignore` and add any missing lines.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: retire code/ scripts, reorganize results/paper/figures under experiments/ and paper/"
```

---

### Task 15: End-to-end verification, README, and final review

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/ -v`
Expected: all 51 tests pass (6 base + 9 datasets + 6 metrics + 2 classical models + 3 ML models + 2 hybrid/naive + 1 Prophet + 16 foundation stubs + 2 analysis stubs + 4 figures, from Tasks 3-12).

- [ ] **Step 2: Run the real baseline experiment end-to-end**

Run: `C:/venvs/shortseq/Scripts/python.exe experiments/scripts/run_baselines.py`
Expected: completes without crashing, prints RMSE per model per dataset, writes JSON files into `experiments/results/`.

- [ ] **Step 3: Sanity-check the D-Mart numbers against `paper/paper.pdf`**

Read `experiments/results/dmart_food_results.json` and compare its ARIMA RMSE to the paper's reported Food RMSE (255.76). Expect the same ballpark (not necessarily bit-exact — the refactor changes no algorithm, but pmdarima/XGBoost/TensorFlow have some run-to-run nondeterminism even with fixed seeds). A large divergence (e.g. 2x+ different) means something was mis-ported — stop and investigate rather than continuing.

- [ ] **Step 4: Rewrite `README.md`**

Update to describe the `shortseq` package: installation (`pip install -e .`), quickstart (`python experiments/scripts/run_baselines.py`), package layout, current implementation status (6 baselines + Naive done; 8 foundation models stubbed; new datasets/metrics/ranking/boundary analysis stubbed), and a pointer to `docs/superpowers/specs/2026-07-23-shortseq-scaffold-design.md` for the full design rationale. Keep the existing citation/license sections.

- [ ] **Step 5: Review the full diff against the TMLR submission**

```bash
git diff tmlr-submission --stat
```

Read through the output with the user before considering this plan done — nothing should be silently lost; everything removed should be accounted for in the migration mapping table.

- [ ] **Step 6: Final commit**

```bash
git add README.md
git commit -m "docs: rewrite README for the shortseq package scaffold"
```

---

## What this plan deliberately does NOT do

(Restating the design spec's non-goals so nobody "completes" them by accident while executing this plan.) No foundation model is installed or run. No new dataset source (Favorita, Rossmann, real Walmart, Corporación Favorita, StoreSales, M4 Weekly, M3 Monthly, Tourism) is downloaded. No CRPS/coverage/PIT calibration metric is implemented. No Friedman/Nemenyi ranking or boundary-map logistic regression is implemented. No Docker image, hosted leaderboard, or docs site is built. No multi-week experiment campaign from the execution plan is run. Each of those is a separate future phase, each deserving its own brainstorm → plan → build cycle.

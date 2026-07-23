# ShortSeq Package Scaffold — Design

Date: 2026-07-23
Status: Approved by user, pending implementation

## Context

The repo `retail-forecasting-benchmark` currently holds the TMLR submission
("Regime-Dependent Performance of ARIMA and Modern Forecasting Methods"):
flat scripts in `code/` (ARIMA, SARIMA, Prophet, XGBoost, LSTM, Hybrid,
metrics, DM test, figure generation), CSV data already downloaded for
D-Mart, M4 Micro Monthly, UCI Online Retail, Walmart-calibrated, and
M5-calibrated series, and hyperparameters in `configs/hyperparams.yaml`.

This is the base for a much larger follow-on: a NeurIPS 2027 Datasets &
Benchmarks submission ("ShortSeq") that adds 8 foundation models (Chronos,
TimesFM, Moirai, Moment, Timer, TTM, Lag-Llama, ForecastPFN), 7 more
dataset sources, new evaluation machinery (CRPS, coverage/calibration,
Friedman/Nemenyi ranking, boundary analysis), and ships as a
pip-installable benchmark package with a hosted leaderboard.

That full campaign is a multi-week effort requiring GPU access, new
dataset downloads (Kaggle/UCI/etc.), and real experimentation — none of
which belongs in a single scaffolding session. This spec covers **only**
the structural migration: turn the flat TMLR scripts into a proper
`shortseq/` package with the right interfaces, so later work (foundation
model wrappers, new datasets, new metrics) has a clean place to land
without fabricating results or code that hasn't been run.

## Goals

1. Preserve the exact TMLR-submitted state, recoverable via git.
2. Restructure the working baseline code (6 models, metrics, DM test,
   figures) into an importable, testable `shortseq/` package.
3. Define the abstractions (`BaseForecaster`, `SeriesDataset`) that
   foundation models and new datasets will plug into later.
4. Add real stub interfaces for not-yet-implemented pieces (foundation
   models, new metrics, boundary analysis, ranking) so the shape is
   correct, without faking behavior or results.
5. Make the package `pip install -e .`-able.
6. Prove the migration didn't break anything: smoke tests run the
   migrated models against the real existing data and match (or
   reasonably approximate) the numbers already published in `paper.pdf`.

## Non-goals (explicitly out of scope this session)

- Installing, running, or downloading any of the 8 foundation models.
- Downloading the 7 new dataset sources (Favorita, Rossmann, real
  Walmart, Corporación Favorita, StoreSales, M4 Weekly, M3 Monthly,
  Tourism).
- Implementing CRPS, coverage/calibration, or PIT-histogram metrics
  (these require probabilistic forecasts, which none of the current
  baseline models produce).
- Implementing Friedman/Nemenyi ranking or the boundary-map logistic
  regression (need many more model/dataset combinations to be
  meaningful — implementing now would mean untested, unvalidated code).
- Docker image, hosted leaderboard, docs site, GitHub Pages.
- Running the actual multi-week experiment campaigns from the execution
  plan.

## Safety step

Before any restructuring: `git tag tmlr-submission` on the current HEAD
(`8d539c5`, "Add final TMLR submission PDF"). This guarantees the exact
submitted state is recoverable by tag regardless of what happens to
`code/` afterward.

## Package layout

```
retail-forecasting-benchmark/
├── shortseq/                        # pip-installable package
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base.py                  # SeriesDataset dataclass: series (pd.Series/DataFrame),
│   │   │                            # name, freq, and regime stats (n, cv, ac1, zero_frac)
│   │   ├── dmart.py                 # loads data/real_{food,electronics,clothing,furniture}.csv
│   │   ├── uci.py                   # loads data/uci_*.csv
│   │   ├── m4.py                    # loads data/m4_monthly_train.csv + m4_sample_ids.csv
│   │   ├── walmart.py               # loads data/walmart.csv
│   │   ├── m5.py                    # loads data/m5.csv
│   │   └── registry.py              # load_all(), load_regime(cv_range, ac1_range) dispatch;
│   │                                # unimplemented sources raise NotImplementedError with a
│   │                                # docstring pointing at the execution plan
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseForecaster ABC: fit(train) -> self, predict(h) -> Forecast
│   │   │                            # Forecast dataclass: point (np.ndarray), dist (optional, for
│   │   │                            # future probabilistic models)
│   │   ├── arima.py                 # ported from code/experiment.py:run_arima
│   │   ├── sarima.py                # ported (seasonal branch of run_arima)
│   │   ├── prophet_model.py         # ported from run_prophet
│   │   ├── xgboost_model.py         # ported from run_xgboost + create_features_xgb
│   │   ├── lstm_model.py            # ported from run_lstm
│   │   ├── hybrid.py                # ported from run_hybrid
│   │   ├── naive.py                 # new: seasonal-naive reference model (named in the
│   │   │                            # execution plan's baseline table, not in old code)
│   │   └── foundation/              # stub package — one file per model, each a
│   │       ├── __init__.py          # BaseForecaster subclass whose fit/predict raise
│   │       ├── chronos.py           # NotImplementedError("not yet implemented — see
│   │       ├── timesfm.py           # execution plan Week 1"), so registry/tests can already
│   │       ├── moirai.py            # reference them by name.
│   │       ├── moment.py
│   │       ├── timer.py
│   │       ├── ttm.py
│   │       ├── lag_llama.py
│   │       └── forecastpfn.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               # ported compute_metrics (RMSE/MAE/MAPE/sMAPE); stub
│   │   │                            # functions for crps/coverage_80/coverage_95 that raise
│   │   │                            # NotImplementedError
│   │   ├── dm_test.py               # ported diebold_mariano_test + a bonferroni_correct()
│   │   │                            # helper (new, simple — mechanical correction only)
│   │   └── ranking.py               # stub: friedman_nemenyi() raises NotImplementedError
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── boundary.py              # stub: fit_boundary() raises NotImplementedError
│   └── visualization/
│       ├── __init__.py
│       └── figures.py               # ported fig_* functions from code/figures_v2.py and
│                                     # code/figures_new_datasets.py, parameterized on a
│                                     # results dict instead of reading fixed file paths
├── experiments/
│   ├── configs/
│   │   └── hyperparams.yaml         # moved from configs/, unchanged
│   ├── scripts/
│   │   └── run_baselines.py         # thin CLI: loads datasets via shortseq.datasets,
│   │                                # runs shortseq.models baselines, writes results —
│   │                                # replaces real_experiment.py/m4_experiment.py/ablation.py
│   └── results/                     # moved from results/, unchanged content
├── data/                            # unchanged — existing CSVs stay put
├── paper/
│   ├── paper.pdf                    # moved from top level
│   └── figures/                     # moved from figures/
├── tests/
│   ├── test_metrics.py              # smoke test: metrics.py + dm_test.py against known inputs
│   ├── test_models.py               # smoke test: each baseline model fits/predicts on a
│   │                                # real D-Mart series without erroring
│   └── test_datasets.py             # smoke test: each dataset loader returns a SeriesDataset
│                                     # with the stats reported in paper.pdf (n, CV, AC1)
├── pyproject.toml                   # new — PEP 621 + setuptools, makes `pip install -e .` work
├── requirements.txt                 # kept; split dev/test deps into requirements-dev.txt
├── README.md                        # rewritten to describe shortseq + migration status
└── docs/superpowers/specs/          # this file
```

`code/` is retired (deleted) once its logic is fully ported and the smoke
tests pass — recoverable via the `tmlr-submission` git tag.

## Key interfaces

```python
# shortseq/models/base.py
@dataclass
class Forecast:
    point: np.ndarray
    dist: object | None = None   # reserved for foundation models' predictive distributions

class BaseForecaster(ABC):
    @abstractmethod
    def fit(self, train: pd.Series) -> "BaseForecaster": ...
    @abstractmethod
    def predict(self, h: int) -> Forecast: ...

# shortseq/datasets/base.py
@dataclass
class SeriesDataset:
    name: str
    series: pd.Series          # datetime-indexed
    freq: str                  # 'D', 'W', 'M'
    n: int
    cv: float
    ac1: float
    zero_frac: float
    real: bool                 # True for real data, False for calibrated/synthetic
```

These two contracts are the ones every future foundation model wrapper
and every future dataset loader must satisfy — designed now so later
work (out of scope here) doesn't require reshaping the core package.

## Migration mapping (old → new)

| Old (`code/`) | New (`shortseq/`) |
|---|---|
| `experiment.py:compute_metrics` | `evaluation/metrics.py` |
| `experiment.py:diebold_mariano_test` | `evaluation/dm_test.py` |
| `experiment.py:run_arima` | `models/arima.py`, `models/sarima.py` |
| `experiment.py:run_prophet` | `models/prophet_model.py` |
| `experiment.py:create_features_xgb`, `run_xgboost` | `models/xgboost_model.py` |
| `experiment.py:run_lstm` | `models/lstm_model.py` |
| `experiment.py:run_hybrid` | `models/hybrid.py` |
| `real_experiment.py` | `experiments/scripts/run_baselines.py` (D-Mart path) |
| `m4_experiment.py` | `experiments/scripts/run_baselines.py` (M4 path) |
| `ablation.py` | `experiments/scripts/run_ablation.py` (kept separate, ported as-is) |
| `figures_v2.py`, `figures_new_datasets.py` | `visualization/figures.py` |
| `generate_primary_data`, `generate_walmart_data`, `generate_m5_data` (synthetic generators used only when real data absent) | dropped — the repo already has real/calibrated CSVs in `data/`; loaders read those directly instead of regenerating synthetic stand-ins |

## Verification plan

- `pytest tests/` passes.
- `experiments/scripts/run_baselines.py` run on D-Mart data reproduces
  RMSE figures in the same ballpark as Table in `paper.pdf` (ARIMA
  ≈255.76 Food, ≈227.20 Electronics, etc.) — not bit-exact (refactor
  changes nothing algorithmic, but float/library nondeterminism is
  expected), but no gross divergence.
- `pip install -e .` succeeds in a clean virtualenv and `import shortseq`
  works.
- `git diff tmlr-submission --stat` reviewed by user before `code/` is
  deleted, so nothing is lost silently.

## Open questions for implementation plan

None — scope is fully bounded by the goals/non-goals above. The
writing-plans step should sequence: (1) tag + pyproject scaffold, (2)
datasets migration, (3) models migration, (4) evaluation migration, (5)
figures migration, (6) stub interfaces for foundation models / new
metrics, (7) tests, (8) retire `code/`, (9) README rewrite.

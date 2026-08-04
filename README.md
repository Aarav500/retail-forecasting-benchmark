# Regime-Dependent Performance of ARIMA and Modern Forecasting Methods
### An Empirical Benchmark on Small-Scale Retail Demand Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

**Aarav Shah** · University of California, Riverside · `ashah264@ucr.edu`  
GitHub: [github.com/Aarav500/retail-forecasting-benchmark](https://github.com/Aarav500/retail-forecasting-benchmark)

---

This repo hosts the paper above **and** its in-progress successor,
**ShortSeq** — a pip-installable benchmark package for evaluating foundation
models against classical baselines on short retail time series. The TMLR
results (6 models, 5 dataset sources) are preserved exactly and are now
served by the `shortseq` package described below; ShortSeq's follow-on work
is partly under way — the additional datasets are loaded (368 series in
total) and Chronos is implemented and swept across all five of its model
sizes, while the other 7 foundation models and the probabilistic metrics
are still scaffolding. See [Project status](#project-status) for exactly what's
implemented vs. stubbed, and
[`docs/superpowers/specs/2026-07-23-shortseq-scaffold-design.md`](docs/superpowers/specs/2026-07-23-shortseq-scaffold-design.md)
for the full design rationale behind the package layout.

---

## Windows setup note: put your venv outside the repo

On Windows, create the virtualenv **outside this repo**, at a short path such as
`C:\venvs\shortseq` — not an in-repo `.venv`. TensorFlow's and JupyterLab's own
package internals contain deeply-nested paths that, combined with a long project
path (especially under a synced folder like OneDrive), exceed Windows' 260-character
path limit and break installation (`OSError: [Errno 2] No such file or directory`).
The alternative fix, enabling Windows Long Path support, is a system-settings change
this project has opted not to require.

```bash
"C:/Users/you/AppData/Local/Programs/Python/Python311/python.exe" -m venv "C:/venvs/shortseq"
C:/venvs/shortseq/Scripts/python.exe -m pip install --upgrade pip
C:/venvs/shortseq/Scripts/python.exe -m pip install -e .
```

`pip install -e .` (installing the `shortseq` package itself, editable) pulls in
its runtime dependencies from `pyproject.toml` and is the recommended install
path now that the code lives in an importable package rather than flat
scripts. `requirements.txt` is kept as an alternative/pinned-versions option
and for the `jupyter`/`ipykernel` extras it includes that aren't package
dependencies.

Linux/macOS contributors aren't affected and can use a normal in-repo `.venv` as usual.

---

## Summary

*This section describes the TMLR submission's scope. For what the repo
covers today — 368 series, plus the Chronos campaign — see
[Project status](#project-status).*

We benchmark six forecasting methods across **5 dataset sources, 3 countries, 35 time series**:

| Dataset | Source | Country | n | Real? |
|:---|:---|:---|---:|:---|
| D-Mart (4 category series) | Single retail store | India | 181/series | ✓ Real |
| UCI Online Retail (5 series) | E-commerce retailer | UK | 53/series | ✓ Real |
| M4 Micro Monthly (24 series) | M4 competition | International | 68–197 | ✓ Real |
| Walmart (1 series) | Calibrated to Kaggle data | US | 143 | Calibrated |
| M5 Intermittent (1 series) | Calibrated to M5 stats | — | 365 | Calibrated |

**Central finding**: Model selection must be regime-aware.

| Regime | CV | AC(1) | Best model |
|:---|:---|:---|:---|
| Low-SNR (D-Mart daily) | ~0.02 | <0.21 | **ARIMA** |
| High-variance (UCI, Walmart) | 0.32–0.59 | >0.40 | **Hybrid (ARIMA+XGBoost)** |
| High-autocorrelation (M4 monthly) | any | >0.60 | **Structured model** |
| Intermittent (M5) | N/A | ~0 | Specialized methods |

---

## Key Results

### D-Mart (India) — ARIMA wins all 4 categories (CV≈0.021)

| Model | Food | Electronics | Clothing | Furniture |
|:---|---:|---:|---:|---:|
| **ARIMA** | **255.76** | **227.20** | 238.78 | **219.06** |
| LSTM | 262.82 | 230.06 | **235.72** | 219.88 |
| Hybrid | 312.37 | 240.65 | 274.02 | 262.81 |
| Prophet | 530.09 | 278.02 | 723.46 | 287.82 |

### UCI Online Retail (UK) — Hybrid wins 4/5 at CV 0.32–0.59

| Series | CV | ARIMA | Hybrid | Winner |
|:---|---:|---:|---:|:---|
| Jumbo Bag | 0.59 | 7,153 | 8,021 | XGBoost |
| Lunch Bag | 0.49 | 1,367 | 1,770 | **ARIMA** |
| Red Retrospot | 0.32 | 953 | **855** | Hybrid |
| Regency Cakestand | 0.58 | 1,023 | **973** | Hybrid |
| Store Total | 0.45 | 97,295 | **86,038** | Hybrid |

### M4 Micro Monthly — AR(1) beats naive in 21/24 series

Mean AR(1)/Naive ratio: **0.623** (37.7% RMSE reduction). Consistent across all CV bins when AC(1)≈0.88. Confirms AC(1) is the primary predictor of structured model advantage.

### Walmart weekly — Hybrid wins by 49.6%

ARIMA: 165.23 → Hybrid: **83.15**. All DM tests: p<0.001.

*(These are the TMLR-submitted numbers, from `paper/paper.pdf` Table II and
its companion tables. Re-running `experiments/scripts/run_baselines.py` in a
different environment reproduces ARIMA/SARIMA/XGBoost/Hybrid to the same
values — those models are deterministic — while LSTM will vary slightly
run-to-run, see [Reproducibility](#reproducibility).)*

---

## Quickstart

Windows users: see the "Windows setup note" above first — create your venv outside this repo (e.g. `C:\venvs\shortseq`) before running `pip install` below.

```bash
git clone https://github.com/Aarav500/retail-forecasting-benchmark
cd retail-forecasting-benchmark
pip install -e .

# Smoke-test on a single dataset first (finishes in under ~2 minutes on CPU;
# LSTM's default 100-epoch training dominates the runtime even for one dataset)
python experiments/scripts/run_baselines.py --datasets dmart_food

# Run the full baseline sweep: all 7 baselines x every currently-loaded
# series (368 of them: D-Mart x4, UCI x5, calibrated Walmart, M5, M4
# Micro x24, M4 Weekly x50, M3 Monthly x100, Rossmann x50, real Walmart
# x50, Favorita x83). The 35-series TMLR subset alone took 30-90+ minutes
# on CPU; SARIMA's seasonal search and LSTM's rolling forecast both scale
# badly past n~1000, so the whole set is a multi-day job — scope it with
# --datasets unless you mean it.
python experiments/scripts/run_baselines.py

# Run the training-window-size ablation (D-Mart Food)
python experiments/scripts/run_ablation.py
python experiments/scripts/run_ablation.py --lstm   # include the LSTM arm

# Rank the five Chronos sizes within each length stratum
# (Friedman omnibus + Nemenyi post-hoc). CPU-only, runs NO models, and
# reads the already-committed size-scaling results — it finishes in
# seconds and needs no campaign re-run, no GPU, and no model downloads.
python experiments/scripts/run_ranking.py
# Expected output: two strata, SHORT (n<200) 199 series x 5 sizes and
# LONG (n>=200) 169 series x 5 sizes, with the omnibus Friedman test
# significant in both. Project status links to the write-up of what
# that result actually means.

# Generate figures (see shortseq/visualization/figures.py for the fig_*
# functions; each takes a results dict rather than reading fixed paths)
```

`run_ranking.py` is deterministic apart from its `generated_at` stamp, so
re-running it leaves the two files under `experiments/results/ranking/`
showing as modified with only that one line changed. That is expected —
it confirms the reproduction rather than casting doubt on it. The safe
response is to leave the timestamp change alone or `git stash` it;
reflexively `git checkout`-ing result files to tidy the worktree is a
habit worth not forming here, where those files are the scientific record
and line-ending normalisation can rewrite far more than the one line you
meant to discard.

Results land as one JSON per dataset in `experiments/results/`, e.g.
`experiments/results/dmart_food_results.json` (RMSE/MAE/MAPE per model,
Bonferroni-corrected DM tests vs. ARIMA, and any per-model failures).

Per-timestep residuals are **stripped by default** (they inflate result
files roughly 7-10x, and results are git-tracked). Post-hoc Diebold-Mariano
testing needs them, so pass `--keep-residuals` if a run is meant to support
DM analysis. It is accepted by the four sweeps that write a per-model
`metrics` block — `run_baselines.py`, `run_chronos.py`,
`run_size_scaling.py` and `run_arima_only.py`. The committed results,
including the 1,840-file Chronos scaling sweep, were produced *without*
it, so DM tests on that corpus require re-running the sweep rather than
just re-reading the files.

Run the test suite with:

```bash
pytest tests/
```

---

## Repository Structure

```
retail-forecasting-benchmark/
├── shortseq/                        # the pip-installable package
│   ├── constants.py                 # CHRONOS_SIZES — the one canonical size list
│   ├── results_io.py                # strict-JSON result writer (write_result_json)
│   ├── datasets/                    # SeriesDataset loaders + registry (dmart, uci,
│   │                                 # walmart, walmart_real, m5, m4, m4w, m3m,
│   │                                 # rossmann, favorita;
│   │                                 # load_all()/load_regime()/load_source())
│   ├── models/                      # BaseForecaster implementations
│   │   └── foundation/              # Chronos (implemented) + 7 stubs
│   ├── evaluation/                  # metrics, DM test, Friedman/Nemenyi ranking
│   ├── analysis/                    # boundary analysis (not yet implemented)
│   └── visualization/               # reproducible figure generation
├── experiments/
│   ├── configs/hyperparams.yaml     # all model hyperparameters
│   ├── scripts/                     # prepare_datasets.py, run_baselines.py,
│   │                                 # run_ablation.py, run_arima_only.py,
│   │                                 # run_chronos.py, run_size_scaling.py,
│   │                                 # patch_scaling_baselines.py, run_ranking.py
│   └── results/                     # JSON results (TMLR + any local reruns)
│       ├── foundation/              # per-dataset Chronos vs. ARIMA runs
│       ├── scaling/                 # the Chronos size-scaling campaign (1,840 files)
│       └── ranking/                 # Friedman/Nemenyi output, one file per stratum
├── data/                            # D-Mart / UCI / M4 Micro / Walmart / M5 CSVs
│   └── derived/                     # series extracted by prepare_datasets.py
│                                     # (Rossmann, real Walmart, Favorita,
│                                     # M3 Monthly, M4 Weekly); the raw Kaggle
│                                     # dumps they come from are gitignored
├── paper/
│   ├── paper.pdf                    # the TMLR submission
│   └── figures/                     # the paper's 12 figures (PNG + PDF)
├── tests/                           # pytest suite for shortseq/ and experiments/scripts
├── docs/superpowers/                # design specs + implementation plans, incl. the
│                                     # ranking spec and its recorded Outcome
├── notebooks/                       # EDA and result exploration
├── pyproject.toml                   # PEP 621 + setuptools — `pip install -e .`
├── requirements.txt                 # pinned deps (alternative to pyproject.toml install)
└── requirements-gpu.txt             # torch + chronos-forecasting, for the GPU sweep only
```

---

## Project status

This repo is mid-migration from the flat TMLR scripts (`code/`, now retired
and recoverable via the `tmlr-submission` git tag) into `shortseq`, the base
package for a larger follow-on NeurIPS Datasets & Benchmarks submission. See
[`docs/superpowers/specs/2026-07-23-shortseq-scaffold-design.md`](docs/superpowers/specs/2026-07-23-shortseq-scaffold-design.md)
for the full rationale, interface contracts (`BaseForecaster`, `SeriesDataset`),
and the old-code-to-new-package migration mapping.

**Implemented and verified against real data (7 baselines):**
ARIMA, SARIMA, XGBoost, LSTM, Hybrid (ARIMA+XGBoost residual correction),
Prophet, and SeasonalNaive — all in `shortseq/models/`, run end-to-end by
`experiments/scripts/run_baselines.py`, which sweeps whatever `load_all()`
returns (currently 368 series).

*Committed* coverage is uneven, which is worth knowing before reading
`experiments/results/`: the original 35 TMLR series carry all seven
baselines; the 333 series added since carry **ARIMA only**. SARIMA's
seasonal search and LSTM's rolling forecast scale badly past n≈1000
(Favorita n=1684, Rossmann n=942), so `experiments/scripts/run_arima_only.py`
backfilled just the ARIMA reference the Chronos campaign needs. It writes
the same schema, so a later job can merge the remaining baselines in
rather than replacing those files.

**Implemented — foundation models (1 of 8):**
Chronos (`shortseq/models/foundation/chronos.py`), a real wrapper over the
`chronos-forecasting` package — not a stub. Run per-dataset by
`experiments/scripts/run_chronos.py` and swept across all five model sizes
by `experiments/scripts/run_size_scaling.py`. The committed campaign is
**1,840 runs** (368 series x 5 sizes, zero failures) under
`experiments/results/scaling/`. The sweep itself needs a GPU; everything
downstream of it does not.

**Implemented — evaluation and figures:**
RMSE/MAE/MAPE metrics, the Diebold-Mariano test with a Bonferroni-correction
helper, and Friedman/Nemenyi statistical ranking
(`shortseq/evaluation/ranking.py`, driven by
`experiments/scripts/run_ranking.py`) — all in `shortseq/evaluation/` —
plus reproducible figure generation (`shortseq/visualization/figures.py`).

The ranking has been **run**, not merely implemented. The five Chronos
sizes are ranked within each length stratum in
`experiments/results/ranking/chronos_sizes_short.json` and
`chronos_sizes_long.json`: the omnibus Friedman test is significant in
both strata, the pre-registered prediction about *adjacent* size pairs was
**not** borne out (5 of 8 adjacent pairs separate), and the long-stratum
`base` anomaly turns out to be noise that within-series ranking actually
reverses. That outcome is written up in full — beneath the untouched
prediction it is being scored against — in
[the Outcome section of the ranking design spec](docs/superpowers/specs/2026-07-29-friedman-nemenyi-ranking-design.md#outcome-recorded-2026-07-29-after-the-analysis).

**Stubbed (interfaces defined, raise `NotImplementedError`, real work not started):**
- 7 foundation models — TimesFM, Moirai, Moment, Timer, TTM,
  Lag-Llama, ForecastPFN (`shortseq/models/foundation/`)
- 1 additional dataset source — Tourism (named in
  `shortseq/datasets/registry.py`, not yet acquired; it needs R and
  `tsibbledata` to extract)
- Probabilistic metrics — CRPS, prediction-interval coverage
  (`shortseq/evaluation/metrics.py`) — meaningless until one of the
  models above produces a predictive distribution
- Boundary-map logistic regression (`shortseq/analysis/boundary.py`) —
  needs foundation-model results across many (n, AC(1)) combinations, and
  is deferred for a second reason the ranking spec records: its premise,
  that there is a region where no foundation model wins, is undermined by
  Chronos-large already winning 71% of short series

None of the stubbed pieces fabricate results — they raise clearly, with a
docstring pointing at what has to happen first.

---

## Reproducibility

- Fixed random seeds: `numpy.random.seed(42)`, `tf.random.set_seed(42)`
- Strict temporal 80/20 split — no look-ahead
- Practitioner-default hyperparameters throughout (`experiments/configs/hyperparams.yaml`)
- **CPU-only for the classical baselines, the ablation, and every analysis
  step** — including `run_ranking.py`, which reads committed results and
  runs no models. The Chronos work is the exception: `run_chronos.py` and
  `run_size_scaling.py` load pretrained transformer checkpoints and were
  run on a GPU instance against `requirements-gpu.txt`. Re-running *that*
  sweep needs a GPU; re-running anything downstream of its committed
  results does not.
- All DM tests: two-sided, Newey-West variance, squared-error loss, p<0.001
- ARIMA, SARIMA, XGBoost, and Hybrid are deterministic given the seed
  **within a fixed dependency environment**, but are not guaranteed
  bit-for-bit across environments — e.g. an XGBoost major-version change
  (2.0→2.x) can shift floating-point summation order in tree construction
  even with a fixed `random_state`. Observed drift when re-running this
  migration's own verification against newer dependency versions than
  produced the original TMLR numbers: XGBoost ~1–2%. **LSTM is not
  deterministic even within one environment**: TensorFlow's CPU kernels
  are not run-to-run deterministic even with a fixed seed, so expect
  variation (occasionally larger than "small" — see Known limitations
  below) in LSTM's numbers between runs/machines — this is inherited from
  the original implementation, not introduced by the `shortseq` migration.

## Known limitations

- **Prophet diverges severely on short, strongly-trending series** — most
  visibly on all 5 UCI Online Retail series (RMSE 20–94x worse than
  ARIMA; e.g. Store Total RMSE ≈1.95M vs. ARIMA's ≈97K). Root cause:
  Prophet's default unconstrained linear-trend extrapolation runs away
  on a 53-week series with a real trend and no saturation point — the
  same failure mode `paper/paper.pdf` already documents for Prophet on
  Walmart, just more severe here. This is a genuine result of the
  benchmark's practitioner-default-hyperparameters methodology, not a
  bug — tuning Prophet's changepoint/growth settings to avoid it would
  contradict that methodology, so the numbers are left as-is and
  reported honestly rather than excluded or hidden.
- **LSTM's Walmart number diverges more than its usual run-to-run
  noise** — this migration's verification run measured RMSE 411.59
  vs. the paper's published 907.65 (a ~2.2x shift), larger than the
  "small variation" LSTM ordinarily shows between runs. Most likely
  cause: this environment's TensorFlow (2.21) is several major versions
  newer than whatever produced the paper's original numbers, and
  `requirements.txt`/`pyproject.toml` pin no upper bound. Flagged here
  rather than silently accepted as ordinary noise.

---

## Citation

```bibtex
@article{shah2025forecasting,
  title={Regime-Dependent Performance of {ARIMA} and Modern Forecasting Methods:
         An Empirical Benchmark on Small-Scale Retail Demand Data},
  author={Shah, Aarav},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://github.com/Aarav500/retail-forecasting-benchmark}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

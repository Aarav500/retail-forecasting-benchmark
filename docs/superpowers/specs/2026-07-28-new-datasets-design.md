# Phase B: New Dataset Acquisition — Design

Date: 2026-07-28
Status: Approved by user, pending implementation

## Context

`shortseq` currently loads 35 real/calibrated series from 5 sources
(D-Mart, UCI, Walmart-calibrated, M5-calibrated, M4 Micro Monthly). The
NeurIPS execution plan targets 500+ real series across many more
sources. This is Phase B of the four-phase roadmap
(`docs/superpowers/plans/`-adjacent roadmap artifact): pure data
acquisition, no GPU, independent of Phase A (Chronos, now merged).

## Goals

Add 5 new real dataset sources, taking the registry from 35 series to
several hundred, using the existing `SeriesDataset`/`make_dataset`
contract and the existing loader/test patterns.

## Sources in scope

| Source | Origin | Country | Freq | Target series | Registry key(s) |
|---|---|---|---|---|---|
| Rossmann | Kaggle (local) | Germany | Daily | 50 sampled stores | `rossmann_store_<id>` |
| Walmart (real) | Kaggle (local) | US | Weekly | 50 sampled store-dept | `walmart_real_<store>_<dept>` |
| Favorita | Kaggle (local) | Ecuador | Daily | 2 aggregation views | `favorita_store_item_*`, `favorita_family_*` |
| M4 Weekly | `datasetsforecast` | International | Weekly | 50 sampled | `m4w_<id>` |
| M3 Monthly | `datasetsforecast` | International | Monthly | 100 sampled | `m3m_<id>` |

**Favorita overlap (investigated, confirmed):** the execution plan's
"Favorita", "Corporación Favorita", and "StoreSales" rows are three
aggregation views of the *same* Kaggle competition data
(`store-sales-time-series-forecasting`, itself a re-release of the
older `favorita-grocery-sales-forecasting`). Counting them as three
independent sources would inflate the benchmark's claimed diversity.
They are therefore implemented as **one source with two documented
aggregation views** (store-item level and product-family level), and
any "number of sources" claim in the paper must count Favorita once.

**Tourism deferred:** distributed via the R package `tsibbledata`;
needs R/rpy2 or a vetted CSV mirror. Out of scope here, stays in
`_PLANNED_SOURCES`.

## Data handling

Raw Kaggle files are already present locally under
`data/{rossmann-store-sales,store-sales-time-series-forecasting,walmart-recruiting-store-sales-forecasting}/`
but are large (121MB Favorita train.csv, 38MB Rossmann train.csv,
zipped Walmart files). Committing them would bloat the repo.

**Approach:** raw Kaggle directories are gitignored. Each loader reads
the raw file if present and derives its series; a one-time preparation
script writes small aggregated per-series CSVs into `data/derived/`,
which ARE committed (same pattern as the existing `real_food.csv`
etc.). Loaders read from `data/derived/`, so the committed repo is
reproducible without re-downloading anything, while the multi-hundred-MB
raw files stay local.

M4 Weekly / M3 Monthly download via `datasetsforecast` into a
gitignored cache; the same preparation script samples and writes their
derived CSVs.

## Sampling

The plan calls for sampled subsets (50 Rossmann stores, 50 Walmart
store-dept, 50 M4 Weekly, 100 M3 Monthly) rather than all series.
Sampling uses a fixed seed (42, matching the project's existing
convention) and is done once by the preparation script, with the
selected IDs recorded in the derived CSVs — so the sample is frozen and
reproducible, not re-drawn per run.

## Structure

- `experiments/scripts/prepare_datasets.py` — one-time raw → derived
  conversion (Kaggle files + datasetsforecast downloads → `data/derived/*.csv`).
- `shortseq/datasets/rossmann.py`, `walmart_real.py`, `favorita.py`,
  `m4_weekly.py`, `m3_monthly.py` — one loader per source, each
  returning `dict[str, SeriesDataset]` via `make_dataset`, reading only
  from `data/derived/`.
- `shortseq/datasets/registry.py` — new loaders added to `_LOADERS`;
  the 7 now-implemented names removed from `_PLANNED_SOURCES`
  (`tourism` remains).
- `tests/test_datasets_new.py` — per-source tests mirroring the
  existing `tests/test_datasets.py` pattern (series counts, `n` ranges,
  `real=True`, sane CV).

## Non-goals

- Tourism (R dependency).
- Re-running any model against the new data — that's Phase C.
- Changing the existing 5 loaders or 35 series.
- CRPS/ranking/boundary analysis (still stubs).

## Verification

- `prepare_datasets.py` runs end-to-end from the local raw files and
  produces the expected derived CSV count.
- New loader tests pass; existing 63-test suite still passes unchanged.
- `load_all()` returns 35 + new series, with no key collisions (note
  the existing calibrated `walmart` key vs. new `walmart_real_*`).

## Addendum (2026-07-28): series-length stratification

Task 1's M4 Weekly sample came back with median n=837, max n=2296 —
long series, in a benchmark whose thesis is the short-series regime
(n < 200). Decision (user, 2026-07-28): **keep the long series, but
require length-stratified reporting.**

Rationale: long series are the necessary contrast case — they are
precisely where foundation models are expected to win, so they make the
short-series finding meaningful rather than diluting it. Pooling them
into a single headline number, however, would obscure exactly the
effect this benchmark exists to measure.

Binding requirement for Phase C/D: any aggregate result (ranking,
boundary map, win/loss counts, mean RMSE ratios) MUST be reported
stratified by series length — at minimum short (n < 200) vs. long
(n >= 200) — never pooled across the full registry. The existing 35
series are almost entirely n <= 197, so an unstratified pooled number
would silently mix two regimes with opposite expected outcomes.

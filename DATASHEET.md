# Datasheet for ShortSeq

Following the framework of Gebru et al., *Datasheets for Datasets* (CACM, 2021).

**Artifact:** ShortSeq — a benchmark for evaluating forecasters on short time series
**Version:** 0.1.0
**Date:** 2026-07-29
**Author:** Aarav Shah (<ashah264@ucr.edu>)
**Repository:** https://github.com/Aarav500/retail-forecasting-benchmark

> **Read this first.** ShortSeq is a *benchmark assembled from third-party sources*, not a
> newly collected dataset. It contributes loaders, an evaluation protocol, and computed
> results. Every claim below that concerns an upstream provider is limited to what the
> repository itself records; where the repository records nothing, this datasheet says so
> rather than supplying an answer. Section 6.2 contains a **partially resolved licensing
> position** — two sources confirmed CC BY 4.0, one confirmed to have no licence at all, and
> three Kaggle competitions still unverified — which a prospective user must read before
> relying on the distributed data.

---

## 1. Motivation

**For what purpose was the benchmark created?**

To test whether large pretrained forecasting models retain their advantage when the
training history is short. Foundation models are typically evaluated on long series, where
sample size favours them. ShortSeq assembles series short enough that this advantage is not
assumed, and stratifies them so that short-series behaviour cannot be averaged away by long
ones.

It grew out of an earlier empirical study (Shah, *Regime-Dependent Performance of ARIMA and
Modern Forecasting Methods*, TMLR 2025), which covered 35 series. ShortSeq extends that to
368 and adds foundation-model coverage and multi-model statistical ranking.

**Who created it and who funded it?**

Created by Aarav Shah. **Funding is not recorded in the repository and is not stated here.**

---

## 2. Composition

### 2.1 What do the instances represent?

Each instance is a **univariate time series of retail or economic demand**, indexed by
timestamp, with a single non-negative-by-convention numeric value per period (three series
contain negatives; see 2.6). Series are the unit of analysis; there are no labels, no
features beyond the value itself, and no relationships between instances except as noted in
2.5.

### 2.2 How many instances are there?

**368 series**, split into two strata that are analysed separately and **never pooled**:

| stratum | rule | count |
|---|---|---|
| short | n < 200 | **199** |
| long | n ≥ 200 | **169** |

The boundary is `SHORT_MAX_N = 200` (`experiments/scripts/run_size_scaling.py:44`).

### 2.3 Composition by source

| registry key | series | n (min/max/median) | freq | short/long | real? |
|---|---|---|---|---|---|
| `dmart` | 4 | 181 / 181 / 181 | D | 4 / 0 | yes |
| `uci` | 5 | 53 / 53 / 53 | W-SUN | 5 / 0 | yes |
| `walmart` | 1 | 143 / 143 / 143 | W-SUN | 1 / 0 | **no — simulated** |
| `m5` | 1 | 365 / 365 / 365 | D | 0 / 1 | **no — simulated** |
| `m4` | 24 | 68 / 197 / 150 | M | 24 / 0 | yes |
| `m4_weekly` | 50 | 93 / 2296 / 785.5 | W-SUN | 15 / 35 | yes |
| `m3_monthly` | 100 | 69 / 144 / 133 | M | 100 / 0 | yes |
| `rossmann` | 50 | 942 / 942 / 942 | D | 0 / 50 | yes |
| `walmart_real` | 50 | 143 / 143 / 143 | **W-FRI** | 50 / 0 | yes |
| `favorita` | 83 | 1684 / 1684 / 1684 | D | 0 / 83 | yes |
| **total** | **368** | | | **199 / 169** | **366 real, 2 simulated** |

### 2.4 How many *sources* is that, honestly?

**Fewer than the ten registry keys suggest.** State this carefully in any paper:

- **Two keys are not data at all.** `walmart` and `m5` are `numpy`-generated
  (`real=False`), produced by seeded simulators in the retired `code/experiment.py`
  (`np.random.seed(123)` and `seed(999)` respectively, recoverable at tag
  `tmlr-submission`). They are calibrated to resemble the real Walmart and M5 data but
  contain none of it.
- **`m4` and `m4_weekly` are one competition.** Two registry keys over M4 (Micro Monthly
  from a committed R-export; Weekly via `datasetsforecast`).
- **`favorita` is one Kaggle dump held as two views**, deliberately merged under one key so
  the source count stays honest (`shortseq/datasets/registry.py:15-28`).

**The defensible statement is: 366 real series from at most 8 distinct upstream providers,
plus 2 simulated series.** Not "368 series from ten public sources."

### 2.5 Are relationships between instances made explicit?

**Yes, and one is critical.** The two Favorita views are **not independent observations** —
the family view is literally the store-item transactions summed over stores
(`shortseq/datasets/favorita.py:14-21`). Any statistic that treats all 368 series as
independent draws is, to that extent, double-counting Favorita.

The two views are also **not nested**: the family view sums all 54 stores, while the
store-item view samples from only 46 after dropping eight late-opening stores. So family
totals include stores the store-item view excludes.

### 2.6 Is any information missing, noisy, or surprising?

- **Gaps in the index are not filled.** Favorita omits four dates (2013–2016-12-25),
  deliberately and documented (`favorita.py:59-73`).
- **All five UCI series are missing the week of 2011-01-02** (53 observations where the
  weekly span implies 54). **This is undocumented in the source code and was found by
  computation during the preparation of this datasheet.** Its cause is unknown.
- **Three of the 50 `walmart_real` series contain negative values** (returns exceeding
  sales in a week). Kept deliberately (`walmart_real.py:32-34`).
- **Zero-inflation is substantial in places.** `favorita_family_books` is 82.96% zeros;
  Rossmann series run 16.8–19.1% (closed days); the simulated `m5` is 32.9%.
- **Two loader docstrings overstate length variation.** `rossmann.py:38` says "n=941-942"
  (all 50 are 942); `walmart_real.py:45` says "n=100-143" (all 50 are 143). Do not copy
  those ranges.
- **`freq` is a hand-typed string, not inferred**, so `"W"` covers two different anchors:
  W-SUN for most sources, **W-FRI for `walmart_real`**.
- **Verified clean:** zero NaN values and zero duplicated timestamps across all 368 series;
  every index is monotonically increasing.

### 2.7 Does the benchmark contain personal or sensitive data?

All series are **aggregated commercial demand** — store-level, department-level, product-
family-level, or category-level totals. No individual-level records, identifiers, or
transactions are present in the distributed series.

**No formal personal-data or ethics review is recorded in the repository for any source.**
This datasheet does not assert that one was performed.

### 2.8 What else is distributed?

**2,260 committed result files** under `experiments/results/`:

- **380 `*_results.json`** — one per series, plus 12 legacy TMLR-era files. Model coverage
  is uneven by design: **333 contain ARIMA only**; **35 contain all seven baselines**.
- **`foundation/`, 35 files** — Chronos-small vs ARIMA on the original 35 series.
- **`scaling/`, 1,840 files** — 368 series × 5 Chronos sizes. Zero failures, zero null
  baselines.
- **`ranking/`, 2 files** — the Friedman/Nemenyi analysis, the only derived analysis
  artifacts in the tree.

**Per-timestep residuals are stripped from all committed results.** Diebold-Mariano testing
requires them, so post-hoc DM analysis on this corpus requires re-running with
`--keep-residuals` (see `README.md`).

---

## 3. Collection Process

**How was the data acquired?**

ShortSeq performed no primary collection. Series were obtained from third-party providers
and transformed by `experiments/scripts/prepare_datasets.py`, or bundled from the earlier
TMLR study.

**Origins, exactly as the repository records them:**

| source | origin as stated in repo |
|---|---|
| `rossmann` | Kaggle `rossmann-store-sales` competition dump |
| `walmart_real` | Kaggle `walmart-recruiting-store-sales-forecasting` competition dump |
| `favorita` | Kaggle `store-sales-time-series-forecasting`, a re-release of the older `favorita-grocery-sales-forecasting` data (Corporación Favorita, Ecuador) |
| `m4_weekly` | M4, via the `datasetsforecast` distribution |
| `m3_monthly` | M3, via the `datasetsforecast` distribution |
| `m4` | M4 Micro Monthly, from a committed R-exported wide table |
| `uci` | "UCI Online Retail (UK, weekly)" — **no repository entry id or URL recorded** |
| `dmart` | **No upstream provider recorded at all** — see below |
| `walmart`, `m5` | Not collected; seeded simulators |

**Sampling.** Where a source was subsampled, the rule is recorded in
`prepare_datasets.py`: Rossmann drew from the 935 of 1115 stores with gap-free histories;
`walmart_real` required gap-free series of ≥ 100 weeks; Favorita store-item pairs required
a non-late-opening store and ≤ 50% zeros. **The selection rule for the 24 `m4` series is
not recorded** — the manifest stores only `id,n,cv,ac1,bin`.

**D-Mart provenance is unresolved.** The loader states only "D-Mart real retail data (India,
daily)". No provider, URL, retailer agreement, collection period rationale, or statement of
whether the underlying `data/Dmart.csv` is a real point-of-sale extract or a generated file
exists anywhere in the repository. **Do not cite an origin for D-Mart on the strength of
this artifact.**

**Over what timeframe?** Recorded per source only through the series' own date indices, some
of which are fabricated (see 4.2). No collection dates, snapshot dates, dataset versions, or
checksums are recorded for any raw input.

---

## 4. Preprocessing, Cleaning, and Labeling

### 4.1 What was done to every series

- **Sorting:** every series is sorted chronologically before use. Verified: all 368 indices
  are monotonically increasing.
- **Imputation: none.** No loader fills, interpolates, or smooths. Verified zero NaN values.
- **Zeros are always retained as observations.** The only zero-based filtering is
  series-level eligibility (Favorita store-item pairs must be ≤ 50% zero), never row-level.
  Closed-day zeros in Rossmann are kept deliberately.
- **Gaps are handled by exclusion at preparation time, not by filling** — Rossmann and
  `walmart_real` restrict their pools to gap-free series; Favorita drops late-opening stores.
- **Deduplication: not performed and not enforced.** No `drop_duplicates` call exists. The
  data happens to contain no duplicate timestamps, but this is a property of the inputs
  rather than a guaranteed invariant.

### 4.2 Transformations that alter the series

- **`favorita_family` aggregates**: unit sales summed across all 54 stores to one daily
  series per product family. `favorita_store_item` does not aggregate.
- **`dmart` aggregates**: daily sums over the ~800 SKUs in each category. **The script
  performing this aggregation is not in the repository** — only a notebook that reads a path
  which does not exist. The committed category files were verified to match a groupby-sum of
  the committed SKU file to within floating-point tolerance (max abs diff 1.8e-12).
- **`rossmann` and `walmart_real` do not aggregate** — they select columns from one store or
  one (store, department) pair.

### 4.3 Fabricated date indices — important for seasonality claims

Three sources carry **synthetic timestamps**, because the upstream data does not publish
calendar dates:

- **`m4`**: a monthly index generated from 2000-01 onward.
- **`m4_weekly`**: a weekly index generated from 2000-01-02 onward.
- **`m3_monthly`**: dates are *real* but normalised from month-end to month-start; a minority
  of upstream series carry a 1900-01 placeholder start, confirmed present.

Any analysis that reads meaning into calendar position — holiday effects, month-of-year
seasonality, alignment across sources — is invalid for these series.

### 4.4 Is the raw data available?

Partly. Bundled and tracked: D-Mart, UCI, `m4`, and the two simulated sources. Gitignored and
**not** distributed: the three raw Kaggle dumps and the `datasetsforecast` caches. The
*derived* series for all six prepared sources **are** distributed (see 6.2).

**The derived files are not verifiably reproducible from raw.** `prepare_datasets.py` is a
one-time script with no checksum manifest, and no test re-derives from raw inputs — the test
suite reads only the committed derived files.

---

## 5. Uses

**What has it been used for?**
1. The TMLR 2025 study (35 series, classical and ML baselines).
2. A Chronos zero-shot validation (35 series).
3. A Chronos model-size scaling campaign (368 series × 5 sizes, 1,840 runs).
4. Friedman/Nemenyi ranking of the five Chronos sizes per stratum.

**What tasks is it appropriate for?** Univariate one-step-ahead point forecasting under a
rolling protocol, and comparison of forecasters on short histories.

**The evaluation protocol, precisely:** a contiguous 80/20 prefix/suffix split (no shuffling,
no cross-validation folds, no gap); rolling **one-step-ahead** forecasting in which the true
value is revealed after each step and the model updated. The number of forecasts equals the
test length. **There is no multi-step-ahead horizon anywhere in the codebase.** Reported
significance uses two-sided Diebold-Mariano at h=1 with Bonferroni correction applied
per-dataset, and Friedman + Nemenyi for multi-model ranking.

**What should it NOT be used for?**

- **Do not pool the strata.** Short and long series exhibit opposite-signed effects at the
  small-model end; a pooled ranking averages away the structure the benchmark exists to
  measure. The code refuses to emit a pooled ranking.
- **Do not treat all 368 series as independent** when computing aggregate statistics — see
  2.5 on Favorita.
- **Do not infer real-world Walmart or M5 behaviour** from the `walmart` and `m5` keys.
- **Do not make calendar-seasonality claims** about `m4`, `m4_weekly`, or `m3_monthly`.
- **Do not compare across models using the top-level results without checking coverage** —
  333 of 380 files contain ARIMA only.
- **Do not treat this as a demand-planning or inventory system.** It measures forecast error
  on historical aggregates; it has not been validated for operational deployment, and the
  short-history regime it targets is precisely where forecasts are least reliable.

---

## 6. Distribution

### 6.1 How is it distributed?

Via GitHub (above), with archival releases planned on Zenodo. The software is **MIT
licensed** (`LICENSE`). The `shortseq` package is intended for PyPI.

### 6.2 Licensing of the data — PARTIALLY RESOLVED

The repository distributes **339 derived series files** under `data/derived/`, plus bundled
raw files for three further sources. The MIT licence covers **the software only**. Upstream
licensing was reviewed on 2026-07-29; the results are below. Rows marked **PENDING** are
unresolved and are the blocker for an archival release.

| source | files distributed | upstream licence | verified | may derived series be redistributed? |
|---|---|---|---|---|
| `uci` | bundled raw + derived | **CC BY 4.0** | ✅ | **Yes**, with attribution (see 6.3) |
| `m3_monthly` | 100 derived | **CC BY 4.0** (Zenodo 4656298) | ✅ | **Yes**, with attribution (see 6.3) |
| `m4_weekly` | 50 derived | **none stated** | ✅ (confirmed absent) | **Unresolved** — see below |
| `m4` | bundled raw + derived | **none stated** | partial | **Unresolved** — see below |
| `rossmann` | 50 derived | not established | ❌ | **PENDING** |
| `walmart_real` | 50 derived | not established | ❌ | **PENDING** |
| `favorita_store_item` | 50 derived | not established | ❌ | **PENDING** |
| `favorita_family` | 33 derived | not established | ❌ | **PENDING** |
| `dmart` | bundled raw + derived | **no provenance recorded** | ❌ | **Unresolvable** as things stand |
| `walmart`, `m5` | bundled | n/a — original simulations | ✅ | Yes (MIT) |

**M4 (`m4`, `m4_weekly`) — no licence exists to rely on.** A CC BY 4.0 release of M4 Weekly
does exist on Zenodo (record 3892676, Monash), **but it is not the source used here.** The
`datasetsforecast` package downloads M4 from
`raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/` — the original
competition repository, which carries no LICENSE file and states no terms in its README. An
issue asking under which licence the data is released
(`Mcompetitions/M4-methods` issue 16, opened 2018) was closed without an answer. Citing the
Monash CC BY 4.0 for these series would be incorrect.

**The three Kaggle competitions — not verified, and the default points the wrong way.**
Their rules pages require login, so they could not be checked programmatically. Kaggle's
standard competition rules template states that participants must not "transmit, duplicate,
publish, redistribute or otherwise provide or make available the Data to any party not
participating in the Competition" — a restriction that applies whether or not commercial use
is permitted. **If that clause governs these three competitions, distributing the 183 derived
files from them is not permitted.** This is not a determination: per-competition terms vary,
Getting Started competitions are frequently more permissive than Featured or Recruiting ones,
and whether an aggregated derivative constitutes "the Data" is a genuine open question. It
must be checked at `kaggle.com/competitions/<slug>/rules` for each of
`rossmann-store-sales`, `walmart-recruiting-store-sales-forecasting`, and
`store-sales-time-series-forecasting`, and the result recorded in the table above.

**Until those rows are resolved, a user should treat the Rossmann, Walmart, and Favorita
derived series as of uncertain licensing status.** None of the above is legal advice.

**Acquisition paths**, for anyone regenerating rather than relying on the distributed files:

| source | how to obtain |
|---|---|
| `uci` | UCI ML Repository, dataset 352, DOI `10.24432/C5BW33` |
| `m3_monthly` | Zenodo record `4656298`, or via `datasetsforecast` |
| `m4`, `m4_weekly` | `Mcompetitions/M4-methods`, `Dataset/` directory, or via `datasetsforecast` |
| `rossmann` | Kaggle competition `rossmann-store-sales` (account + rules acceptance required) |
| `walmart_real` | Kaggle competition `walmart-recruiting-store-sales-forecasting` |
| `favorita` | Kaggle competition `store-sales-time-series-forecasting` |
| `dmart` | **Unknown** — no provenance recorded (Section 3) |

### 6.3 Required attributions

Two sources are CC BY 4.0, which permits redistribution **on condition that credit is given**.
That condition is satisfied here and must be carried by anyone redistributing these series:

- **UCI Online Retail** — Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5BW33. Licensed CC BY 4.0.
- **M3 Monthly** — Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., Montero-Manso, P.
  *M3 Monthly Dataset*. Zenodo. https://doi.org/10.5281/zenodo.4656298. Licensed CC BY 4.0.
  The depositors request citation of Makridakis, S. and Hibon, M. (2000), "The M3-competition:
  results, conclusions and implications", *International Journal of Forecasting* 16(4),
  451–476.

For the remaining sources, attribution is recorded in Section 3 as far as the repository
supports it; no licence obliges a specific form of credit because none is established.

### 6.4 Export controls or regulatory restrictions?

None known. None assessed.

---

## 7. Maintenance

**Who maintains it?** Aarav Shah, on a **best-effort basis**. No guaranteed response time is
offered.

**How to contact / report errors:** via GitHub issues on the repository (preferred, public
and searchable), or by email to <ashah264@ucr.edu>.

**Will it be updated?** Corrections will be made on a best-effort basis. Substantive changes
will be reflected in tagged releases; each archived release is immutable once minted on
Zenodo.

**Is there an erratum?** Not yet. The following are known open items at version 0.1.0 and
should be treated as the initial erratum:

1. The licensing status in **6.2 is partially resolved**. UCI and M3 are confirmed CC BY 4.0
   and are attributed in 6.3. M4 has **no licence at all** at the source actually used. The
   three Kaggle competitions (183 distributed files) are **still unverified** and are the
   blocker for an archival release.
2. **D-Mart has no recorded provenance** (Section 3).
3. The **undocumented UCI gap** at 2011-01-02 (Section 2.6).
4. Two loader docstrings **overstate length ranges** (Section 2.6).
5. The **`m4` sampling rule is unrecorded** (Section 3).
6. `data/derived/` is **not verifiably reproducible** from raw inputs (Section 4.4).

**Will older versions remain supported?** Zenodo releases are immutable and permanently
resolvable. The GitHub repository reflects the current state and is not a stable citation
target — cite the DOI.

**Can others contribute?** Contributions are welcome via pull request. There is no formal
contribution process at this version.

---

## Provenance of this datasheet

Every quantitative claim was computed from the committed repository state at
commit `d2e149f`, not transcribed from documentation. Where the repository does not support
a claim a datasheet would normally make, this document says so explicitly rather than
supplying an answer from outside knowledge. The unverified-licensing finding in 6.2 and the
undocumented UCI gap in 2.6 were both discovered during its preparation.

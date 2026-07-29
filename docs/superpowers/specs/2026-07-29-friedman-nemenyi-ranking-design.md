# Phase D (first analysis): Friedman/Nemenyi Ranking — Design

Date: 2026-07-29
Status: Approved by user, pending implementation

## Context

Phase C's first campaign (Chronos model-size scaling) is complete: 1,840
runs, 368 series x 5 Chronos sizes, zero failures. The descriptive
result **contradicts the project's original headline hypothesis**:

| stratum | tiny (8M) | mini | small | base | large (710M) |
|---|---|---|---|---|---|
| SHORT (n<200, 199 series) mean ratio | 0.972 | 0.942 | 0.930 | 0.914 | **0.892** |
| SHORT wins vs ARIMA | 109 | 125 | 123 | 127 | **141**/199 |
| LONG (n>=200, 169 series) mean ratio | 1.092 | 0.979 | 0.916 | 0.927 | **0.853** |
| LONG wins vs ARIMA | 55 | 85 | 106 | 103 | **132**/169 |

Scaling improves short-series accuracy monotonically from 8M to 710M,
with mean and median moving together. The planned claim — "billion-
parameter models lose to the sample mean on short series" — is not
supported.

These are **raw RMSE ratios with no significance testing**. Before any
of this becomes a paper claim, we need to know whether the ordering is
statistically real. That is what this spec covers.

## Why the rest of Phase D is deferred

The original Phase D bundle assumed the refuted hypothesis:

- **Boundary map** (`P(FM beats ARIMA | n, AC1)`, crossover contour):
  premised on there *being* a region where no foundation model wins.
  Chronos-large already wins 71% of short series. Fitting a logistic
  regression would still produce a contour — that is what fitting does —
  which would be an artifact of the method, not a finding.
- **Failure taxonomy** ("how FMs fail on short series"): they largely
  did not fail. Diagnosing the 58/199 losses is a legitimate but much
  smaller, different exercise.
- **Figures, Docker, leaderboard**: downstream of knowing what the
  actual finding is.

Deferring these is not abandoning them; it is refusing to build
analysis whose premise the data has already undermined.

## Goal

Answer two questions using only data already collected:

1. **Do the five Chronos sizes differ significantly**, within each
   length stratum?
2. **Is the `base` (200M) anomaly real?** On the long stratum, `base`
   (0.927) is *worse* than `small` (0.916), breaking the otherwise
   monotonic trend. Either that is a genuine effect worth investigating
   or it is noise — only a post-hoc test distinguishes them.

## Why not DM tests (and what must be fixed)

The standing protocol calls for Diebold-Mariano tests with Bonferroni
correction. **These cannot be computed from existing data.**

Two independent causes, both confirmed by inspection:

1. `run_size_scaling.py` never computed DM tests at all. The Phase C
   spec called for "a DM test vs. ARIMA's predictions where available";
   the implemented script only stores metrics. This was missed in
   review.
2. More fundamentally, **per-timestep residuals are stripped before
   writing** in both `run_size_scaling.py` (line 73) and
   `run_baselines.py` (line 148). A DM test compares two models' error
   series point-by-point; RMSE alone cannot reconstruct it.

Friedman/Nemenyi, by contrast, operate on per-series *scores*, so the
existing 368x5 RMSE grid supports them fully.

**In-scope fix:** add an opt-in `--keep-residuals` flag to both scripts
so a future run can support DM tests. Opt-in rather than always-on:
residuals are ~337 floats per series per model, which would
substantially inflate 1,840 result files for a capability not always
needed. This is a small change that prevents a future re-run hitting the
same dead end.

Whether to spend ~$12-15 of GPU time re-running with residuals retained
is a decision deferred until ranking results are in — they may make it
unnecessary or make it clearly worthwhile.

## Approach

`scipy.stats.friedmanchisquare` for the omnibus test, plus a
hand-implemented Nemenyi critical difference, rather than adding
`scikit-posthocs`.

Rationale: scipy is already a dependency and provides Friedman. Nemenyi
is not in scipy, but its critical difference is a short, well-documented
formula (Demsar 2006):

    CD = q_alpha * sqrt( k*(k+1) / (6*N) )

where k = number of models, N = number of series, and q_alpha is the
Studentized range statistic divided by sqrt(2). Hand-implementing it
with a unit test against a published worked example is more trustworthy
than adding a small, less-audited dependency that would need its own
verification anyway.

## Design

### `shortseq/evaluation/ranking.py`

Replace the Task 11 stub, keeping its existing signature
(`friedman_nemenyi(results: pd.DataFrame)`) so nothing downstream
breaks.

Input: DataFrame, rows = series, columns = models, values = a
lower-is-better metric (RMSE).

Returns a dict with:
- `n_series`, `n_models`
- `mean_ranks` per model (rank 1 = best on that series)
- `friedman_statistic`, `friedman_p_value`
- `critical_difference` at the requested alpha
- `pairwise`: for each model pair, the mean-rank difference and whether
  it exceeds CD (i.e. significantly different)

Design choices:
- Ranks are computed **within each series** (average ranks for ties),
  which is what makes Friedman non-parametric over heterogeneous series
  whose RMSE scales differ by orders of magnitude (n=53 UCI vs n=1684
  Favorita).
- If the Friedman test is not significant, the pairwise Nemenyi results
  are still returned but flagged `friedman_significant: False` — post-hoc
  comparisons after a non-significant omnibus test are not meaningful,
  and the flag makes that impossible to overlook.
- `q_alpha` values are tabulated for the small k range this benchmark
  needs (k = 2..10) at alpha = 0.05 and 0.10, with the source noted in a
  comment. Out-of-range k raises rather than silently extrapolating.

### `experiments/scripts/run_ranking.py`

Loads `experiments/results/scaling/*.json`, builds one RMSE matrix per
stratum, runs `friedman_nemenyi` on each, prints a readable report, and
writes `experiments/results/ranking/chronos_sizes_{stratum}.json`.

**Stratification is mandatory**, per the Phase B spec's binding
addendum: short (n<200) and long (n>=200) are ranked separately and a
pooled ranking is never produced. The two strata have opposite-signed
effects at the small end (tiny is 0.972 short vs 1.092 long), so a
pooled ranking would average away the very structure being measured.

### Fix: `--keep-residuals`

Add the flag to `run_baselines.py` and `run_size_scaling.py`. When set,
the `residuals` list survives into the written JSON. Default off, so
existing behaviour and file sizes are unchanged.

## Testing

- **Nemenyi CD against a published worked example** with known expected
  output — this is the one piece of genuinely new statistics and the
  only place a silent formula error could produce confident nonsense.
- Friedman on synthetic data where one model is uniformly best
  (must be significant) and where all models are identical
  (must not be).
- Ranking never emits a pooled cross-stratum result (mirrors the
  existing `test_size_scaling.py` guard).
- Tie handling: identical RMSEs receive average ranks.
- `--keep-residuals` off by default; on when requested.

## Verification

- New tests pass; existing 104 pass unchanged.
- `run_ranking.py` produces results for both strata from committed data,
  with no new compute and no GPU.
- The `base`-vs-`small` long-stratum comparison is explicitly reported
  as either within or beyond critical difference.

## Expected finding (stated in advance)

Given the monotonic mean/median trend and the win-count spread
(109 -> 141 short; 55 -> 132 long), the omnibus Friedman test will very
likely be significant in both strata. The genuinely uncertain parts are
(a) which *adjacent* size pairs separate under Nemenyi — plausibly few,
since adjacent sizes are close — and (b) whether `base` and `small`
differ on the long stratum. **If adjacent pairs mostly do not separate,
that materially weakens "scaling helps monotonically" into "the extremes
differ but the steps do not," and must be reported that way.**

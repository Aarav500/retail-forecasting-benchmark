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

## Outcome (recorded 2026-07-29, after the analysis)

The prediction above is left exactly as written. This section records what
actually happened.

Source: `experiments/results/ranking/chronos_sizes_short.json` and
`experiments/results/ranking/chronos_sizes_long.json`, produced by
`experiments/scripts/run_ranking.py` from the committed 1,840-file scaling
corpus at alpha = 0.05, k = 5 sizes, SHORT N = 199, LONG N = 169.

### Omnibus: as predicted

Significant in both strata, by a wide margin.

| stratum | N | Friedman chi2 | p | significant |
|---|---:|---:|---:|:--|
| SHORT (n<200) | 199 | 102.6471 | 2.686e-21 | yes |
| LONG (n>=200) | 169 | 284.9636 | 1.896e-60 | yes |

### Adjacent pairs: the prediction did NOT hold

The prediction expected "plausibly few" adjacent pairs to separate. In
fact **5 of the 8 adjacent pairs separate**:

| stratum | adjacent pairs separating | detail |
|---|---|---|
| SHORT | 2 of 4 | tiny–mini and base–large separate; mini–small and small–base do not |
| LONG | 3 of 4 | tiny–mini, mini–small and base–large separate; only small–base does not |

**The escape clause was therefore not triggered.** Stating that
explicitly rather than leaving it to inference: the finding is *not*
"the extremes differ but the steps do not," and it must not be written up
that way. Most of the ladder does separate, especially on the long
stratum.

The broader picture is stronger still: **8 of 10 pairs separate on short,
9 of 10 on long.** The only pair that fails in both strata is small–base.

**One marginal call, stated rather than smoothed.** On the short stratum
`tiny_vs_mini` clears the critical difference by only **0.0249** — a gap
of 0.4573 against a CD of 0.4324. It is a separation at alpha = 0.05 and
is counted as one above, but it is a near-threshold one; reporting it as
a clean separation without the margin would overclaim. The corresponding
long-stratum gap (0.7692 against CD 0.4692) is not marginal.

### Goal Q2: the `base` anomaly is noise — and the answer is stronger than "non-significant"

`small_vs_base` is the one comparison that fails in both strata, and it
fails with room to spare: **0.2739 against CD 0.4324 on short, 0.1775
against CD 0.4692 on long.** The anomaly is not confirmed.

The long stratum says more than that: **the rank ordering reverses the
anomaly.** Mean rank `base` = 2.5947 *beats* `small` = 2.7722, where the
raw RMSE ratios that raised the question in the first place had `base`
(0.927) *worse* than `small` (0.916). Within-series ranking does not
merely fail to confirm the break in monotonicity; it points the other
way.

The mechanism is visible in the underlying grid. Across the 169 long
series:

- `base` beats `small` head-to-head on raw RMSE in **95 of 169** series
  (56%), and `base`'s *median* raw RMSE is the lower of the two (282.97
  vs 289.94).
- `base`'s *mean* raw RMSE is nevertheless the higher (1664.11 vs
  1498.99) — the mean is being pulled by a minority of large-magnitude
  series on which `base` does worse.
- The same concentration shows up on the ratio scale: **five Rossmann
  store series** (ARIMA RMSE 2,034–5,754) account for **91%** of the
  *net* summed per-series difference between `base`'s and `small`'s RMSE
  ratios (+1.69 of +1.85 across all 169).

Stated honestly, the summary statistics disagree with each other:
`base` wins the head-to-head count and the median raw RMSE; `small` wins
the mean ratio (0.9265 vs 0.9156) and the median ratio (0.9518 vs
0.9190). That disagreement is precisely why a rank test is the right
arbiter. A mean over per-series ratios lets five series outvote the other
164; a within-series rank gives every series exactly one rank position
regardless of its error magnitude — which is the artifact rank-based
testing exists to defeat. Here it is enough to flip the sign of the
comparison, while leaving the difference comfortably inside the critical
difference either way.

The reading that follows: **there is no detectable `small`-vs-`base`
difference in either stratum, and the apparent break in monotonicity is
best explained as a summary-statistic artifact of averaging ratios across
series whose error magnitudes span orders of magnitude, rather than as a
property of the models.**

### Scope caveat

This establishes an ordering *among the five Chronos sizes*, per stratum,
on the committed corpus. It does **not** establish per-series
Chronos-vs-ARIMA significance. That would need Diebold-Mariano tests,
which need per-timestep residuals, which the committed corpus does not
carry (see "Why not DM tests" above). The `--keep-residuals` flag added
in this phase makes a future run *capable* of supporting them; whether to
spend the GPU time on a re-sweep is a separate decision and remains
deferred.

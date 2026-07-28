# Phase C (first campaign): Chronos Model-Size Scaling — Design

Date: 2026-07-28
Status: Approved by user, pending implementation

## Context

Phase C of the NeurIPS roadmap is the experiment-campaign phase
(context-length sensitivity, multi-horizon, model-size scaling,
fine-tuning, ensembling, temperature sweeps). It was scoped as
depending on Phases A + B.

**Phase B is complete** (368 series across 10 sources).
**Phase A is 1/8 complete** — only Chronos is wrapped; TimesFM, Moirai,
Moment, Timer, TTM, Lag-Llama, and ForecastPFN remain stubs.

That dependency gap rules out the cross-model parts of Phase C
(model-vs-model comparison, "ensemble the top-3 foundation models").
It does **not** rule out model-size scaling, which needs only Chronos:
the wrapper already supports all five published sizes
(tiny 8M, mini 20M, small 46M, base 200M, large 710M).

This spec covers that one campaign, and only that one. It is the
highest-value experiment currently runnable because it directly tests
the paper's headline claim: **more parameters do not help below the
short-series boundary.**

## Goal

Produce, for every one of the 368 registry series, the RMSE of all five
Chronos sizes against the already-committed ARIMA baseline — enough to
answer "does scaling Chronos from 8M to 710M parameters improve
short-series forecasting?" with real numbers and DM tests.

## Non-goals

- The other three Chronos-only campaigns (context-length sensitivity,
  multi-horizon, temperature sweep). Deferred; they are less central to
  the headline claim and are cheaper to redo once more models exist.
- The other 7 foundation models (Phase A continuation).
- Fine-tuning and ensembling campaigns (need multiple models).
- The boundary-map regression and Friedman/Nemenyi ranking themselves —
  those are Phase D consumers of this campaign's output. This spec
  produces the data they need, not the analysis.
- Re-running or modifying the classical baselines.

## Approach

A new script, `experiments/scripts/run_size_scaling.py`, rather than
extending `run_chronos.py` with a `--sizes` flag.

Rationale: `run_chronos.py` re-runs ARIMA alongside Chronos for each
invocation. Sweeping five sizes through it would run ARIMA five times
per series — not merely wasteful, but capable of producing five
slightly different ARIMA numbers per series if anything in the stack is
nondeterministic, muddying exactly the comparison this campaign exists
to make. The new script instead loads ARIMA's RMSE once from the
committed `experiments/results/*_results.json` and varies only the
Chronos size.

## Design

### Inputs
- `shortseq.datasets.registry.load_all()` — 368 series.
- `experiments/results/{dataset}_results.json` — existing per-dataset
  classical-baseline results; ARIMA's `rmse` is the reference.

### Per (series, size) work
Fit `ChronosForecaster(size=<size>)` and run `predict_rolling` over the
strict 80/20 temporal split (same protocol as every other model in the
benchmark — no look-ahead, one step at a time, true value revealed
after each step). Compute metrics with the existing
`shortseq.evaluation.metrics.compute_metrics`, and a DM test vs. ARIMA's
predictions where available.

### Output
One JSON per (dataset, size) at
`experiments/results/scaling/{dataset}_chronos_{size}.json`, written
**incrementally as each completes** — a crash at hour 4 of a 5-hour run
must not lose the first 4 hours.

Schema matches the existing results convention (`dataset`, `n_train`,
`n_test`, `metrics`, `dm_tests`, `failed_models`) so Phase D can consume
scaling results and baseline results with the same code.

### Resumability
Skip any (dataset, size) whose output JSON already exists, unless
`--force`. `run_baselines.py` lacks this and it was flagged in its
review as a gap; at a 5-hour runtime it stops being cosmetic.

### Stratified summary — required, not optional
Per the Phase B spec's binding addendum, the script's summary output
reports short (n < 200) and long (n >= 200) strata **separately** and
never emits a single pooled figure. The registry splits 199 short / 169
long, so a pooled mean would average two regimes with opposite expected
outcomes and obscure the very effect being measured.

### Failure handling
Each (series, size) is individually wrapped: a failure records the
exception string in `failed_models` and continues. Chronos-base (200M)
and especially large (710M) may OOM on the T4's 16GB alongside rolling
state. A size that OOMs must be **recorded as failed, visibly**, never
silently skipped — an absent result and a failed result mean very
different things when interpreting a scaling curve.

## Infrastructure

Relaunch a `g4dn.xlarge` (T4, 16GB) in `us-east-1`, reusing the
artifacts already created for the Chronos campaign and still present:
AMI `ami-0b9c598335204bf5b`, security group `sg-0056c10f41d9c88e6`
(SSH from the user's IP only), key pair `chronos-gpu-key`, public
subnet `subnet-09ea79c58650980ee`, Elastic IP
`eipalloc-01c116ab335572014`.

Note the earlier stopped instance (`i-0d99558093edd2c09`) also still
exists; starting it is an alternative to launching fresh, and avoids
redoing dependency installation.

Estimated cost, extrapolated from the measured Chronos throughput
(0.022 s per forward pass on the T4) against the registry's ~50,500
rolling steps per full pass: roughly 5 GPU-hours for all five sizes,
about $3 at the on-demand rate. Larger sizes dominate; if `large`
proves impractically slow, that is itself a reportable finding.

**Cost control:** Claude launches/starts and stops the instance via the
AWS CLI but asks for explicit user confirmation before any action that
begins billing, and before leaving it running unattended. The instance
is stopped as soon as results are copied back.

## Verification

- Script runs end-to-end on a single dataset before the full sweep
  (`--datasets dmart_food`), producing 5 JSONs with sane RMSEs.
- Full sweep produces 368 x 5 = 1840 result files, minus any recorded
  failures.
- Existing test suite (98 passed, 1 skipped) still passes unchanged.
- A structural test covers the new script's summary/stratification
  logic without requiring a GPU (mirroring how
  `tests/test_experiment_scripts.py` covers `run_baselines.py`).
- Results committed; instance confirmed stopped.

## Expected finding (stated in advance, to avoid post-hoc storytelling)

The paper's thesis predicts flat-or-noisy scaling on the short stratum
(more parameters buy little or nothing when n < 200) and a clearer
downward RMSE trend on the long stratum. **If the data shows otherwise
— e.g. large clearly beating small on short series — that is the
result, and it gets reported as such.** The prior Chronos-small run
already produced a genuinely mixed picture (16 wins / 19 losses vs.
ARIMA), so an unambiguous outcome should not be assumed.

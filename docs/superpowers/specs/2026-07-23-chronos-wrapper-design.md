# Chronos Foundation Model Wrapper — Design

Date: 2026-07-23
Status: Approved by user, pending implementation

## Context

The `shortseq` package (built across a prior 15-task scaffold plan,
`docs/superpowers/plans/2026-07-23-shortseq-scaffold.md`) has 7 real
baseline models (ARIMA, SARIMA, XGBoost, LSTM, Hybrid, SeasonalNaive,
Prophet) and 8 honest stub classes for foundation models
(`shortseq/models/foundation/*.py`), all implementing a shared
`BaseForecaster` interface (`fit(train) -> self`,
`predict_rolling(test) -> Forecast`). The stubs exist so the shape is
right, but none of them run — wrapping a real foundation model needs
GPU access and new heavy dependencies that were explicitly out of scope
for the scaffold.

This is the first phase of that follow-on work: wrap **one** foundation
model (Chronos) for real, end-to-end, before attempting the other 7.
The full NeurIPS execution plan calls for 8 foundation models across
500+ series with multiple follow-on experiment campaigns (context-length
sensitivity, multi-horizon, fine-tuning, ensembling, temperature
sweeps) — that scope was explicitly decomposed (with the user) into
independent sub-projects:

- **A. Foundation model wrapper phase** (this spec) — GPU env + wrap
  Chronos, validate against the 35 datasets already in the registry.
- **B. New dataset acquisition** — the 7 additional sources needed to
  reach 500+ series. Independent of A, no GPU needed.
- **C. Extended experiment campaigns** — context-length, multi-horizon,
  fine-tuning, ensembling, temperature. Depends on A.
- **D. Statistical analysis + packaging** — ranking, boundary analysis,
  Docker/leaderboard. Depends on A+B+C.

This spec covers **A, and only for Chronos** — the other 7 foundation
models stay as stubs until this pattern is proven out.

## Goals

1. Get one real foundation model (Chronos) producing real forecasts
   against real data, comparable (same protocol, same DM test) to the
   already-working classical baselines.
2. Prove out the GPU workflow (provision → install → run → copy results
   → stop) so wrapping the remaining 7 models is mechanical repetition,
   not new infrastructure design.
3. Keep the existing CPU-only test suite (64 tests, runs on the
   Windows dev machine with no GPU) green and unaffected.
4. Control cost: the GPU instance only runs for as long as work is
   actually happening on it, with explicit confirmation before it's
   left running.

## Non-goals (explicitly out of scope this phase)

- The other 7 foundation models (TimesFM, Moirai, Moment, Timer, TTM,
  Lag-Llama, ForecastPFN) — stay as Task 10's stubs.
- New dataset sources (Phase B) — Chronos is validated against the
  existing 35 real/calibrated series only.
- Multi-horizon, context-length sensitivity, fine-tuning, ensembling,
  temperature-sensitivity experiments (Phase C) — single zero-shot
  rolling-forecast run only, one model size.
- CRPS/coverage metrics (still stubs from Task 5) — `Forecast.dist` gets
  populated with Chronos's sample paths so those stubs have real data
  to consume *later*, but implementing CRPS/coverage itself is not part
  of this phase.
- Friedman/Nemenyi ranking, boundary-map analysis (Task 11's stubs) —
  still meaningless with only 2 models (ARIMA-family + Chronos) worth
  comparing; unchanged.
- A hosted leaderboard, Docker image, or any packaging work.

## Infrastructure

- **Instance:** `g4dn.xlarge` (1× NVIDIA T4, 16GB VRAM, 4 vCPU) in
  `us-east-1`, launched from an AWS Deep Learning AMI (CUDA/PyTorch
  drivers preinstalled, avoids driver-install pain).
- **Networking:** default VPC (`vpc-02925a1ca38ef69fd`, confirmed
  present), a dedicated security group allowing inbound SSH (port 22)
  only from the user's current IP.
- **Access:** a new EC2 key pair, SSH used for setup and running
  scripts; no persistent service/API exposed.
- **Provisioning owner:** Claude launches/stops/terminates the instance
  via the AWS CLI (confirmed working: account `325527186655`,
  on-demand G/VT quota is 4 vCPUs — exactly enough for one
  `g4dn.xlarge`, no quota-increase request needed) — but **always with
  explicit user confirmation before any action that starts billing or
  is destructive** (launch, and anything that would leave it running
  unattended). Stopping/terminating after work is done does not need
  re-confirmation — that's cost-saving, not risk.
- **Cost shape:** on-demand ~$0.526/hr; expect a small number of hours
  total for setup + one validation run, not a standing resource.

## Code changes

### `shortseq/models/foundation/chronos.py`

Replace the stub with a real implementation using the `chronos-forecasting`
pip package (Amazon's own library):

```python
class ChronosForecaster(BaseForecaster):
    def __init__(self, size: str = "small"):
        super().__init__()
        self.size = size
        self.name = f"Chronos-{size}"
        self._pipeline = None      # loaded lazily in fit()
        self._context = None       # torch tensor, grows each predict_rolling step

    def fit(self, train: pd.Series) -> "ChronosForecaster":
        # Loads ChronosPipeline.from_pretrained("amazon/chronos-t5-{size}", ...)
        # onto GPU if available. Zero-shot: no gradient training happens here,
        # this just loads the model and stores the initial context.
        ...

    def predict_rolling(self, test: pd.Series) -> Forecast:
        # One step at a time, matching ARIMA/XGBoost's existing loop:
        # for each test point, pipeline.predict(context, prediction_length=1)
        # -> take median of sampled trajectories as the point forecast,
        # append the TRUE test value to context, repeat.
        # Forecast.dist is populated with the raw sample paths (reserved
        # for this since Task 3 — CRPS/coverage stubs can consume it later).
        ...
```

Exact `fit`/`predict_rolling` bodies get written during implementation,
not fully specified here — the loop shape and interface contract above
are fixed, the internals (device placement, dtype, batching within a
single step) are implementation detail.

### Dependencies

New, GPU-only dependencies (`chronos-forecasting`, `torch`) do **not**
go into the base `pyproject.toml` used by the Windows CPU dev
environment — they'd make `pip install -e .` there pull in a multi-GB
CUDA-enabled torch wheel for a machine that will never use it. Instead:
a separate `requirements-gpu.txt` (or a `[project.optional-dependencies]
gpu = [...]` extra) installed only on the EC2 instance.

### Testing

The existing 64-test suite must stay green with no GPU/torch/chronos
installed (verified: it doesn't import `shortseq.models.foundation.chronos`
anywhere yet outside `tests/test_foundation_stubs.py`, which only checks
the stub's `NotImplementedError`). New Chronos-specific tests
(`tests/test_chronos_model.py` or similar) get collected but
auto-skipped when `torch`/`chronos_forecasting` aren't importable
(`pytest.importorskip` or an equivalent marker) — so they simply don't
run on the Windows machine, and only execute for real on the GPU
instance as a separate verification pass.

### Validation script

A new script (name TBD during planning — likely
`experiments/scripts/run_chronos.py`, mirroring `run_baselines.py`'s
shape) runs `ChronosForecaster(size="small")` against every dataset in
`shortseq.datasets.registry.load_all()` (the same 35 real/calibrated
series `run_baselines.py` already covers), computes the same
`compute_metrics`/`diebold_mariano_test` comparison against ARIMA, and
writes results in the same JSON shape `run_baselines.py` uses (so a
future merge into one combined script, or a shared results directory,
is straightforward) — kept as a separate script rather than folded into
`run_baselines.py` itself, since `run_baselines.py` must stay
GPU-dependency-free for the CPU dev environment.

## Verification plan

- CPU-side: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/ -v`
  still shows all existing tests passing (Chronos tests skipped, not
  failed, due to missing torch/chronos on this machine).
- GPU-side (on the EC2 instance): Chronos-specific tests pass; running
  the validation script against at least D-Mart Food produces a
  sensible RMSE (same order of magnitude as ARIMA's 255.76, not
  necessarily better — the whole point of the NeurIPS story is that it
  might *not* beat ARIMA on short series, and that's a valid, expected
  result, not a bug) and a real DM-test p-value against ARIMA.
- Instance is stopped (not terminated, in case a quick re-run is needed)
  once the validation run completes and results are copied back to the
  local repo / committed.

## Open questions for the implementation plan

- Exact SSH/file-sync mechanism (scp vs. git clone on the instance vs.
  rsync) — implementation detail, not a design decision.
- Whether results get committed back into `experiments/results/`
  alongside the classical baselines' JSON, or into a separate
  `experiments/results/foundation/` subdirectory — lean toward the
  latter to keep GPU-dependent artifacts visually separated, but not
  load-bearing enough to block on now.

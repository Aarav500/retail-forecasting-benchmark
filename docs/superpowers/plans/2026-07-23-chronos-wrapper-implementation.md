# Chronos Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for Tasks 1-3 (local code, no real money involved). **Tasks 4-7 involve real AWS billing and MUST be executed by the coordinator directly, with explicit user confirmation before every action that starts billing or is destructive — do NOT delegate these to an autonomous implementer subagent that could launch/terminate paid infrastructure without a human in the loop.** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `ChronosForecaster` stub with a real zero-shot wrapper, validate it end-to-end on a GPU against the 35 real datasets already in the registry, producing DM-tested RMSE numbers comparable to the existing classical baselines.

**Architecture:** Local (Windows, no GPU) work writes the wrapper code, tests (gated behind `pytest.importorskip` so they don't run without torch/chronos installed), and a validation script — all committable and reviewable without touching AWS. Separately, a `g4dn.xlarge` EC2 instance gets provisioned, the repo synced to it, GPU-only deps installed there, and the actual model run for real — that part is infrastructure execution, not code review, and stays under direct human confirmation throughout.

**Tech Stack:** `chronos-forecasting` (Amazon's official pip package), PyTorch, AWS EC2 (Deep Learning AMI), existing `shortseq` package conventions.

---

## Design reference

Full rationale: `docs/superpowers/specs/2026-07-23-chronos-wrapper-design.md`. This plan implements that spec's Phase A only (Chronos, one model, against the existing 35 datasets — not the other 7 foundation models, not new datasets, not extended experiments).

## Facts already verified this session (don't re-verify, just use)

- AWS account `325527186655`, region `us-east-1`, authenticated via the `AWS_API_MCP_Server` tools (`call_aws` with `aws ...` CLI commands).
- Default VPC `vpc-02925a1ca38ef69fd` exists.
- On-demand G/VT instance quota is 4 vCPUs — exactly enough for one `g4dn.xlarge` (4 vCPU), no quota-increase request needed.
- The CPU dev venv is `C:/venvs/shortseq/Scripts/python.exe`, editable-installed pointing at the main repo root (`C:\Users\aarav\OneDrive\Desktop\Incomplete papers\retail-forecasting-benchmark`). This worktree (`.worktrees/chronos-wrapper`) is a separate checkout on branch `feature/chronos-wrapper` — the same shared venv works for local (non-GPU) test runs here since none of Tasks 1-3 need new Windows-side dependencies.

---

### Task 1: GPU dependency declaration

**Files:**
- Create: `requirements-gpu.txt`

- [ ] **Step 1: Write `requirements-gpu.txt`**

```
torch>=2.2.0
chronos-forecasting>=1.4.0
```

Kept separate from `pyproject.toml`/`requirements.txt` deliberately: installing `torch` on the Windows CPU dev machine would pull a multi-GB wheel for a device that will never run it, and isn't needed for anything except this GPU-only model. Only the EC2 instance ever installs this file.

- [ ] **Step 2: Commit**

```bash
git add requirements-gpu.txt
git commit -m "feat: add requirements-gpu.txt for GPU-only foundation model dependencies"
```

---

### Task 2: Real `ChronosForecaster` implementation

**Files:**
- Modify: `shortseq/models/foundation/chronos.py` (replace the Task 10 stub)
- Test: `tests/test_chronos_model.py`

- [ ] **Step 1: Write the test (will be skipped locally, run for real only on GPU)**

`tests/test_chronos_model.py`:
```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("chronos")

from shortseq.datasets.dmart import load_dmart
from shortseq.models.foundation.chronos import ChronosForecaster


def _split(series, frac=0.8):
    i = int(len(series) * frac)
    return series.iloc[:i], series.iloc[i:]


def test_chronos_forecaster_fits_and_predicts_real_dmart_food():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = ChronosForecaster(size="small").fit(train)
    forecast = model.predict_rolling(test)
    assert forecast.point.shape == (len(test),)
    assert model.train_time_ is not None
    assert model.pred_time_ is not None
    assert model.name == "Chronos-small"
    # dist should carry one sample-path list per test point
    assert len(forecast.dist) == len(test)


def test_chronos_predict_rolling_is_idempotent_across_repeated_calls():
    train, test = _split(load_dmart()["dmart_food"].series)
    model = ChronosForecaster(size="small").fit(train)
    fc1 = model.predict_rolling(test)
    fc2 = model.predict_rolling(test)
    assert np.array_equal(fc1.point, fc2.point)
```

- [ ] **Step 2: Run test to verify it's SKIPPED (not failed) on the Windows dev machine**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/test_chronos_model.py -v`
Expected: `2 skipped` (torch/chronos aren't installed here — that's correct and expected, not a failure). If it says `FAILED` instead of `SKIPPED`, the `importorskip` lines are wrong — fix them before continuing.

- [ ] **Step 3: Write the real `shortseq/models/foundation/chronos.py`** (replaces the Task 10 stub entirely)

```python
"""Chronos (Amazon) — zero-shot foundation model forecaster.

Uses the official `chronos-forecasting` package. Zero-shot: `fit()` only
loads the pretrained model and stores context; no gradient training
happens. `predict_rolling` mirrors the classical baselines' one-step
rolling protocol (predict 1 step, reveal the true value, extend
context, repeat) rather than a single batch call, so DM tests stay
comparable across all models. `Forecast.dist` is populated with the
raw sampled trajectories per step (reserved for this since Task 3 —
CRPS/coverage stubs can consume it later).
"""
import time

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

from ..base import BaseForecaster, Forecast

_MODEL_IDS = {
    "tiny": "amazon/chronos-t5-tiny",
    "mini": "amazon/chronos-t5-mini",
    "small": "amazon/chronos-t5-small",
    "base": "amazon/chronos-t5-base",
    "large": "amazon/chronos-t5-large",
}


class ChronosForecaster(BaseForecaster):
    def __init__(self, size: str = "small", num_samples: int = 20):
        super().__init__()
        if size not in _MODEL_IDS:
            raise ValueError(f"Unknown Chronos size {size!r}; choose from {list(_MODEL_IDS)}")
        self.size = size
        self.num_samples = num_samples
        self.name = f"Chronos-{size}"
        self._pipeline = None
        self._context: list = []

    def fit(self, train: pd.Series) -> "ChronosForecaster":
        t0 = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipeline = ChronosPipeline.from_pretrained(
            _MODEL_IDS[self.size],
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        self._context = list(train.values.astype(float))
        self.train_time_ = time.time() - t0
        return self

    def predict_rolling(self, test: pd.Series) -> Forecast:
        t0 = time.time()
        context = list(self._context)
        test_values = test.values.astype(float)
        points = []
        dist = []
        for i in range(len(test_values)):
            context_tensor = torch.tensor(context, dtype=torch.float32)
            samples_tensor = self._pipeline.predict(
                context=context_tensor,
                prediction_length=1,
                num_samples=self.num_samples,
            )
            samples = samples_tensor[0, :, 0].float().cpu().numpy()
            points.append(float(np.median(samples)))
            dist.append(samples.tolist())
            context.append(test_values[i])  # true value revealed, no look-ahead
        self.pred_time_ = time.time() - t0
        return Forecast(point=np.array(points), dist=dist)
```

- [ ] **Step 4: Update `tests/test_foundation_stubs.py` — Chronos is no longer a stub**

Read the current `tests/test_foundation_stubs.py` (from Task 10). It parametrizes `STUB_CLASSES` over all 8 foundation models including `ChronosForecaster`, asserting `fit`/`predict_rolling` raise `NotImplementedError`. That's no longer true for Chronos. Remove `ChronosForecaster` from the `STUB_CLASSES` list and from the default-name test (`test_foundation_stub_default_names`) — the other 7 (TimesFM, Moirai, Moment, Timer, TTM, Lag-Llama, ForecastPFN) stay in that list unchanged, since only Chronos is being un-stubbed in this plan.

- [ ] **Step 5: Run the full local suite to confirm nothing regressed**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/ -v`
Expected: same pass count as before minus the 2 Chronos-stub assertions removed from `test_foundation_stubs.py`, plus `tests/test_chronos_model.py`'s 2 tests showing as `SKIPPED` (not failed, not passed — skipped, since torch/chronos aren't installed on this machine).

- [ ] **Step 6: Commit**

```bash
git add shortseq/models/foundation/chronos.py tests/test_chronos_model.py tests/test_foundation_stubs.py
git commit -m "feat: implement real ChronosForecaster (zero-shot, rolling one-step protocol)"
```

---

### Task 3: Validation script

**Files:**
- Create: `experiments/scripts/run_chronos.py`

- [ ] **Step 1: Write `experiments/scripts/run_chronos.py`**

```python
"""CLI to run ChronosForecaster against every implemented ShortSeq dataset
and compare it to ARIMA via the same metrics/DM-test protocol
run_baselines.py uses for the classical baselines.

Kept separate from run_baselines.py so the CPU-only orchestration script
never needs torch/chronos-forecasting as a dependency — this script only
runs on the GPU instance, where those are installed (see
requirements-gpu.txt).
"""
import argparse
import json
from pathlib import Path

from shortseq.datasets.registry import load_all
from shortseq.evaluation.dm_test import bonferroni_correct, diebold_mariano_test
from shortseq.evaluation.metrics import compute_metrics
from shortseq.models.arima import ARIMAForecaster
from shortseq.models.foundation.chronos import ChronosForecaster

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "foundation"


def run_dataset(name: str, dataset, size: str, split: float = 0.8) -> dict:
    series = dataset.series
    split_idx = int(len(series) * split)
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]

    results = {}
    preds_by_model = {}
    failed_models = {}

    for model_name, model in [("ARIMA", ARIMAForecaster()), (f"Chronos-{size}", ChronosForecaster(size=size))]:
        try:
            model.fit(train)
            forecast = model.predict_rolling(test)
            preds = forecast.point[: len(test)]
            metrics = compute_metrics(test.values, preds, model.name, model.train_time_, model.pred_time_)
            preds_by_model[model_name] = preds
            results[model_name] = metrics
            print(f"  {model_name}: RMSE={metrics['rmse']:.2f}")
        except Exception as exc:
            failed_models[model_name] = str(exc)
            print(f"  [{name}/{model_name}] failed: {exc}")

    dm_tests = {}
    if "ARIMA" in preds_by_model and f"Chronos-{size}" in preds_by_model:
        dm_stat, p_val = diebold_mariano_test(test.values, preds_by_model["ARIMA"], preds_by_model[f"Chronos-{size}"])
        dm_tests[f"Chronos-{size}"] = {"dm_stat": dm_stat, "p_value": p_val}
        corrected = bonferroni_correct({f"Chronos-{size}": p_val})
        dm_tests[f"Chronos-{size}"]["significant_bonferroni"] = corrected[f"Chronos-{size}"]["significant"]

    output = {
        "dataset": name,
        "n_train": split_idx,
        "n_test": len(test),
        "metrics": {k: {mk: mv for mk, mv in v.items() if mk != "residuals"} for k, v in results.items()},
        "dm_tests": dm_tests,
        "failed_models": failed_models,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{name}_chronos_{size}_results.json", "w") as f:
        json.dump(output, f, indent=2)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="small", choices=["tiny", "mini", "small", "base", "large"])
    parser.add_argument("--datasets", nargs="+", help="Only run these dataset names")
    args = parser.parse_args()

    datasets = load_all()
    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    print(f"Running ARIMA vs Chronos-{args.size} on {len(datasets)} datasets...")
    for name, dataset in datasets.items():
        print(f"\n{'=' * 60}\n{name} | n={dataset.n} | freq={dataset.freq}\n{'=' * 60}")
        run_dataset(name, dataset, args.size)
    print(f"\nDone. Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add experiments/scripts/run_chronos.py
git commit -m "feat: add run_chronos.py validation script (ARIMA vs Chronos, all datasets)"
```

---

### Task 4: Provision the GPU instance

**⚠️ COORDINATOR EXECUTES THIS TASK DIRECTLY. Do not dispatch to an autonomous subagent. Every step that creates a billable resource requires the user's explicit go-ahead in chat immediately before running it — not a blanket "yes" from earlier in the conversation.**

- [ ] **Step 1: Confirm with the user, in chat, immediately before running anything in this task** — restate: "About to create a security group, key pair, and launch one `g4dn.xlarge` (~$0.526/hr on-demand) in `us-east-1`. Proceed?" Wait for an explicit yes.

- [ ] **Step 2: Create a security group restricted to the user's current IP**

Get the current public IP first (needed for the security group rule):
```bash
aws ec2 describe-security-groups --filters Name=group-name,Values=chronos-gpu-sg --region us-east-1
```
If it doesn't exist yet, create it:
```bash
aws ec2 create-security-group --group-name chronos-gpu-sg --description "SSH-only access for Chronos GPU box" --vpc-id vpc-02925a1ca38ef69fd --region us-east-1
```
This returns a `GroupId`. Then find the current public IP (ask the user for it, or use a command like `curl -s https://checkip.amazonaws.com` if available) and authorize SSH from it only:
```bash
aws ec2 authorize-security-group-ingress --group-id <GroupId> --protocol tcp --port 22 --cidr <USER_IP>/32 --region us-east-1
```

- [ ] **Step 3: Create a key pair**

```bash
aws ec2 create-key-pair --key-name chronos-gpu-key --query "KeyMaterial" --output text --region us-east-1
```
Save the output to a local `.pem` file (e.g. in the scratchpad directory, NOT committed to git — add `*.pem` to `.gitignore` if not already covered). Set restrictive permissions on it if the OS supports it.

- [ ] **Step 4: Find the latest Deep Learning AMI**

```bash
aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu*" "Name=state,Values=available" --query "sort_by(Images, &CreationDate)[-1].[ImageId,Name]" --output text --region us-east-1
```
Use whatever AMI ID this returns — don't hardcode one, AMIs update frequently and a stale ID may not exist.

- [ ] **Step 5: Confirm with the user again, showing the exact launch parameters (instance type, AMI ID, security group, key pair), then launch**

```bash
aws ec2 run-instances --image-id <AMI_ID> --instance-type g4dn.xlarge --key-name chronos-gpu-key --security-group-ids <GroupId> --subnet-id <a subnet in vpc-02925a1ca38ef69fd> --count 1 --region us-east-1 --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=chronos-gpu}]"
```
(Get a subnet ID first via `aws ec2 describe-subnets --filters Name=vpc-id,Values=vpc-02925a1ca38ef69fd --region us-east-1` if not already known.)

- [ ] **Step 6: Wait for it to be running, get its public IP**

```bash
aws ec2 describe-instances --filters "Name=tag:Name,Values=chronos-gpu" "Name=instance-state-name,Values=running" --query "Reservations[].Instances[].PublicIpAddress" --output text --region us-east-1
```

Report the instance ID and public IP to the user before proceeding to Task 5.

---

### Task 5: Set up the GPU instance and run the validation

**⚠️ COORDINATOR EXECUTES THIS TASK DIRECTLY (SSH into a live, billing instance).**

- [ ] **Step 1: SSH in and clone the repo**

```bash
ssh -i <path-to-pem> ubuntu@<PUBLIC_IP> "git clone https://github.com/Aarav500/retail-forecasting-benchmark.git && cd retail-forecasting-benchmark && git checkout feature/chronos-wrapper"
```
(If the branch hasn't been pushed to GitHub yet, push it first from the local worktree: `git push -u origin feature/chronos-wrapper` — ask the user before pushing, per standing instructions on pushes.)

- [ ] **Step 2: Install dependencies on the instance**

```bash
ssh -i <path-to-pem> ubuntu@<PUBLIC_IP> "cd retail-forecasting-benchmark && pip install -e . && pip install -r requirements-gpu.txt"
```

- [ ] **Step 3: Verify torch sees the GPU**

```bash
ssh -i <path-to-pem> ubuntu@<PUBLIC_IP> "python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'"
```
Expected: `True <T4 device name>`. If `False`, stop and investigate the AMI/driver setup before spending more time — don't proceed to running the model on a CPU fallback silently.

- [ ] **Step 4: Run the Chronos-specific tests for real**

```bash
ssh -i <path-to-pem> ubuntu@<PUBLIC_IP> "cd retail-forecasting-benchmark && python3 -m pytest tests/test_chronos_model.py -v"
```
Expected: 2 passed (not skipped now — torch/chronos are installed).

- [ ] **Step 5: Run the validation script on one dataset first (smoke test)**

```bash
ssh -i <path-to-pem> ubuntu@<PUBLIC_IP> "cd retail-forecasting-benchmark && python3 experiments/scripts/run_chronos.py --size small --datasets dmart_food"
```
Expected: completes without error, prints ARIMA and Chronos-small RMSE for D-Mart Food, writes `experiments/results/foundation/dmart_food_chronos_small_results.json`.

- [ ] **Step 6: Run the full validation across all 35 datasets**

```bash
ssh -i <path-to-pem> ubuntu@<PUBLIC_IP> "cd retail-forecasting-benchmark && python3 experiments/scripts/run_chronos.py --size small"
```
This will take a while — Chronos does one forward pass per rolling step per dataset, so budget real time here too (order of magnitude comparable to or faster than LSTM's CPU cost, likely faster given the T4, but don't assume — watch it run).

- [ ] **Step 7: Copy results back to the local worktree**

```bash
scp -i <path-to-pem> -r ubuntu@<PUBLIC_IP>:~/retail-forecasting-benchmark/experiments/results/foundation ./experiments/results/foundation
```
(Run from `.worktrees/chronos-wrapper` locally.)

- [ ] **Step 8: Sanity-check at least one result**

Read `experiments/results/foundation/dmart_food_chronos_small_results.json` locally. Confirm the RMSE is a sensible number (not NaN, not wildly larger than ARIMA's ~255.76 by some absurd factor that would suggest a wiring bug) — a Chronos RMSE worse than ARIMA is a fine, expected, real result on a short low-SNR series; a Chronos RMSE that's NaN or off by many orders of magnitude means something is broken and needs investigating before treating any of this run's numbers as valid.

---

### Task 6: Stop the instance and commit results

**⚠️ COORDINATOR EXECUTES THIS TASK DIRECTLY.**

- [ ] **Step 1: Stop (not terminate) the instance**

```bash
aws ec2 stop-instances --instance-ids <INSTANCE_ID> --region us-east-1
```
Stop rather than terminate in case a quick re-run is needed — a stopped instance doesn't bill for compute (only negligible EBS storage). Confirm with the user this is what they want before running it, and separately confirm whether they'd rather terminate outright (no further use planned) or stop (might re-run soon).

- [ ] **Step 2: Verify it's actually stopped**

```bash
aws ec2 describe-instances --instance-ids <INSTANCE_ID> --query "Reservations[].Instances[].State.Name" --output text --region us-east-1
```
Expected: `stopped`.

- [ ] **Step 3: Commit the foundation-model results**

```bash
cd .worktrees/chronos-wrapper
git add experiments/results/foundation
git commit -m "feat: add Chronos-small validation results vs ARIMA across 35 datasets"
```

- [ ] **Step 4: Run the full local test suite one more time to confirm nothing's broken**

Run: `C:/venvs/shortseq/Scripts/python.exe -m pytest tests/ -v`
Expected: same pass/skip counts as Task 2 Step 5 (Chronos tests still SKIPPED here on Windows — that's correct, they only run for real on the now-stopped GPU instance).

---

### Task 7: Report and next steps

**⚠️ COORDINATOR EXECUTES THIS TASK DIRECTLY — this is a reporting step, not code.**

- [ ] **Step 1: Summarize to the user**: does Chronos-small beat ARIMA on D-Mart's low-SNR series, or not? What's the DM-test significance? This is the actual "does the wall exist for at least one foundation model" answer the whole NeurIPS story depends on — report it plainly, whichever way it comes out.

- [ ] **Step 2: Note this plan intentionally stops here** — it validates ONE model at ONE size against the EXISTING 35 datasets. Scaling to the other 7 foundation models, other Chronos sizes, and Phase B's new datasets are separate follow-on work, not part of this plan.

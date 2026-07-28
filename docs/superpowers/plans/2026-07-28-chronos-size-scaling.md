# Chronos Size-Scaling Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for Tasks 1-2 (local code, no billing). **Tasks 3-5 involve real AWS billing and MUST be executed by the coordinator directly, with explicit user confirmation before anything that starts billing — do NOT delegate those to an autonomous subagent.**

**Goal:** Measure RMSE for all five Chronos sizes (tiny 8M → large 710M) across all 368 registry series, against the committed ARIMA baseline, with length-stratified reporting — answering whether more parameters help below the short-series boundary.

**Architecture:** A new `run_size_scaling.py` loads ARIMA's RMSE once from existing committed results and varies only the Chronos size, writing one JSON per (dataset, size) incrementally so a long run is crash-safe and resumable. A GPU instance runs the sweep; results are copied back and committed.

**Tech Stack:** `chronos-forecasting`, PyTorch, AWS EC2 `g4dn.xlarge` (T4), existing `shortseq` package.

---

## Design reference

Full rationale: `docs/superpowers/specs/2026-07-28-chronos-size-scaling-design.md` (in this worktree). Read it before starting — in particular the "Expected finding" section, which pre-commits to reporting whatever the data shows.

## Environment facts (verified; use, don't re-derive)

- Local CPU venv: `C:/venvs/shortseq/Scripts/python.exe`. It has **no torch/chronos** — that's deliberate. Anything importing `ChronosForecaster` cannot run locally and must be tested via `pytest.importorskip` gating, exactly like `tests/test_chronos_model.py` does.
- The venv's editable `shortseq` install points at the MAIN repo checkout, not this worktree. To exercise worktree code, set `PYTHONPATH` to the worktree root:
  `cd <worktree> && PYTHONPATH="$(pwd)" C:/venvs/shortseq/Scripts/python.exe ...`
- Registry: `load_all()` returns **368 series**, takes ~30s (23s of which is the pre-existing slow `m4` loader — not worth optimizing here).
- Length split: **199 short (n<200) / 169 long (n>=200)**.
- Existing suite baseline: **98 passed, 1 skipped**.
- Measured Chronos throughput on the T4: **0.022 s/forward pass**; ~50,500 rolling steps per full pass over the registry.

## AWS facts (all still exist from the Chronos campaign)

- Region `us-east-1`, account `325527186655`.
- AMI `ami-0b9c598335204bf5b` (Deep Learning OSS Nvidia PyTorch 2.10, Ubuntu 24.04).
- Security group `sg-0056c10f41d9c88e6` (SSH from user IP only). **The user's IP may have changed since it was created** — re-check and re-authorize if SSH times out.
- Key pair `chronos-gpu-key`; private key at
  `C:/Users/aarav/AppData/Local/Temp/claude/C--Users-aarav-OneDrive-Desktop-Incomplete-papers/d9416108-212c-4a08-9866-547106382de8/scratchpad/chronos-gpu-key.pem`
- Public subnet `subnet-09ea79c58650980ee` (the default subnet `subnet-02324bf2df4e694e6` has **no internet gateway route** — do not use it, SSH will time out).
- Elastic IP `eipalloc-01c116ab335572014` (`100.49.214.171`).
- Stopped instance `i-0d99558093edd2c09` already has all dependencies installed — **starting it is preferable to launching fresh**, saves ~10 min of installs.
- On-demand G/VT vCPU quota is **4** — exactly one `g4dn.xlarge`. A second concurrent instance will fail with `VcpuLimitExceeded`.
- On the instance, the Python env is activated with `source /opt/pytorch/bin/activate` (Python 3.13, torch 2.10.0+cu130 preinstalled).

---

### Task 1: The size-scaling script

**Files:**
- Create: `experiments/scripts/run_size_scaling.py`

- [ ] **Step 1: Write the script**

```python
"""Chronos model-size scaling campaign.

Runs all five Chronos sizes (tiny 8M -> large 710M) across every
registry series and compares each to the ARIMA baseline already
committed under experiments/results/.

Deliberately does NOT re-run ARIMA: run_chronos.py re-fits ARIMA
alongside Chronos on every invocation, so sweeping five sizes through
it would fit ARIMA five times per series and could yield five slightly
different reference numbers, muddying the very comparison this campaign
exists to make. ARIMA's RMSE and predictions come from the committed
baseline results instead.

Only runs on a GPU instance (imports torch via ChronosForecaster); see
requirements-gpu.txt.
"""
import argparse
import json
from pathlib import Path

from shortseq.datasets.registry import load_all
from shortseq.evaluation.metrics import compute_metrics
from shortseq.models.foundation.chronos import ChronosForecaster

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "experiments" / "results"
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "scaling"

SIZES = ["tiny", "mini", "small", "base", "large"]
SHORT_MAX_N = 200  # series with n < SHORT_MAX_N are the "short" stratum


def load_arima_rmse(dataset_name: str) -> float | None:
    """ARIMA's RMSE for one dataset, from the committed baseline results.

    Returns None if the baseline file is missing or ARIMA failed there —
    the caller records that rather than silently treating it as a win.
    """
    path = BASELINE_DIR / f"{dataset_name}_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("metrics", {}).get("ARIMA", {}).get("rmse")


def result_path(dataset_name: str, size: str) -> Path:
    return RESULTS_DIR / f"{dataset_name}_chronos_{size}.json"


def run_one(dataset_name: str, dataset, size: str, split: float = 0.8) -> dict:
    """Run one (dataset, size) pair. Returns the result dict it wrote."""
    series = dataset.series
    split_idx = int(len(series) * split)
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]

    model_key = f"Chronos-{size}"
    metrics = {}
    failed_models = {}

    try:
        model = ChronosForecaster(size=size)
        model.fit(train)
        forecast = model.predict_rolling(test)
        preds = forecast.point[: len(test)]
        m = compute_metrics(
            test.values, preds, model.name, model.train_time_, model.pred_time_
        )
        metrics[model_key] = {k: v for k, v in m.items() if k != "residuals"}
        print(f"  {model_key}: RMSE={m['rmse']:.2f}")
    except Exception as exc:
        failed_models[model_key] = f"{type(exc).__name__}: {exc}"
        print(f"  [{dataset_name}/{model_key}] FAILED: {type(exc).__name__}: {exc}")

    arima_rmse = load_arima_rmse(dataset_name)
    output = {
        "dataset": dataset_name,
        "size": size,
        "n": dataset.n,
        "stratum": "short" if dataset.n < SHORT_MAX_N else "long",
        "n_train": split_idx,
        "n_test": len(test),
        "arima_rmse": arima_rmse,
        "metrics": metrics,
        "failed_models": failed_models,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(result_path(dataset_name, size), "w") as f:
        json.dump(output, f, indent=2)
    return output


def summarize() -> dict:
    """Stratified summary over whatever result files exist.

    Reports short (n < 200) and long (n >= 200) strata SEPARATELY and
    never emits a pooled figure — the registry is 199/169, so a pooled
    mean would average two regimes with opposite expected outcomes and
    hide the effect this campaign measures. See the Phase B spec's
    stratification addendum.
    """
    per_size = {}
    for path in sorted(RESULTS_DIR.glob("*_chronos_*.json")):
        with open(path) as f:
            d = json.load(f)
        size = d["size"]
        key = f"Chronos-{size}"
        rmse = d.get("metrics", {}).get(key, {}).get("rmse")
        arima = d.get("arima_rmse")
        bucket = per_size.setdefault(
            size, {"short": [], "long": [], "failed": 0, "no_baseline": 0}
        )
        if rmse is None:
            bucket["failed"] += 1
            continue
        if arima is None or arima == 0:
            bucket["no_baseline"] += 1
            continue
        bucket[d["stratum"]].append(rmse / arima)

    summary = {}
    for size in SIZES:
        b = per_size.get(size)
        if not b:
            continue
        entry = {"failed": b["failed"], "no_baseline": b["no_baseline"]}
        for stratum in ("short", "long"):
            ratios = b[stratum]
            if ratios:
                wins = sum(1 for r in ratios if r < 1.0)
                entry[stratum] = {
                    "n_series": len(ratios),
                    "mean_ratio": round(sum(ratios) / len(ratios), 4),
                    "median_ratio": round(sorted(ratios)[len(ratios) // 2], 4),
                    "chronos_wins": wins,
                    "arima_wins": len(ratios) - wins,
                }
            else:
                entry[stratum] = None
        summary[size] = entry
    return summary


def print_summary(summary: dict) -> None:
    print(f"\n{'=' * 70}")
    print("SIZE-SCALING SUMMARY (Chronos RMSE / ARIMA RMSE; <1.0 = Chronos better)")
    print("Strata reported separately by design — never pooled.")
    print(f"{'=' * 70}")
    for stratum in ("short", "long"):
        label = "SHORT (n<200)" if stratum == "short" else "LONG (n>=200)"
        print(f"\n{label}")
        print(f"{'size':<8}{'series':>8}{'mean':>10}{'median':>10}{'C wins':>9}{'A wins':>9}")
        for size in SIZES:
            s = summary.get(size, {}).get(stratum)
            if not s:
                continue
            print(
                f"{size:<8}{s['n_series']:>8}{s['mean_ratio']:>10.4f}"
                f"{s['median_ratio']:>10.4f}{s['chronos_wins']:>9}{s['arima_wins']:>9}"
            )
    print("\nfailures / missing-baseline per size:")
    for size in SIZES:
        e = summary.get(size)
        if e:
            print(f"  {size:<8} failed={e['failed']:<5} no_baseline={e['no_baseline']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=SIZES, choices=SIZES)
    parser.add_argument("--datasets", nargs="+", help="Only these dataset names")
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if a result file exists"
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="Just print the summary and exit"
    )
    args = parser.parse_args()

    if args.summary_only:
        print_summary(summarize())
        return

    datasets = load_all()
    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    total = len(datasets) * len(args.sizes)
    print(f"Size-scaling sweep: {len(datasets)} datasets x {len(args.sizes)} sizes = {total} runs")

    done = 0
    for size in args.sizes:
        for name, dataset in datasets.items():
            done += 1
            if not args.force and result_path(name, size).exists():
                print(f"[{done}/{total}] {name} @ {size}: cached, skipping")
                continue
            print(f"[{done}/{total}] {name} | n={dataset.n} | size={size}")
            run_one(name, dataset, size)

    print_summary(summarize())
    print(f"\nDone. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it's syntactically valid and fails only at the torch import**

Run:
```bash
cd "C:/Users/aarav/OneDrive/Desktop/Incomplete papers/retail-forecasting-benchmark/.worktrees/size-scaling"
PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" experiments/scripts/run_size_scaling.py --help
```
Expected: a `ModuleNotFoundError: No module named 'torch'` traceback whose LAST frame is `shortseq/models/foundation/chronos.py` at `import torch`. That specific failure at that specific line means the script itself is fine and only the (deliberately absent) GPU dependency is missing. Any OTHER error is a real bug — fix it.

- [ ] **Step 3: Commit**

```bash
git add experiments/scripts/run_size_scaling.py
git commit -m "feat: add Chronos size-scaling campaign script"
```

---

### Task 2: Structural tests for the summary logic

The GPU-dependent parts can't run locally, but `summarize()` and `load_arima_rmse()` are pure functions over JSON files — they can and should be tested on CPU. This is the safety net for the stratification requirement, which is the one thing in this campaign that would silently corrupt the finding if it broke.

**Files:**
- Create: `tests/test_size_scaling.py`

- [ ] **Step 1: Write the tests**

```python
"""Structural tests for experiments/scripts/run_size_scaling.py.

Covers the pure-Python summary/stratification logic on synthetic result
files. The GPU-dependent run_one() is not covered here — it needs torch
and a real model; it gets exercised on the GPU instance instead.

Stratification is the thing most worth pinning: the campaign's whole
point is that short and long series behave differently, so a summary
that silently pooled them would produce a confidently wrong finding.
"""
import json

import pytest

import experiments.scripts.run_size_scaling as rss


def _write_result(tmp_path, dataset, size, n, rmse, arima_rmse):
    """Write one synthetic result file in the schema run_one() emits."""
    payload = {
        "dataset": dataset,
        "size": size,
        "n": n,
        "stratum": "short" if n < rss.SHORT_MAX_N else "long",
        "n_train": int(n * 0.8),
        "n_test": n - int(n * 0.8),
        "arima_rmse": arima_rmse,
        "metrics": (
            {f"Chronos-{size}": {"rmse": rmse}} if rmse is not None else {}
        ),
        "failed_models": (
            {} if rmse is not None else {f"Chronos-{size}": "RuntimeError: CUDA OOM"}
        ),
    }
    path = tmp_path / f"{dataset}_chronos_{size}.json"
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def test_summarize_separates_short_and_long_strata(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "RESULTS_DIR", tmp_path)
    # short series: Chronos worse than ARIMA (ratio 1.5)
    _write_result(tmp_path, "short_a", "small", n=100, rmse=150.0, arima_rmse=100.0)
    _write_result(tmp_path, "short_b", "small", n=150, rmse=150.0, arima_rmse=100.0)
    # long series: Chronos better than ARIMA (ratio 0.5)
    _write_result(tmp_path, "long_a", "small", n=900, rmse=50.0, arima_rmse=100.0)

    summary = rss.summarize()

    assert summary["small"]["short"]["n_series"] == 2
    assert summary["small"]["short"]["mean_ratio"] == pytest.approx(1.5)
    assert summary["small"]["short"]["chronos_wins"] == 0
    assert summary["small"]["short"]["arima_wins"] == 2

    assert summary["small"]["long"]["n_series"] == 1
    assert summary["small"]["long"]["mean_ratio"] == pytest.approx(0.5)
    assert summary["small"]["long"]["chronos_wins"] == 1

    # The pooled mean would be 1.1667 — a number that describes neither
    # stratum. Assert it appears nowhere in the summary.
    assert "mean_ratio" not in summary["small"]
    assert "pooled" not in summary["small"]


def test_summarize_counts_failures_separately_from_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "RESULTS_DIR", tmp_path)
    _write_result(tmp_path, "ok", "large", n=100, rmse=90.0, arima_rmse=100.0)
    _write_result(tmp_path, "oom", "large", n=100, rmse=None, arima_rmse=100.0)

    summary = rss.summarize()

    # A failed (e.g. OOM) run must NOT be silently dropped, and must not
    # be counted as either a win or a loss.
    assert summary["large"]["failed"] == 1
    assert summary["large"]["short"]["n_series"] == 1
    assert summary["large"]["short"]["chronos_wins"] == 1


def test_summarize_counts_missing_baseline_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "RESULTS_DIR", tmp_path)
    _write_result(tmp_path, "nobase", "tiny", n=100, rmse=90.0, arima_rmse=None)

    summary = rss.summarize()

    # No ARIMA reference means no ratio can be computed; that's distinct
    # from both a win and a failure.
    assert summary["tiny"]["no_baseline"] == 1
    assert summary["tiny"]["short"] is None


def test_load_arima_rmse_returns_none_when_baseline_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "BASELINE_DIR", tmp_path)
    assert rss.load_arima_rmse("does_not_exist") is None


def test_load_arima_rmse_reads_committed_baseline_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(rss, "BASELINE_DIR", tmp_path)
    payload = {
        "dataset": "toy",
        "metrics": {"ARIMA": {"rmse": 123.45}, "LSTM": {"rmse": 200.0}},
        "dm_tests": {},
        "failed_models": {},
    }
    with open(tmp_path / "toy_results.json", "w") as f:
        json.dump(payload, f)
    assert rss.load_arima_rmse("toy") == pytest.approx(123.45)


def test_short_max_n_matches_phase_b_stratification_decision():
    # Pinned deliberately: the Phase B spec addendum fixes the boundary
    # at n < 200. Changing it silently would invalidate cross-campaign
    # comparisons.
    assert rss.SHORT_MAX_N == 200
```

- [ ] **Step 2: Run them**

Run:
```bash
cd "C:/Users/aarav/OneDrive/Desktop/Incomplete papers/retail-forecasting-benchmark/.worktrees/size-scaling"
PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" -m pytest tests/test_size_scaling.py -v
```
Expected: 6 passed.

Note: importing `run_size_scaling` imports `ChronosForecaster` → `torch`, which is absent locally. If collection fails with `ModuleNotFoundError: torch`, add `pytest.importorskip("torch")` before the `import experiments.scripts.run_size_scaling` line — but FIRST check whether it actually fails, because skipping these tests would defeat their purpose on the only machine that can run them. If they must be skipped locally, say so explicitly in the report so the coordinator knows to run them on the GPU box instead.

- [ ] **Step 3: Run the full suite for regressions**

Run: `PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" -m pytest tests/ -q`
Expected: 104 passed, 1 skipped (98+1 baseline, plus these 6).

- [ ] **Step 4: Commit**

```bash
git add tests/test_size_scaling.py
git commit -m "test: add structural tests for size-scaling summary and stratification"
```

---

### Task 3: Start the GPU instance

**⚠️ COORDINATOR EXECUTES DIRECTLY — real billing. Confirm with the user immediately before the start/launch call.**

- [ ] **Step 1: Confirm with the user in chat**, restating: starting `i-0d99558093edd2c09` (`g4dn.xlarge`, ~$0.526/hr), expected ~5 GPU-hours ≈ $3 for the full five-size sweep.

- [ ] **Step 2: Start the existing stopped instance** (it already has torch/chronos/shortseq installed)

```bash
aws ec2 start-instances --instance-ids i-0d99558093edd2c09 --region us-east-1
```

- [ ] **Step 3: Wait for running state**

```bash
until aws ec2 describe-instances --instance-ids i-0d99558093edd2c09 --query 'Reservations[].Instances[].State.Name' --output text --region us-east-1 2>/dev/null | grep -q running; do sleep 5; done; echo running
```

- [ ] **Step 4: Get its public IP and re-check SSH access**

```bash
aws ec2 describe-instances --instance-ids i-0d99558093edd2c09 --query "Reservations[].Instances[].PublicIpAddress" --output text --region us-east-1
```
A stopped instance loses its auto-assigned public IP; the Elastic IP `eipalloc-01c116ab335572014` may need re-associating:
```bash
aws ec2 associate-address --instance-id i-0d99558093edd2c09 --allocation-id eipalloc-01c116ab335572014 --region us-east-1
```

Also re-check the security group still permits the user's CURRENT IP (it may have changed since the group was created):
```bash
curl -s https://checkip.amazonaws.com
aws ec2 describe-security-groups --group-ids sg-0056c10f41d9c88e6 --query "SecurityGroups[].IpPermissions" --region us-east-1
```
If the current IP isn't authorized:
```bash
aws ec2 authorize-security-group-ingress --group-id sg-0056c10f41d9c88e6 --protocol tcp --port 22 --cidr <CURRENT_IP>/32 --region us-east-1
```

- [ ] **Step 5: Confirm SSH and GPU**

```bash
ssh -i "C:/Users/aarav/AppData/Local/Temp/claude/C--Users-aarav-OneDrive-Desktop-Incomplete-papers/d9416108-212c-4a08-9866-547106382de8/scratchpad/chronos-gpu-key.pem" -o ConnectTimeout=15 ubuntu@<IP> "nvidia-smi --query-gpu=name,memory.total --format=csv"
```
Expected: `Tesla T4, 15360 MiB`.

---

### Task 4: Run the sweep

**⚠️ COORDINATOR EXECUTES DIRECTLY.**

- [ ] **Step 1: Sync the branch to the instance**

The instance has a clone at `~/retail-forecasting-benchmark` on branch `feature/chronos-wrapper`. This campaign's code is on `feature/size-scaling`, which is local-only. Push it first (ASK THE USER before pushing — pushes are outward-facing):

```bash
cd "C:/Users/aarav/OneDrive/Desktop/Incomplete papers/retail-forecasting-benchmark/.worktrees/size-scaling"
git push -u origin feature/size-scaling
```
Then on the instance:
```bash
ssh -i <key> ubuntu@<IP> "cd retail-forecasting-benchmark && git fetch origin && git checkout feature/size-scaling && git pull"
```

The instance's clone predates Phase B, so it lacks the 333 new derived-data CSVs. `git pull` brings them (they're committed). Verify the registry sees all 368:
```bash
ssh -i <key> ubuntu@<IP> "source /opt/pytorch/bin/activate && cd retail-forecasting-benchmark && python -c 'from shortseq.datasets.registry import load_all; print(len(load_all()))'"
```
Expected: `368`. If it prints 35, the pull didn't bring the derived data — stop and investigate rather than running a sweep over the wrong registry.

- [ ] **Step 2: Smoke-test one dataset across all five sizes**

```bash
ssh -i <key> ubuntu@<IP> "source /opt/pytorch/bin/activate && cd retail-forecasting-benchmark && python experiments/scripts/run_size_scaling.py --datasets dmart_food"
```
Expected: five RMSE lines (tiny/mini/small/base/large) and five JSONs in `experiments/results/scaling/`. This is also the first real test of whether `base` and `large` fit in the T4's 16GB — if either OOMs, it will be printed as a FAILED line rather than crashing. Report which sizes (if any) OOM before proceeding.

- [ ] **Step 3: Run the full sweep in the background**

This is the long one (~5 GPU-hours). Run it detached with output to a log so a dropped SSH connection doesn't kill it:

```bash
ssh -i <key> ubuntu@<IP> "source /opt/pytorch/bin/activate && cd retail-forecasting-benchmark && nohup python experiments/scripts/run_size_scaling.py > scaling_run.log 2>&1 &"
```
Then poll periodically:
```bash
ssh -i <key> ubuntu@<IP> "tail -5 retail-forecasting-benchmark/scaling_run.log; ls retail-forecasting-benchmark/experiments/results/scaling | wc -l"
```
Expected final count: 1840 files (368 x 5), minus none — failures still write a file (with `failed_models` populated), so a short count means the run didn't finish.

**Do not assume completion from silence.** The script prints a `[n/total]` progress line per run and a summary block at the end; confirm the summary block is present in the log before treating the sweep as done.

- [ ] **Step 4: Print the stratified summary**

```bash
ssh -i <key> ubuntu@<IP> "source /opt/pytorch/bin/activate && cd retail-forecasting-benchmark && python experiments/scripts/run_size_scaling.py --summary-only"
```

- [ ] **Step 5: Copy results back**

```bash
cd "C:/Users/aarav/OneDrive/Desktop/Incomplete papers/retail-forecasting-benchmark/.worktrees/size-scaling"
scp -i <key> -r ubuntu@<IP>:~/retail-forecasting-benchmark/experiments/results/scaling ./experiments/results/scaling
ls experiments/results/scaling | wc -l
```

---

### Task 5: Stop the instance, commit, report

**⚠️ COORDINATOR EXECUTES DIRECTLY.**

- [ ] **Step 1: Stop the instance immediately once results are copied** (billing stops; the EBS volume persists for a future campaign)

```bash
aws ec2 stop-instances --instance-ids i-0d99558093edd2c09 --region us-east-1
until aws ec2 describe-instances --instance-ids i-0d99558093edd2c09 --query 'Reservations[].Instances[].State.Name' --output text --region us-east-1 2>/dev/null | grep -q stopped; do sleep 5; done; echo stopped
```

- [ ] **Step 2: Commit the results**

```bash
git add experiments/results/scaling
git commit -m "feat: add Chronos size-scaling results (5 sizes x 368 series)"
```

- [ ] **Step 3: Verify the local suite still passes**

Run: `PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" -m pytest tests/ -q`
Expected: 104 passed, 1 skipped.

- [ ] **Step 4: Report the finding**

Report the stratified table plainly. The spec pre-commits to this: the thesis predicts flat/noisy scaling on the short stratum and a clearer downward trend on the long stratum, **but if the data disagrees — e.g. large clearly beating small on short series — that is the finding and it gets reported as such.** State per-stratum mean/median ratios and win counts per size, plus any OOM failures, without smoothing over an inconvenient result.

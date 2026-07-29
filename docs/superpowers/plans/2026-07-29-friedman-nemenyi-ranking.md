# Friedman/Nemenyi Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. All tasks are local, CPU-only — no GPU, no AWS, no billing.

**Goal:** Determine whether the Chronos size-scaling ordering is statistically real, per length stratum, using Friedman + Nemenyi on the already-collected 368×5 RMSE grid.

**Architecture:** Implement `friedman_nemenyi()` in the existing `shortseq/evaluation/ranking.py` stub, using `scipy.stats.friedmanchisquare` for the omnibus test and `scipy.stats.studentized_range` to compute the Nemenyi critical difference at runtime. An analysis script applies it separately to the short and long strata and writes JSON reports. Separately, fix the residual-stripping that foreclosed DM testing.

**Tech Stack:** scipy 1.17.1 (already installed; `friedmanchisquare` and `studentized_range` both verified available), pandas, existing `shortseq` package.

---

## Design reference

Full rationale: `docs/superpowers/specs/2026-07-29-friedman-nemenyi-ranking-design.md` in this worktree. Read it — especially "Why the rest of Phase D is deferred" and the pre-registered expected finding.

## IMPORTANT deviation from the spec (deliberate, verified)

The spec proposed **hardcoding a q_alpha table** for k=2..10. **Do not do that.** A three-way independent verification of the Nemenyi formula concluded that computing q at runtime is strictly better: it removes transcription risk and generalises past k=10. Use:

```python
q = studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0)
```

The `/sqrt(2)` is the single most common source of error in Nemenyi implementations, and it pairs with the `6*N` denominator. Getting it wrong inflates CD by ~1.414× and silently produces wrong significance conclusions.

## Verified constants (independently reproduced on this machine — use as test fixtures)

| quantity | value |
|---|---|
| `q(alpha=0.05, k=5)` | 2.727774 |
| `CD(k=5, N=199, alpha=0.05)` | **0.4323813055932977** |
| `CD(k=5, N=199, alpha=0.10)` | 0.3898595 |
| value if `/sqrt(2)` is wrongly omitted | **0.611480** ← the bug to guard against |
| `q(0.05, k=2)` | 1.959963985 == `norm.ppf(0.975)` exactly |

## Environment facts (verified — use, don't re-derive)

- Venv: `C:/venvs/shortseq/Scripts/python.exe`. scipy 1.17.1 present.
- The venv's editable `shortseq` points at the MAIN checkout, not this worktree. Always run with PYTHONPATH set:
  `cd <worktree> && PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" ...`
- Existing suite baseline: **104 passed, 1 skipped**.
- Scaling data: `experiments/results/scaling/{dataset}_chronos_{size}.json`, 1840 files. Each has `dataset`, `size`, `n`, `stratum` ("short"/"long"), `arima_rmse`, `metrics` (key `Chronos-{size}` → dict with `rmse`), `failed_models`.
- Strata: **199 short (n<200), 169 long (n>=200)**. Zero failures, zero missing baselines.

---

### Task 1: `friedman_nemenyi()` implementation

**Files:**
- Modify: `shortseq/evaluation/ranking.py` (replaces the Task 11 stub)
- Test: `tests/test_ranking.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_ranking.py`:
```python
"""Tests for Friedman + Nemenyi ranking.

The Nemenyi critical difference is the one piece of genuinely new
statistics here, and a wrong constant would silently produce confident
but wrong significance conclusions. The constants below were verified by
three independent derivations and reproduced directly with scipy:
    CD(k=5, N=199, alpha=0.05) = 0.4323813055932977
The classic failure mode is omitting the /sqrt(2) when converting the
studentized range, which inflates CD to 0.611480 -- there is an explicit
named guard against exactly that.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from shortseq.evaluation.ranking import friedman_nemenyi, nemenyi_critical_difference


def test_cd_matches_independently_verified_value():
    assert nemenyi_critical_difference(k=5, n_blocks=199, alpha=0.05) == pytest.approx(
        0.4323813055932977, abs=1e-9
    )


def test_cd_at_alpha_010():
    assert nemenyi_critical_difference(k=5, n_blocks=199, alpha=0.10) == pytest.approx(
        0.3898595, abs=1e-6
    )


def test_cd_is_not_off_by_sqrt2():
    # Named guard against the single most common Nemenyi bug: pairing the
    # RAW studentized range with the 6N denominator. That yields 0.611480.
    cd = nemenyi_critical_difference(k=5, n_blocks=199, alpha=0.05)
    assert abs(cd - 0.611480) > 0.1


def test_k2_collapses_to_normal_quantile():
    # Decisive convention check: at k=2 Nemenyi must reduce to a two-sided
    # normal test. Under the raw-q convention this fails.
    cd = nemenyi_critical_difference(k=2, n_blocks=100, alpha=0.05)
    expected = norm.ppf(0.975) * np.sqrt(2 * 3 / (6 * 100))
    assert cd == pytest.approx(expected, abs=1e-12)


def test_cd_shrinks_as_series_count_grows():
    # More blocks -> tighter critical difference.
    assert nemenyi_critical_difference(5, 1000, 0.05) < nemenyi_critical_difference(5, 199, 0.05)


def test_cd_grows_with_more_models():
    assert nemenyi_critical_difference(8, 199, 0.05) > nemenyi_critical_difference(3, 199, 0.05)


def test_friedman_detects_a_uniformly_better_model():
    # model_a beats the others on every single series -> must be significant.
    rows = []
    for i in range(30):
        rows.append({"model_a": 1.0 + i * 0.01, "model_b": 5.0 + i * 0.01, "model_c": 9.0 + i * 0.01})
    result = friedman_nemenyi(pd.DataFrame(rows))
    assert result["friedman_p_value"] < 0.001
    assert result["friedman_significant"] is True
    # rank 1 = best (lowest error)
    assert result["mean_ranks"]["model_a"] == pytest.approx(1.0)
    assert result["mean_ranks"]["model_c"] == pytest.approx(3.0)


def test_friedman_does_not_fire_on_identical_models():
    # All models identical on every series -> all ties, no real difference.
    rows = [{"m1": 5.0, "m2": 5.0, "m3": 5.0} for _ in range(30)]
    result = friedman_nemenyi(pd.DataFrame(rows))
    assert result["friedman_significant"] is False


def test_mean_ranks_sum_to_k_times_k_plus_1_over_2():
    # Structural invariant of within-series ranking. Catches ranking across
    # the wrong axis, which is an easy and silent mistake.
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.normal(size=(50, 4)), columns=["a", "b", "c", "d"])
    result = friedman_nemenyi(df)
    k = 4
    assert sum(result["mean_ranks"].values()) == pytest.approx(k * (k + 1) / 2)
    for r in result["mean_ranks"].values():
        assert 1.0 <= r <= k


def test_ties_get_average_ranks():
    # Two models tied for best on every series -> both get mean rank 1.5.
    rows = [{"a": 1.0, "b": 1.0, "c": 2.0} for _ in range(20)]
    result = friedman_nemenyi(rows_to_df(rows))
    assert result["mean_ranks"]["a"] == pytest.approx(1.5)
    assert result["mean_ranks"]["b"] == pytest.approx(1.5)
    assert result["mean_ranks"]["c"] == pytest.approx(3.0)


def rows_to_df(rows):
    return pd.DataFrame(rows)


def test_pairwise_flags_significance_against_cd():
    rows = []
    for i in range(40):
        rows.append({"best": 1.0 + i * 0.01, "mid": 5.0 + i * 0.01, "worst": 9.0 + i * 0.01})
    result = friedman_nemenyi(pd.DataFrame(rows))
    pair = result["pairwise"]["best_vs_worst"]
    assert pair["rank_difference"] == pytest.approx(2.0)
    assert pair["significant"] is True


def test_incomplete_grid_is_rejected():
    # The CD formula assumes every model is scored on every series. A NaN
    # means a missing cell, which would silently invalidate the result.
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, np.nan]})
    with pytest.raises(ValueError, match="complete"):
        friedman_nemenyi(df)


def test_requires_at_least_three_models_for_friedman():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="at least 3"):
        friedman_nemenyi(df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
cd "C:/Users/aarav/OneDrive/Desktop/Incomplete papers/retail-forecasting-benchmark/.worktrees/ranking"
PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" -m pytest tests/test_ranking.py -v
```
Expected: `ImportError: cannot import name 'nemenyi_critical_difference'` (the stub has only `friedman_nemenyi`, which raises `NotImplementedError`).

- [ ] **Step 3: Write `shortseq/evaluation/ranking.py`** (replaces the stub entirely)

```python
"""Statistical ranking across many series - Friedman test + Nemenyi post-hoc.

Ranks models within each series (so series whose RMSE differ by orders of
magnitude - n=53 UCI vs n=1684 Favorita - contribute equally), runs the
Friedman omnibus test on those ranks, and computes the Nemenyi critical
difference for pairwise comparisons.

The Nemenyi critical difference is

    CD = q_alpha * sqrt( k * (k + 1) / (6 * N) )

where k is the number of models, N the number of series (blocks), and
q_alpha is the studentized range statistic at infinite degrees of freedom
DIVIDED BY sqrt(2). That division is not optional: pairing the raw
studentized range with the 6*N denominator inflates CD by ~1.414x and
silently produces wrong significance conclusions. `q_alpha` is computed at
runtime rather than read from a hardcoded table, which removes
transcription risk and generalises beyond the usual k<=10 tables.

Reference: Demsar (2006), "Statistical Comparisons of Classifiers over
Multiple Data Sets", JMLR 7:1-30.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range


def nemenyi_critical_difference(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for average ranks.

    k: number of models being compared.
    n_blocks: number of series (blocks), NOT the number of observations.
    """
    if k < 2:
        raise ValueError(f"need at least 2 models, got {k}")
    if n_blocks < 1:
        raise ValueError(f"need at least 1 block, got {n_blocks}")
    q = studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def friedman_nemenyi(results: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Rank models across series and test whether the ordering is real.

    results: rows = series, columns = models, values = a LOWER-IS-BETTER
    metric (e.g. RMSE). Every model must be scored on every series.

    Returns a dict with mean ranks (rank 1 = best), the Friedman statistic
    and p-value, the Nemenyi critical difference, and pairwise comparisons.

    If the Friedman omnibus test is not significant, pairwise results are
    still returned but `friedman_significant` is False - post-hoc
    comparisons after a non-significant omnibus test are not meaningful,
    and the flag makes that impossible to overlook.
    """
    if results.isna().any().any():
        raise ValueError(
            "results grid must be complete - every model scored on every "
            "series. Missing cells invalidate the Nemenyi critical difference."
        )
    k = results.shape[1]
    n = results.shape[0]
    if k < 3:
        raise ValueError(f"Friedman needs at least 3 models, got {k}")

    # Rank WITHIN each series, ascending so rank 1 = lowest error = best.
    # Ties get average ranks.
    ranks = np.vstack([rankdata(row) for row in results.to_numpy()])
    mean_ranks = {
        model: float(ranks[:, i].mean()) for i, model in enumerate(results.columns)
    }

    stat, p_value = friedmanchisquare(*[results[c].to_numpy() for c in results.columns])
    significant = bool(p_value < alpha)

    cd = nemenyi_critical_difference(k, n, alpha)

    pairwise = {}
    for a, b in itertools.combinations(results.columns, 2):
        diff = abs(mean_ranks[a] - mean_ranks[b])
        pairwise[f"{a}_vs_{b}"] = {
            "rank_difference": round(diff, 6),
            "significant": bool(diff >= cd),
        }

    return {
        "n_series": int(n),
        "n_models": int(k),
        "alpha": alpha,
        "mean_ranks": mean_ranks,
        "friedman_statistic": float(stat),
        "friedman_p_value": float(p_value),
        "friedman_significant": significant,
        "critical_difference": cd,
        "pairwise": pairwise,
    }
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" -m pytest tests/test_ranking.py -v`
Expected: 13 passed.

Note: `test_friedman_does_not_fire_on_identical_models` uses all-identical values. `scipy.stats.friedmanchisquare` may emit a warning or return `nan` for a fully-degenerate input; if `p_value` comes back `nan`, treat it as not significant (`significant = bool(p_value < alpha)` already evaluates `nan < alpha` to `False`, which is the behaviour we want). If scipy instead raises, catch it in `friedman_nemenyi` and return `friedman_statistic: float("nan"), friedman_p_value: float("nan"), friedman_significant: False`, and note that in the docstring.

- [ ] **Step 5: Run the full suite for regressions**

Run: `PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" -m pytest tests/ -q`
Expected: 117 passed, 1 skipped (104 + 13 new).

Note `tests/test_analysis_stubs.py` has `test_friedman_nemenyi_not_yet_implemented`, which asserts the stub raises `NotImplementedError`. That test is now false. **Delete that one test** (keep `test_fit_boundary_not_yet_implemented` — boundary analysis is still a stub). Expected count becomes 116 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add shortseq/evaluation/ranking.py tests/test_ranking.py tests/test_analysis_stubs.py
git commit -m "feat: implement Friedman/Nemenyi ranking with runtime-computed critical difference"
```

---

### Task 2: Ranking analysis script

**Files:**
- Create: `experiments/scripts/run_ranking.py`
- Test: `tests/test_run_ranking.py`

- [ ] **Step 1: Write the failing test**

`tests/test_run_ranking.py`:
```python
"""Tests for the ranking analysis script.

Focus is the stratification guarantee: short (n<200) and long (n>=200)
series must be ranked separately and never pooled. The two strata have
opposite-signed effects at the small end (tiny is 0.972 short vs 1.092
long), so a pooled ranking would average away the structure being
measured.
"""
import json

import pytest

import experiments.scripts.run_ranking as rr


def _write_scaling(tmp_path, dataset, size, n, rmse, arima_rmse=100.0):
    payload = {
        "dataset": dataset,
        "size": size,
        "n": n,
        "stratum": "short" if n < 200 else "long",
        "n_train": int(n * 0.8),
        "n_test": n - int(n * 0.8),
        "arima_rmse": arima_rmse,
        "metrics": {f"Chronos-{size}": {"rmse": rmse}},
        "failed_models": {},
    }
    with open(tmp_path / f"{dataset}_chronos_{size}.json", "w") as f:
        json.dump(payload, f)


def test_build_matrices_separates_strata(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for size, rmse in [("tiny", 90.0), ("small", 80.0), ("large", 70.0)]:
        _write_scaling(tmp_path, "s1", size, n=100, rmse=rmse)
        _write_scaling(tmp_path, "s2", size, n=150, rmse=rmse)
        _write_scaling(tmp_path, "L1", size, n=900, rmse=rmse)

    matrices = rr.build_matrices()

    assert set(matrices) == {"short", "long"}
    assert list(matrices["short"].index) == ["s1", "s2"]
    assert list(matrices["long"].index) == ["L1"]
    assert "pooled" not in matrices


def test_build_matrices_columns_are_sizes_in_scaling_order(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for size in ["large", "tiny", "small"]:  # deliberately unsorted on disk
        _write_scaling(tmp_path, "s1", size, n=100, rmse=50.0)
    matrices = rr.build_matrices()
    # Columns must follow parameter-count order, not filesystem order,
    # so the report reads tiny -> large.
    assert list(matrices["short"].columns) == ["tiny", "small", "large"]


def test_build_matrices_drops_incomplete_series(tmp_path, monkeypatch):
    # A series missing one size would make the grid incomplete, which the
    # Nemenyi CD formula does not tolerate. It must be dropped loudly,
    # not silently NaN-filled.
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for size in ["tiny", "small", "large"]:
        _write_scaling(tmp_path, "complete", size, n=100, rmse=50.0)
    _write_scaling(tmp_path, "partial", "tiny", n=100, rmse=50.0)

    matrices = rr.build_matrices()

    assert list(matrices["short"].index) == ["complete"]
    assert not matrices["short"].isna().any().any()


def test_analyse_stratum_returns_ranking_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "SCALING_DIR", tmp_path)
    for i in range(15):
        _write_scaling(tmp_path, f"s{i}", "tiny", n=100, rmse=90.0 + i)
        _write_scaling(tmp_path, f"s{i}", "small", n=100, rmse=80.0 + i)
        _write_scaling(tmp_path, f"s{i}", "large", n=100, rmse=70.0 + i)

    matrices = rr.build_matrices()
    payload = rr.analyse(matrices["short"])

    assert payload["n_series"] == 15
    assert payload["n_models"] == 3
    assert payload["mean_ranks"]["large"] < payload["mean_ranks"]["tiny"]
    assert "critical_difference" in payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" -m pytest tests/test_run_ranking.py -v`
Expected: `ModuleNotFoundError: No module named 'experiments.scripts.run_ranking'`

- [ ] **Step 3: Write `experiments/scripts/run_ranking.py`**

```python
"""Rank the five Chronos sizes per length stratum, using Friedman + Nemenyi.

Consumes the committed size-scaling results and answers two questions the
raw RMSE ratios cannot:

  1. Do the five sizes differ significantly, within each stratum?
  2. Is the `base` (200M) anomaly on the long stratum - where it is WORSE
     than `small`, breaking the otherwise monotonic trend - a real effect
     or noise?

Stratification is mandatory, not cosmetic: short (n<200) and long
(n>=200) are ranked separately and a pooled ranking is never produced.
The strata have opposite-signed effects at the small end, so pooling
would average away the structure being measured. See the Phase B spec's
stratification addendum.

CPU-only; consumes committed data, runs no models.
"""
import json
from pathlib import Path

import pandas as pd

from shortseq.evaluation.ranking import friedman_nemenyi

REPO_ROOT = Path(__file__).resolve().parents[2]
SCALING_DIR = REPO_ROOT / "experiments" / "results" / "scaling"
OUT_DIR = REPO_ROOT / "experiments" / "results" / "ranking"

# Parameter-count order, so reports read smallest -> largest.
SIZE_ORDER = ["tiny", "mini", "small", "base", "large"]


def build_matrices() -> dict:
    """One RMSE matrix per stratum: rows = series, columns = sizes.

    Series missing any size are dropped (loudly) rather than NaN-filled -
    the Nemenyi critical difference assumes a complete grid.
    """
    records = []
    for path in sorted(SCALING_DIR.glob("*_chronos_*.json")):
        with open(path) as f:
            d = json.load(f)
        rmse = d.get("metrics", {}).get(f"Chronos-{d['size']}", {}).get("rmse")
        if rmse is None:
            continue
        records.append(
            {
                "dataset": d["dataset"],
                "size": d["size"],
                "stratum": d["stratum"],
                "rmse": rmse,
            }
        )

    df = pd.DataFrame(records)
    matrices = {}
    for stratum in ("short", "long"):
        sub = df[df["stratum"] == stratum]
        if sub.empty:
            continue
        wide = sub.pivot(index="dataset", columns="size", values="rmse")
        cols = [s for s in SIZE_ORDER if s in wide.columns]
        wide = wide[cols]
        before = len(wide)
        wide = wide.dropna()
        dropped = before - len(wide)
        if dropped:
            print(f"  [{stratum}] dropped {dropped} series with incomplete size coverage")
        matrices[stratum] = wide
    return matrices


def analyse(matrix: pd.DataFrame, alpha: float = 0.05) -> dict:
    return friedman_nemenyi(matrix, alpha=alpha)


def print_report(stratum: str, payload: dict) -> None:
    label = "SHORT (n<200)" if stratum == "short" else "LONG (n>=200)"
    print(f"\n{'=' * 66}")
    print(f"{label}  |  {payload['n_series']} series x {payload['n_models']} sizes")
    print(f"{'=' * 66}")
    print(f"Friedman chi2 = {payload['friedman_statistic']:.4f}, "
          f"p = {payload['friedman_p_value']:.3e}, "
          f"significant = {payload['friedman_significant']}")
    print(f"Nemenyi critical difference (alpha={payload['alpha']}) = "
          f"{payload['critical_difference']:.4f}")

    print("\nmean ranks (1 = best):")
    for size in SIZE_ORDER:
        if size in payload["mean_ranks"]:
            print(f"  {size:<8}{payload['mean_ranks'][size]:.4f}")

    if not payload["friedman_significant"]:
        print("\n  Friedman NOT significant - pairwise results below are not "
              "meaningful and must not be reported as findings.")

    print("\npairwise (significant = mean-rank gap >= critical difference):")
    for pair, info in payload["pairwise"].items():
        mark = "YES" if info["significant"] else "no "
        print(f"  {pair:<28} diff={info['rank_difference']:.4f}  sig={mark}")


def main():
    matrices = build_matrices()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for stratum, matrix in matrices.items():
        payload = analyse(matrix)
        print_report(stratum, payload)
        with open(OUT_DIR / f"chronos_sizes_{stratum}.json", "w") as f:
            json.dump(payload, f, indent=2)

    print(f"\nWrote per-stratum ranking to {OUT_DIR}")
    print("NOTE: no pooled cross-stratum ranking is produced, by design.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" -m pytest tests/test_run_ranking.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run it for real against the committed scaling data**

Run: `PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" experiments/scripts/run_ranking.py`

Expected: two report blocks (SHORT with 199 series, LONG with 169), each with a Friedman result, a critical difference, five mean ranks, and 10 pairwise comparisons. Two JSON files written under `experiments/results/ranking/`.

Sanity checks before moving on:
- SHORT n_series == 199 and LONG n_series == 169. If either differs, series were dropped for incomplete coverage — investigate rather than proceeding.
- `critical_difference` for the short stratum should be ≈ **0.4324** (k=5, N=199, alpha=0.05). If it reads ≈0.611, the `/sqrt(2)` is missing — stop and fix.
- Mean ranks should sum to 15.0 (k(k+1)/2 for k=5).

- [ ] **Step 6: Commit**

```bash
git add experiments/scripts/run_ranking.py tests/test_run_ranking.py experiments/results/ranking
git commit -m "feat: add per-stratum Friedman/Nemenyi ranking of Chronos sizes"
```

---

### Task 3: Fix residual stripping (unblocks future DM tests)

Both result-writing scripts strip `residuals` before writing, which is why DM tests were impossible on the size-scaling data. Add an opt-in flag so a future run can retain them. Opt-in rather than always-on: residuals are ~337 floats per model per series, which would substantially inflate 1,840 files for a capability not always needed.

**Files:**
- Modify: `experiments/scripts/run_size_scaling.py`
- Modify: `experiments/scripts/run_baselines.py`
- Test: `tests/test_keep_residuals.py`

- [ ] **Step 1: Write the failing test**

`tests/test_keep_residuals.py`:
```python
"""Residuals must be retainable on request.

Per-timestep residuals were stripped unconditionally at write time, which
foreclosed Diebold-Mariano testing on the size-scaling campaign: a DM test
compares two models' error series point-by-point, and RMSE alone cannot
reconstruct that. These tests pin the opt-in behaviour so the same dead
end is not hit again.
"""
import inspect

import experiments.scripts.run_baselines as rb
import experiments.scripts.run_size_scaling as rss


def test_size_scaling_run_one_accepts_keep_residuals():
    sig = inspect.signature(rss.run_one)
    assert "keep_residuals" in sig.parameters
    assert sig.parameters["keep_residuals"].default is False


def test_baselines_run_dataset_accepts_keep_residuals():
    sig = inspect.signature(rb.run_dataset)
    assert "keep_residuals" in sig.parameters
    assert sig.parameters["keep_residuals"].default is False


def test_size_scaling_cli_exposes_keep_residuals_flag():
    src = inspect.getsource(rss.main)
    assert "--keep-residuals" in src


def test_baselines_cli_exposes_keep_residuals_flag():
    src = inspect.getsource(rb.main)
    assert "--keep-residuals" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" -m pytest tests/test_keep_residuals.py -v`
Expected: 4 failed — `keep_residuals` is not a parameter of either function.

- [ ] **Step 3: Modify `experiments/scripts/run_size_scaling.py`**

Change the `run_one` signature to accept the flag:
```python
def run_one(dataset_name: str, dataset, size: str, split: float = 0.8,
            keep_residuals: bool = False) -> dict:
```

Replace the metrics-stripping line (currently
`metrics[model_key] = {k: v for k, v in m.items() if k != "residuals"}`) with:
```python
        metrics[model_key] = (
            dict(m) if keep_residuals
            else {k: v for k, v in m.items() if k != "residuals"}
        )
```

In `main()`, add the argument and thread it through:
```python
    parser.add_argument(
        "--keep-residuals", action="store_true",
        help="Retain per-timestep residuals in the output JSON. Needed for "
             "Diebold-Mariano tests; off by default because residuals "
             "substantially inflate result files.",
    )
```
and change the call site `run_one(name, dataset, size)` to
`run_one(name, dataset, size, keep_residuals=args.keep_residuals)`.

- [ ] **Step 4: Modify `experiments/scripts/run_baselines.py`**

Change the `run_dataset` signature:
```python
def run_dataset(name: str, dataset, config: dict, split: float = None,
                keep_residuals: bool = False) -> dict:
```
(Keep whatever the current `split` default is — do not change it.)

Replace the metrics line in the `output` dict (currently
`"metrics": {k: {mk: mv for mk, mv in v.items() if mk != "residuals"} for k, v in results.items()},`)
with:
```python
        "metrics": {
            k: (dict(v) if keep_residuals
                else {mk: mv for mk, mv in v.items() if mk != "residuals"})
            for k, v in results.items()
        },
```

In `main()`, add the same argument:
```python
    parser.add_argument(
        "--keep-residuals", action="store_true",
        help="Retain per-timestep residuals in the output JSON. Needed for "
             "Diebold-Mariano tests; off by default because residuals "
             "substantially inflate result files.",
    )
```
and thread `keep_residuals=args.keep_residuals` into the `run_dataset(...)` call.

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" -m pytest tests/test_keep_residuals.py -v`
Expected: 4 passed.

- [ ] **Step 6: Verify default behaviour is genuinely unchanged**

This is the important regression check — the flag must not alter existing output.

Run:
```bash
PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" -m pytest tests/test_experiment_scripts.py -v
```
Expected: all pass unchanged (they assert `metrics` has no leaked `residuals` key by default).

- [ ] **Step 7: Run the full suite**

Run: `PYTHONPATH="$(pwd)" "C:/venvs/shortseq/Scripts/python.exe" -m pytest tests/ -q`
Expected: 124 passed, 1 skipped (104 baseline − 1 deleted stub test + 13 + 4 + 4 = 124).

- [ ] **Step 8: Commit**

```bash
git add experiments/scripts/run_size_scaling.py experiments/scripts/run_baselines.py tests/test_keep_residuals.py
git commit -m "fix: add opt-in --keep-residuals so future runs can support DM tests"
```

---

### Task 4: Report the finding

**Files:** none (reporting step)

- [ ] **Step 1: Read the two ranking JSONs** at `experiments/results/ranking/chronos_sizes_short.json` and `..._long.json`.

- [ ] **Step 2: Report honestly, against the pre-registered expectation**

The spec pre-registered this: the omnibus Friedman test is very likely significant in both strata (given win counts moving 109→141 short and 55→132 long). The genuinely uncertain parts are:

  a. **Which adjacent size pairs separate under Nemenyi.** Adjacent sizes are close, so plausibly few do. **If adjacent pairs mostly do NOT separate, "scaling helps monotonically" must be reported as the weaker "the extremes differ but the individual steps do not."** That is a materially different claim and must not be smoothed over.

  b. **Whether `base` and `small` differ on the long stratum.** If their rank gap is below the critical difference, the monotonicity break is noise and should be reported as such rather than treated as a finding.

Report per stratum: Friedman statistic and p-value, critical difference, the five mean ranks, which pairs separated, and explicitly the tiny-vs-large and base-vs-small verdicts.

- [ ] **Step 3: Note what remains unanswered**

State plainly that this establishes whether the *ordering among Chronos sizes* is real, and does **not** establish per-series significance of Chronos vs ARIMA — that needs DM tests, which need a re-run with `--keep-residuals` (Task 3 makes that possible; the ~$12–15 GPU decision is separate and deferred).

"""Strict-JSON writing for experiment result files.

Every result-writing script passes `allow_nan=False`, so a bare `NaN`
literal -- which Python's default encoder emits happily and which is
invalid strict JSON -- can never reach a committed result file. That is
the right trade, but it converts a silent corruption into a hard
`ValueError` at write time, and the write sits outside the per-model
`try/except` that the sweeps rely on. Two problems follow.

First, the message is useless on its own:

    ValueError: Out of range float values are not JSON compliant: nan

with 35 datasets x 7 models in flight and no indication of which. The
realistic trigger is documented in `compute_metrics`: an all-zero
`actual` window (intermittent demand, as occurs in M5) leaves MAPE's
mask empty, so `mape` is NaN while every other metric is finite.

Second, run_baselines.py and run_chronos.py have no resume-from-cache --
unlike run_size_scaling.py and run_arima_only.py, they re-run every
dataset from scratch on every invocation. An abort at dataset 20 of 35
therefore discards the 19 already completed, not just the one that
failed.

`write_result_json` addresses both: it names the offending key paths,
and it serialises to a string BEFORE opening the file so a rejected
payload leaves nothing behind to clean up. It signals failure with
`ResultWriteError`, which sweep drivers catch per dataset and re-report
via `print_write_failures` at the end of the run.

Imports only the standard library, deliberately: this is on the write
path of every script, including the CPU-only ones, and has no business
pulling in the modelling stack.
"""
import json
import math
from pathlib import Path


class ResultWriteError(RuntimeError):
    """A result payload could not be written as strict JSON.

    No file exists at `path` as a result of the failed write -- either it
    was never created, or a previous run's file is still there untouched.

    `offenders` names the non-finite entries found by walking the payload
    (e.g. `metrics.Naive.mape = nan`). It is empty when the payload was
    rejected for some other reason, in which case the underlying
    exception is reported verbatim rather than blaming a non-finite value
    that was not actually found. The original exception is also kept as
    `__cause__`.
    """

    def __init__(self, path, offenders, cause):
        self.path = Path(path)
        self.offenders = list(offenders)
        self.cause = cause
        detail = ", ".join(self.offenders) if self.offenders else f"{type(cause).__name__}: {cause}"
        super().__init__(f"{self.path.name} NOT written -- {detail}")


def non_finite_paths(payload, _path: str = "") -> list[str]:
    """Every non-finite float in `payload`, as `key.path = value` strings.

    Recurses through dicts and lists so a NaN buried in a per-timestep
    residual array (`metrics.ARIMA.residuals[3] = nan`, reachable under
    --keep-residuals) is named as precisely as a top-level metric. The
    point is that the caller is told *which* metric of *which* model went
    non-finite, instead of being handed a 35-dataset search.

    Returns [] when nothing is non-finite; `json.dumps` can reject a
    payload for reasons other than NaN (a non-serialisable object), and
    the caller must not then claim a non-finite value it did not find.
    """
    if isinstance(payload, dict):
        found = []
        for key, value in payload.items():
            child = f"{_path}.{key}" if _path else str(key)
            found.extend(non_finite_paths(value, child))
        return found
    if isinstance(payload, (list, tuple)):
        found = []
        for index, value in enumerate(payload):
            found.extend(non_finite_paths(value, f"{_path}[{index}]"))
        return found
    # bool is not a float, and int is always finite, so only floats (and
    # numpy scalars that subclass float, which is what compute_metrics'
    # rounding produces) can reach the non-finite check.
    if isinstance(payload, float) and not math.isfinite(payload):
        return [f"{_path or '<root>'} = {payload}"]
    return []


def write_result_json(path, payload) -> None:
    """Write `payload` to `path` as strict JSON, or raise ResultWriteError.

    Serialises to a string first and only then opens the file. Handing an
    open file to `json.dump` is not equivalent: the encoder streams into
    the file as it walks the payload and raises part-way through, leaving
    a truncated file that is neither valid JSON nor obviously absent --
    and, because `open(..., "w")` truncates on entry, that also destroys
    whatever a previous run had written there. Failing before the file is
    opened leaves the directory exactly as it was.

    The successful path is byte-for-byte what `json.dump(payload, f,
    indent=2, allow_nan=False)` produced; `tests/test_strict_json_writes.py`
    pins that against the pre-fix write.
    """
    path = Path(path)
    try:
        text = json.dumps(payload, indent=2, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ResultWriteError(path, non_finite_paths(payload), exc) from exc
    with open(path, "w") as f:
        f.write(text)


def print_write_failures(failures: dict) -> None:
    """Re-report every `ResultWriteError` collected during a sweep.

    The per-dataset error is printed the moment it happens, but a
    35-dataset (or 175-run) sweep prints for hours afterwards and the
    warning scrolls away. The scripts most exposed to this have no
    resume-from-cache, so the end of the run is the operator's only
    chance to learn which datasets have no file and must be re-run.

    A no-op when nothing failed, so callers can invoke it unconditionally.
    """
    if not failures:
        return
    print(f"\n{'=' * 70}")
    print(f"{len(failures)} RESULT FILE(S) NOT WRITTEN -- payload was not strict JSON")
    print(f"{'=' * 70}")
    for label, exc in failures.items():
        print(f"  {label}: {exc}")
    print("\nThese runs completed but produced NO result file; re-run them after")
    print("fixing the non-finite value. The usual cause is an all-zero `actual`")
    print("window, which makes MAPE NaN (see shortseq.evaluation.metrics).")

"""Re-populate `arima_rmse` in already-written size-scaling result files.

Why this exists: `run_size_scaling.py` reads ARIMA's RMSE from
`experiments/results/{dataset}_results.json` and stores it in each
scaling JSON *at write time*. When the sweep ran, only 35 of 368 series
had baselines, so 333 scaling files recorded `arima_rmse: null` and
`summarize()` counted them under `no_baseline`.

`summarize()` reads the stored field rather than recomputing it, so
re-running the summary after backfilling baselines would NOT pick them
up. This script closes that gap by rewriting only the `arima_rmse`
field from the now-present baseline files.

It deliberately does not touch the Chronos metrics — those are the
expensive part of the sweep and are already correct.

Every rewrite goes through `shortseq.results_io.write_result_json`, which
serialises before it opens the file. That matters more here than in any
sweep: this loop rewrites 1,840 committed result files that cost real GPU
time to produce, and `open(path, "w")` truncates on entry, so a payload
the encoder rejects part-way used to leave a partial file where a finished
one had been — destroying data this script cannot reconstruct. A rejected
file is recorded and skipped rather than aborting the pass; see main().
Every result writer in the repo routes through it except `run_ablation.py`,
which deliberately does not, for the reason its own docstring gives.
"""
import argparse
import json
from pathlib import Path

from shortseq.results_io import ResultWriteError, print_write_failures, write_result_json

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "experiments" / "results"
SCALING_DIR = REPO_ROOT / "experiments" / "results" / "scaling"


def arima_rmse_for(dataset_name: str) -> float | None:
    path = BASELINE_DIR / f"{dataset_name}_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("metrics", {}).get("ARIMA", {}).get("rmse")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing")
    args = parser.parse_args()

    filled = 0
    already = 0
    still_missing = 0
    changed_value = 0
    # Record-and-continue, mirroring the convention the sweep scripts use one
    # level down. Aborting would protect nothing: write_result_json fails
    # before it opens the file, so every other file in the corpus is
    # untouched and independent, and the usual cause -- one dataset's
    # baseline being non-finite -- affects only that dataset's handful of
    # files. Stopping at the first would leave the corpus half-patched with
    # no tally of what remains, and this pass is idempotent and re-runnable,
    # so a skipped file is simply still un-patched: exactly the state
    # `still_missing` already tolerates per file.
    unwritten = {}

    for path in sorted(SCALING_DIR.glob("*_chronos_*.json")):
        with open(path) as f:
            data = json.load(f)
        dataset = data["dataset"]
        current = data.get("arima_rmse")
        fresh = arima_rmse_for(dataset)

        if fresh is None:
            still_missing += 1
            continue
        if current is not None and current == fresh:
            already += 1
            continue
        if current is not None:
            # Should not normally happen; surface it rather than
            # silently overwriting a previously-recorded reference.
            print(f"  CHANGED {path.name}: {current} -> {fresh}")

        if not args.dry_run:
            data["arima_rmse"] = fresh
            try:
                write_result_json(path, data)
            except ResultWriteError as exc:
                # The file on disk is still the original, byte for byte.
                unwritten[path.name] = exc
                print(f"  [{path.name}] {exc}")
                continue

        # Counted only once the rewrite has actually happened, so a rejected
        # file is reported as unwritten rather than as backfilled. In a dry
        # run nothing is attempted, so both counters behave as they always
        # did: they report what *would* change.
        if current is None:
            filled += 1
        else:
            changed_value += 1

    verb = "would fill" if args.dry_run else "filled"
    print(f"{verb}: {filled}")
    print(f"already correct: {already}")
    print(f"value changed (investigate if >0): {changed_value}")
    print(f"still missing a baseline: {still_missing}")
    # Without this the four counters above sum to fewer than the files the
    # pass looked at, and the difference is visible only in the block below.
    # Over 1,840 files an operator reconciling those numbers has no way to
    # tell a short tally from a corpus that was smaller than they thought.
    print(f"not written (see below): {len(unwritten)}")
    # Last, so it cannot scroll past a 1,840-file pass.
    print_write_failures(unwritten)
    if unwritten:
        # The shared banner is worded for the sweeps, where a rejected
        # payload means no result file exists. Here one does -- the previous
        # contents, intact -- and that difference decides what the operator
        # does next.
        print("Each of those files is UNCHANGED on disk: the rewrite was rejected")
        print("before the file was opened, so only the arima_rmse backfill was lost.")


if __name__ == "__main__":
    main()

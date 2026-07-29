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
"""
import argparse
import json
from pathlib import Path

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

    for path in sorted(SCALING_DIR.glob("*_chronos_*.json")):
        with open(path) as f:
            data = json.load(f)
        dataset = data["dataset"]
        current = data.get("arima_rmse")
        fresh = arima_rmse_for(dataset)

        if fresh is None:
            still_missing += 1
            continue
        if current is None:
            filled += 1
        elif current != fresh:
            # Should not normally happen; surface it rather than
            # silently overwriting a previously-recorded reference.
            changed_value += 1
            print(f"  CHANGED {path.name}: {current} -> {fresh}")
        else:
            already += 1
            continue

        if not args.dry_run:
            data["arima_rmse"] = fresh
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    verb = "would fill" if args.dry_run else "filled"
    print(f"{verb}: {filled}")
    print(f"already correct: {already}")
    print(f"value changed (investigate if >0): {changed_value}")
    print(f"still missing a baseline: {still_missing}")


if __name__ == "__main__":
    main()

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

CPU-only; consumes committed data, runs no models. It takes seconds and
needs no campaign re-run.

Re-running it is deterministic except for the `generated_at` stamp, so a
re-run leaves the two files under experiments/results/ranking/ showing as
modified with only that one line changed. That is expected. Do NOT
`git checkout` result files to tidy the worktree - see the README's
Reproducibility section.
"""
import argparse
import collections
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Imported, never re-declared. A local copy of the size list silently
# diverges the moment a size is added to the campaign, and the column
# filter in build_matrices() then drops that size's perfectly valid
# results without a word - the paper would rank a subset and say nothing.
# Taken from shortseq.constants rather than from run_size_scaling: that
# module imports the dataset registry and the metrics layer at module
# level, so routing the constant through it pulled statsmodels and
# sklearn into a script whose contract (above) is that it runs no models.
# CHRONOS_SIZES is in parameter-count order, so reports read smallest ->
# largest.
from shortseq.constants import CHRONOS_SIZES as SIZE_ORDER
from shortseq.evaluation.ranking import friedman_nemenyi
from shortseq.results_io import write_result_json

REPO_ROOT = Path(__file__).resolve().parents[2]
SCALING_DIR = REPO_ROOT / "experiments" / "results" / "scaling"
OUT_DIR = REPO_ROOT / "experiments" / "results" / "ranking"

# Shared by build_matrices() and main()'s provenance count so the two can
# never disagree about which files constitute "the input".
SCALING_GLOB = "*_chronos_*.json"

# Below this many series in a stratum, the ranking is printed but must not
# be reported. Friedman's statistic is only ASYMPTOTICALLY chi-square
# distributed, so at tiny N the p-value - and the significance flag and
# Nemenyi CD that hang off it - are not trustworthy.
#
# The hard `dead` guard below fires only at ZERO usable results, which
# leaves a real gap: at 367/368 unusable the grid survives dropna() as a
# single series and friedman_nemenyi accepts it (its guard is n < 1),
# yielding a confident-looking "1 series x 5 sizes" ranking.
#
# This is a REPORTING threshold, not a statistical precondition - the
# arithmetic is well-defined at any N >= 1 - so it warns rather than
# aborting, and it lives here rather than in shortseq/evaluation/ranking.py,
# which is a general-purpose library with no stake in how its callers
# publish. Deliberately narrow campaigns are legitimate and must still run.
MIN_SERIES_FOR_REPORT = 10


def scaling_paths() -> list[Path]:
    """Every size-scaling result file, in a stable order."""
    return sorted(SCALING_DIR.glob(SCALING_GLOB))


def build_matrices() -> dict:
    """One RMSE matrix per stratum: rows = series, columns = sizes.

    Series missing any size are dropped (loudly) rather than NaN-filled -
    the Nemenyi critical difference assumes a complete grid.

    A missing RMSE is not noise to be skipped past: run_size_scaling
    leaves `metrics` empty and records the exception in `failed_models`
    when a model fails, so `rmse is None` IS the failure signal. It is
    accounted for per size, because the failure mode that matters is a
    size that failed on every series - its column would never reach the
    pivot, the ranking would quietly proceed at k=4, and the CD would be
    computed for a comparison nobody intended to make. Deliberate
    narrowing (`run_size_scaling.py --sizes ...`) is legitimate and is
    distinguished from breakage by what is on disk, not by list length.
    """
    records, skipped, seen = [], collections.Counter(), collections.Counter()
    for path in scaling_paths():
        with open(path) as f:
            d = json.load(f)
        try:
            size, dataset, stratum = d["size"], d["dataset"], d["stratum"]
        except KeyError as exc:
            # Without the filename this is a bare KeyError against 1,840
            # candidate files.
            raise SystemExit(f"{path.name}: missing required key {exc}") from exc
        seen[size] += 1
        rmse = d.get("metrics", {}).get(f"Chronos-{size}", {}).get("rmse")
        if rmse is None:
            skipped[size] += 1
            reason = d.get("failed_models", {}).get(
                f"Chronos-{size}", "no reason recorded"
            )
            print(f"  [skip] {dataset} @ {size}: {reason}")
            continue
        records.append(
            {
                "dataset": dataset,
                "size": size,
                "stratum": stratum,
                "rmse": rmse,
            }
        )

    unknown = sorted(set(seen) - set(SIZE_ORDER))
    if unknown:
        raise SystemExit(
            f"sizes on disk not in SIZE_ORDER: {unknown} - add them or they are "
            f"silently excluded from the ranking"
        )

    dead = sorted(s for s in seen if skipped[s] == seen[s])
    if dead:
        raise SystemExit(
            f"sizes present on disk with zero usable results: {dead} "
            f"({', '.join(f'{s}: 0/{seen[s]}' for s in dead)}). Fix those runs or remove "
            f"the files; ranking the remainder would silently answer a different question."
        )

    for size in SIZE_ORDER:
        if skipped[size]:
            print(f"  [{size}] {skipped[size]}/{seen[size]} runs unusable")
    print(f"  ranking sizes: {[s for s in SIZE_ORDER if s in seen]}")

    if not records:
        raise SystemExit(f"no usable scaling results under {SCALING_DIR}")

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


def _format_p(p: float) -> str:
    """Readable against alpha directly; e-notation only where .4f would
    print a meaningless 0.0000."""
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.3e}"


def print_report(stratum: str, payload: dict) -> None:
    label = "SHORT (n<200)" if stratum == "short" else "LONG (n>=200)"
    print(f"\n{'=' * 66}")
    print(f"{label}  |  {payload['n_series']} series x {payload['n_models']} sizes")
    print(f"{'=' * 66}")
    if payload["n_series"] < MIN_SERIES_FOR_REPORT:
        print(
            f"  WARNING: only {payload['n_series']} series in this stratum "
            f"(< {MIN_SERIES_FOR_REPORT}). Friedman's chi-square approximation is "
            f"unreliable at this N, so the p-value, the significance flag and the "
            f"critical difference below are all untrustworthy. Diagnose the missing "
            f"series; do NOT report any of this as a finding."
        )
    print(f"Friedman chi2 = {payload['friedman_statistic']:.4f}, "
          f"p = {_format_p(payload['friedman_p_value'])}, "
          f"significant = {payload['friedman_significant']}")
    print(f"Nemenyi critical difference (alpha={payload['alpha']}) = "
          f"{payload['critical_difference']:.4f}")

    print("\nmean ranks (1 = best):")
    for size in SIZE_ORDER:
        if size in payload["mean_ranks"]:
            print(f"  {size:<8}{payload['mean_ranks'][size]:.4f}")

    # Friedman and Nemenyi are different statistics with no strict
    # nesting - for k=5 the Nemenyi threshold is ~6.10/sqrt(N) against
    # Friedman's ~6.89/sqrt(N) - so a non-significant omnibus can sit
    # directly above a pairwise gap that clears the CD. Printing that gap
    # as "YES" under a warning invites exactly the finding the warning
    # forbids, so the verdicts are withheld rather than merely captioned.
    # The JSON keeps both the flag and the raw verdicts, by spec.
    ok = payload["friedman_significant"]
    if not ok:
        print("\n  Friedman NOT significant - pairwise verdicts suppressed below; "
              "these gaps must not be reported as findings.")

    print("\npairwise (significant = mean-rank gap >= critical difference):")
    for pair, info in payload["pairwise"].items():
        mark = ("YES" if info["significant"] else "no ") if ok else "n/a"
        print(f"  {pair:<28} diff={info['rank_difference']:.4f}  sig={mark}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the Friedman test and the Nemenyi "
             "critical difference (default: 0.05)",
    )
    args = parser.parse_args()

    matrices = build_matrices()
    # Provenance: these files ARE the scientific claim, and a re-run that
    # produces only one stratum leaves the other stratum's file sitting
    # there looking current. The count is over the whole input corpus, so
    # a ranking computed from a half-finished campaign is visible in the
    # artifact rather than only in a terminal nobody kept.
    provenance = {
        "n_input_files": len(scaling_paths()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for stratum, matrix in matrices.items():
        payload = analyse(matrix, alpha=args.alpha)
        payload.update(provenance)
        print_report(stratum, payload)
        out_path = OUT_DIR / f"chronos_sizes_{stratum}.json"
        # Routed through shortseq.results_io, which documents why: strict
        # JSON (allow_nan=False) is right - the statistic and p-value are
        # nan only if every size scores identically on every series,
        # unreachable on the real grid, and a bare `NaN` literal would
        # push that failure downstream onto whoever reads the artifact.
        # But `json.dump` into an already-open file makes the rejection
        # destructive rather than merely loud: `open(..., "w")` truncates
        # on entry and the encoder then raises part-way through the walk,
        # so the previous run's valid artifact is replaced by a fragment
        # of unparseable JSON. There are two writes per invocation, so a
        # failure on `long` corrupts `long` while `short` sits there
        # looking freshly written. `write_result_json` serialises to a
        # string first and opens the file only if that succeeds, so a
        # rejected payload leaves the directory exactly as it was and
        # raises ResultWriteError naming the offending key path.
        write_result_json(out_path, payload)
        written.append(out_path.name)

    if written:
        print(f"\nWrote {len(written)} stratum ranking(s) to {OUT_DIR}: "
              f"{', '.join(written)}")
    else:
        print(f"\nNo stratum matrix could be built - NOTHING was written to "
              f"{OUT_DIR}; any files already there are stale.")
    print("NOTE: no pooled cross-stratum ranking is produced, by design.")


if __name__ == "__main__":
    main()

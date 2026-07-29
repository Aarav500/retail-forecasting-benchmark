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
            # allow_nan=False: the statistic/p-value are nan only if every
            # size scores identically on every series (unreachable on the
            # real grid). Python's default would emit a bare `NaN` literal,
            # which is invalid strict JSON, so a downstream consumer would
            # fail on a malformed artifact instead of on the actual problem.
            # This turns that impossible-on-real-data case into a loud,
            # immediate failure here.
            json.dump(payload, f, indent=2, allow_nan=False)

    print(f"\nWrote per-stratum ranking to {OUT_DIR}")
    print("NOTE: no pooled cross-stratum ranking is produced, by design.")


if __name__ == "__main__":
    main()

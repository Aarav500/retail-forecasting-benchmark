"""Project-wide constants with no dependencies of their own.

Deliberately imports nothing - not even numpy. The GPU campaign scripts
and the CPU-only ranking script all read from here, and the ranking
script's contract is "consumes committed data, runs no models". Adding
an import to this module would drag the modelling stack (statsmodels,
sklearn, the dataset registry) into every consumer of a five-string
list, and an import-time failure in any loader would break a script
that needs none of them.
"""

# The Chronos model-size family, in PARAMETER-COUNT order
# (tiny 8M -> mini 20M -> small 46M -> base 200M -> large 710M).
#
# The order is load-bearing, not cosmetic, and all three dependants rely
# on it:
#   - experiments/scripts/run_size_scaling.py sweeps and summarises in
#     this order, and exposes it as `--sizes ... choices=`.
#   - experiments/scripts/run_chronos.py exposes it as `--size choices=`,
#     so the single-size CLI accepts exactly the names the sweep does.
#   - experiments/scripts/run_ranking.py filters the RMSE pivot's columns
#     with it, so reports and JSON key order read smallest -> largest.
#
# One canonical list, because a second copy silently diverges the moment
# a size is added to the campaign: run_ranking's column filter would drop
# the new size's perfectly valid results without a word, and the paper
# would rank a subset and say nothing.
#
# `_MODEL_IDS` in shortseq/models/foundation/chronos.py names the same
# five sizes but is not a fourth copy of this list: it maps each size to
# a Hugging Face checkpoint id, which is data this dependency-free module
# has no business carrying. A size added here but not there fails loudly
# in ChronosForecaster.__init__ rather than silently.
CHRONOS_SIZES = ["tiny", "mini", "small", "base", "large"]

"""Dataset registry — dispatches to individual loaders and filters by regime."""
from .base import SeriesDataset
from .dmart import load_dmart
from .favorita import load_favorita_family, load_favorita_store_item
from .m3_monthly import load_m3_monthly
from .m4 import load_m4
from .m4_weekly import load_m4_weekly
from .m5 import load_m5
from .rossmann import load_rossmann
from .uci import load_uci
from .walmart import load_walmart
from .walmart_real import load_walmart_real


def _load_favorita() -> dict[str, SeriesDataset]:
    """Both Favorita views, under the single source key they belong to.

    Favorita is ONE source with TWO aggregation views (store-item and
    product-family) — see the `favorita` module docstring. It gets one
    registry key so that `len(_LOADERS)` stays an honest count of distinct
    data sources; registering the views separately would inflate the
    benchmark's claimed source diversity. The two views' key prefixes
    (`favorita_store_item_*` / `favorita_family_*`) are disjoint, so
    merging them here cannot collide.
    """
    out = load_favorita_store_item()
    out.update(load_favorita_family())
    return out


_LOADERS = {
    "dmart": load_dmart,
    "uci": load_uci,
    "walmart": load_walmart,
    "m5": load_m5,
    "m4": load_m4,
    "m4_weekly": load_m4_weekly,
    "m3_monthly": load_m3_monthly,
    "rossmann": load_rossmann,
    "walmart_real": load_walmart_real,
    "favorita": _load_favorita,
}

# Legacy names from the NeurIPS execution plan that turned out to denote
# sources already registered above under a different (canonical) name.
# They resolve to the canonical loader rather than erroring, so any code or
# config written against the plan's vocabulary keeps working.
#
# - "corporacion_favorita" / "storesales": the plan listed Favorita,
#   Corporación Favorita and StoreSales as three sources. Investigation
#   (recorded in `favorita.py`) showed they are three aggregations of a
#   single Kaggle competition dump from one Ecuadorian grocery chain. They
#   are therefore ONE source, and any count of sources must count it once.
# - "real_walmart": the plan's name for the real Kaggle Walmart data, which
#   is implemented as `walmart_real` (the bare `walmart` key was already
#   taken by the older calibrated series).
_SOURCE_ALIASES = {
    "corporacion_favorita": "favorita",
    "storesales": "favorita",
    "real_walmart": "walmart_real",
}

# Named in the NeurIPS execution plan but not yet acquired/implemented.
# Tourism is deferred: it needs R / `tsibbledata` to extract.
_PLANNED_SOURCES = ["tourism"]


def load_all() -> dict[str, SeriesDataset]:
    """Load every currently implemented dataset source.

    Raises if two sources produce the same series key — a silent overwrite
    there would quietly shrink the benchmark.
    """
    out: dict[str, SeriesDataset] = {}
    for source, loader in _LOADERS.items():
        loaded = loader()
        collisions = sorted(set(loaded) & set(out))
        if collisions:
            raise ValueError(
                f"Series key collision from source {source!r}: {collisions}"
            )
        out.update(loaded)
    return out


def load_regime(
    cv_range: tuple[float, float] | None = None,
    ac1_range: tuple[float, float] | None = None,
) -> dict[str, SeriesDataset]:
    """Load all implemented datasets, filtered to a CV / AC(1) regime."""
    out: dict[str, SeriesDataset] = {}
    for name, ds in load_all().items():
        if cv_range is not None and not (cv_range[0] <= ds.cv <= cv_range[1]):
            continue
        if ac1_range is not None and not (ac1_range[0] <= ds.ac1 <= ac1_range[1]):
            continue
        out[name] = ds
    return out


def load_source(source: str) -> dict[str, SeriesDataset]:
    """Load a single named source; raises for sources not yet implemented.

    Legacy plan names in `_SOURCE_ALIASES` resolve to their canonical
    source (see that dict for why each is a duplicate name, not a source).
    """
    source = _SOURCE_ALIASES.get(source, source)
    if source in _PLANNED_SOURCES:
        raise NotImplementedError(
            f"Dataset source '{source}' is on the NeurIPS execution plan but "
            "not yet acquired/implemented in this repo."
        )
    if source not in _LOADERS:
        raise ValueError(f"Unknown dataset source: {source!r}")
    return _LOADERS[source]()

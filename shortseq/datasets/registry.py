"""Dataset registry — dispatches to individual loaders and filters by regime."""
from .base import SeriesDataset
from .dmart import load_dmart
from .m4 import load_m4
from .m5 import load_m5
from .uci import load_uci
from .walmart import load_walmart

_LOADERS = {
    "dmart": load_dmart,
    "uci": load_uci,
    "walmart": load_walmart,
    "m5": load_m5,
    "m4": load_m4,
}

# Named in the NeurIPS execution plan but not yet acquired/implemented.
_PLANNED_SOURCES = [
    "favorita", "rossmann", "real_walmart", "corporacion_favorita",
    "storesales", "m4_weekly", "m3_monthly", "tourism",
]


def load_all() -> dict[str, SeriesDataset]:
    """Load every currently implemented dataset source."""
    out: dict[str, SeriesDataset] = {}
    for loader in _LOADERS.values():
        out.update(loader())
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
    """Load a single named source; raises for sources not yet implemented."""
    if source in _PLANNED_SOURCES:
        raise NotImplementedError(
            f"Dataset source '{source}' is on the NeurIPS execution plan but "
            "not yet acquired/implemented in this repo."
        )
    if source not in _LOADERS:
        raise ValueError(f"Unknown dataset source: {source!r}")
    return _LOADERS[source]()

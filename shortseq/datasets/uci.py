"""UCI Online Retail real data (UK, weekly) — 5 product/aggregate series."""
import pandas as pd

from .base import DATA_DIR, SeriesDataset, make_dataset

FILES = {
    "uci_jumbo_bag": "uci_jumbo_bag.csv",
    "uci_lunch_bag": "uci_lunch_bag.csv",
    "uci_red_retrospot": "uci_red_retrospot.csv",
    "uci_regency_cakestand": "uci_regency_cakestand.csv",
    "uci_total": "uci_retail_weekly.csv",
}


def load_uci() -> dict[str, SeriesDataset]:
    """Load the 5 real UCI Online Retail series (n=53 each, weekly)."""
    out: dict[str, SeriesDataset] = {}
    for name, fname in FILES.items():
        df = pd.read_csv(DATA_DIR / fname, parse_dates=["ds"])
        series = df.set_index("ds")["y"].astype(float).sort_index()
        out[name] = make_dataset(name, series, freq="W", real=True)
    return out

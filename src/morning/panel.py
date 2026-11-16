"""Stack the per-stock feature frames into one training panel (the study's layout)."""
from __future__ import annotations

import pandas as pd

from .features import WARMUP_ROWS, stock_frame

# Yahoo tickers of the 10 project stocks (keys match src/config.py)

try:
    import config
except ImportError:
    from src import config

# Both the main application and Morning Call use the same registry.
STOCKS = {
    key: stock.ticker
    for key, stock in config.STOCKS.items()
}

NAMES = {
    key: stock.display_name
    for key, stock in config.STOCKS.items()
}

START = {
    key: stock.start
    for key, stock in config.STOCKS.items()
}

STUDY_FROM, STUDY_TO = "2008-06-01", "2025-12-31"
MIN_ROWS = 300


def build_panel(stock_px: dict, px: dict, date_from=STUDY_FROM, date_to=None) -> pd.DataFrame:
    """stock_px = {key: prices}, px = {market ticker: prices}. Returns date, ticker, features, targets."""
    frames = []
    for key in STOCKS:
        if key not in stock_px:
            continue
        d = stock_px[key].loc[START.get(key, "2000-01-01"):]
        d = d.loc[date_from:date_to] if date_to else d.loc[date_from:]
        if len(d) < MIN_ROWS:
            continue
        x = stock_frame(d, px)
        x["ticker"] = key
        frames.append(x.iloc[WARMUP_ROWS:])
    df = pd.concat(frames).reset_index().rename(columns={"index": "date", "Date": "date"})
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)

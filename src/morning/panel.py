"""Stack the per-stock feature frames into one training panel (the study's layout)."""
from __future__ import annotations

import pandas as pd

from .features import WARMUP_ROWS, stock_frame

# Yahoo tickers of the 10 project stocks (keys match src/config.py)
STOCKS = {"RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS", "HDFCBANK": "HDFCBANK.NS",
          "SBI": "SBIN.NS", "ITC": "ITC.NS", "ADANIENT": "ADANIENT.NS", "VEDL": "VEDL.NS",
          "ADANIGREEN": "ADANIGREEN.NS", "ETERNAL": "ETERNAL.NS"}
NAMES = {"RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services", "INFY": "Infosys",
         "HDFCBANK": "HDFC Bank", "SBI": "State Bank of India", "ITC": "ITC",
         "ADANIENT": "Adani Enterprises", "VEDL": "Vedanta", "ADANIGREEN": "Adani Green Energy",
         "ETERNAL": "Eternal (Zomato)"}
# first usable date per stock (listing dates / start of clean history), as in the study
START = {"TCS": "2004-08-25", "ADANIENT": "2004-07-01", "ADANIGREEN": "2018-06-18", "ETERNAL": "2021-07-23"}
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

"""
Price loading for the Morning Call.

Two sources, one shape. Every function returns  {ticker: DataFrame}  where each frame is indexed
by (timezone-free) session date and has the columns  open, high, low, close, adjclose, volume.

  download()        Yahoo Finance, via yfinance. Used by the daily job on GitHub Actions.
  load_long_file()  an archived long-format file (ticker, date, open, ... ) for offline training.
  load_raw_csv()    one of the project's data/raw/<STOCK>_raw.csv files.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .features import PRICE_COLS, clean


def _normalise_index(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(idx))
    if idx.tz is not None:
        idx = idx.tz_localize(None)          # keep the exchange's own calendar date
    return idx.normalize()


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename Yahoo's columns and clean one ticker's frame."""
    ren = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
           "Adj Close": "adjclose", "Volume": "volume"}
    d = frame.rename(columns=ren)
    if "adjclose" not in d.columns:
        d["adjclose"] = d["close"]
    d = d[PRICE_COLS].copy()
    d.index = _normalise_index(d.index)
    d.index.name = "date"
    d = d.dropna(subset=["close"])
    return clean(d)


def download(tickers, period: str = "3y", retries: int = 4, pause: float = 20.0) -> dict:
    """Download daily bars for every ticker. Raises RuntimeError if Yahoo cannot be reached."""
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(tickers, period=period, interval="1d", auto_adjust=False,
                              group_by="ticker", threads=False, progress=False)
            out, missing = {}, []
            for tk in tickers:
                try:
                    part = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
                    d = _tidy(part.dropna(how="all"))
                except Exception:
                    d = pd.DataFrame()
                if len(d) < 30:
                    missing.append(tk)
                else:
                    out[tk] = d
            # retry individually anything the batch call dropped
            for tk in list(missing):
                try:
                    one = yf.Ticker(tk).history(period=period, interval="1d", auto_adjust=False)
                    d = _tidy(one)
                    if len(d) >= 30:
                        out[tk] = d
                        missing.remove(tk)
                except Exception as exc:          # noqa: BLE001
                    last_error = exc
            if not missing:
                return out
            last_error = RuntimeError(f"no data for {missing}")
        except Exception as exc:                  # noqa: BLE001
            last_error = exc
        if attempt < retries:
            time.sleep(pause * attempt)
    raise RuntimeError(f"Yahoo Finance download failed after {retries} attempts: {last_error}")


def load_long_file(path, tickers=None) -> dict:
    """Archived long-format prices: columns ticker, date, open, high, low, close, adjclose, volume."""
    path = Path(path)
    p = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    p["date"] = pd.to_datetime(p["date"])
    out = {}
    for tk, g in p.groupby("ticker"):
        if tickers is not None and tk not in tickers:
            continue
        out[tk] = clean(g.set_index("date")[PRICE_COLS])
    return out


def load_raw_csv(path) -> pd.DataFrame:
    """A project raw file (Date, Close, High, Low, Open, Volume). No adjusted close: close is used."""
    d = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    d.columns = [c.lower() for c in d.columns]
    d["adjclose"] = d["close"]
    d = d[(d.close > 0) & (d.open > 0)]
    d.index.name = "date"
    return d[PRICE_COLS]

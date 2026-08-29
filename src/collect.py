# src/collect.py
"""
Data collection, driven entirely by the stock registry.

Before this module, notebook 01 hardcoded tickers, date ranges, output
paths and an API key in the notebook body. Adding a stock meant editing
notebook cells. This module reads config.STOCKS instead, so the workflow
for a new stock becomes:

    1. add an entry to config.STOCKS
    2. run collect_prices("NEWSTOCK")

Nothing else changes anywhere in the codebase.

Network access
--------------
Price download needs yfinance to reach Yahoo Finance; news download needs
newsapi.org. Both are called only when you explicitly invoke them, so the
rest of the project works offline from the CSVs already in data/raw/.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import config
    from dataio import load_raw_ohlcv
except ImportError:  # pragma: no cover
    from src import config
    from src.dataio import load_raw_ohlcv


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def collect_prices(stock_key: str,
                   start: Optional[str] = None,
                   end: Optional[str] = None,
                   overwrite: bool = False,
                   verbose: bool = True) -> Path:
    """
    Download OHLCV history for one registered stock and save it to
    data/raw/<raw_filename>.

    Parameters
    ----------
    stock_key : key in config.STOCKS
    start, end : override the registry's dates (YYYY-MM-DD)
    overwrite : if False and the file already exists, skip the download

    Returns the path written (or the existing path if skipped).

    A note on date ranges
    ---------------------
    Ask for as much history as the ticker has. Phase 2 showed the model
    needs a long window: Reliance at 443 rows produced no usable signal at
    any horizon, while TCS at 2,354 rows did. The registry default starts
    in 2014 for this reason. Downloading more data costs nothing.
    """
    stock = config.get_stock(stock_key)
    out_path = stock.raw_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        if verbose:
            print(f"  {stock.key}: {out_path.name} already exists "
                  f"(pass overwrite=True to re-download)")
        return out_path

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is not installed. Run:  %pip install yfinance"
        ) from exc

    start = start or stock.start
    end = end or stock.end

    if verbose:
        print(f"  {stock.key}: downloading {stock.ticker} {start} .. {end}")

    df = yf.download(stock.ticker, start=start, end=end,
                     progress=False, auto_adjust=True)

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for ticker '{stock.ticker}'.\n"
            f"Check the symbol on finance.yahoo.com. Indian stocks need the "
            f"'.NS' (NSE) or '.BO' (BSE) suffix, e.g. INFY.NS"
        )

    df = df.reset_index()

    # yfinance returns a MultiIndex column frame for single tickers in
    # recent versions. Flatten it so the saved CSV has plain headers.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df.to_csv(out_path, index=False)

    if verbose:
        print(f"  {stock.key}: saved {len(df)} rows -> data/raw/{out_path.name}")

    return out_path


def collect_all_prices(overwrite: bool = False,
                       verbose: bool = True) -> dict[str, Path]:
    """Download price history for every registered stock."""
    out = {}
    for key in config.list_stocks():
        try:
            out[key] = collect_prices(key, overwrite=overwrite,
                                      verbose=verbose)
        except Exception as exc:
            print(f"  !! {key} failed: {type(exc).__name__}: {exc}")
    return out


def verify_raw_file(stock_key: str, verbose: bool = True) -> dict:
    """
    Check that a downloaded file is usable BEFORE building features.

    Catches the common failures early: wrong ticker (empty file), a changed
    yfinance output shape, or a window too short to model.
    """
    stock = config.get_stock(stock_key)
    result: dict[str, object] = {"stock": stock.key,
                                 "path": str(stock.raw_path)}

    if not stock.raw_path.exists():
        result["ok"] = False
        result["problem"] = "file missing"
        return result

    try:
        df = load_raw_ohlcv(stock.raw_path)
    except Exception as exc:
        result["ok"] = False
        result["problem"] = f"{type(exc).__name__}: {exc}"
        return result

    n = len(df)
    span_days = (df["Date"].max() - df["Date"].min()).days

    result.update({
        "ok": True,
        "rows": n,
        "start": str(df["Date"].min().date()),
        "end": str(df["Date"].max().date()),
        "years": round(span_days / 365.25, 1),
        "usable_for_horizon_3": n >= 900,
        "warning": None,
    })

    # ~900 raw rows is roughly what survives feature warm-up and
    # non-overlapping sampling at horizon 3 with enough left for 5 folds.
    if n < 900:
        result["warning"] = (
            f"only {n} rows (~{result['years']} years). The pipeline will "
            f"fall back to a shorter horizon. For a horizon-3 model, aim "
            f"for 900+ rows, i.e. about 4 years of daily data."
        )

    if verbose:
        status = "OK" if result["ok"] else "FAIL"
        print(f"  {stock.key:12s} {status}  {n} rows, "
              f"{result['start']} .. {result['end']} ({result['years']}y)")
        if result["warning"]:
            print(f"               ! {result['warning']}")

    return result


def verify_all_raw(verbose: bool = True) -> pd.DataFrame:
    """Verification table for every registered stock."""
    rows = [verify_raw_file(k, verbose=verbose) for k in config.list_stocks()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# News data
# ---------------------------------------------------------------------------

def _load_api_key(env_var: str = "NEWSAPI_KEY") -> Optional[str]:
    """
    Read the NewsAPI key from the environment or a .env file.

    The key must never be committed. Notebook 01 previously contained a
    live key in cell 0, which is now rotated and removed. Put the new one
    in a .env file at the project root:

        NEWSAPI_KEY=your_key_here

    .env is already listed in .gitignore.
    """
    key = os.environ.get(env_var)
    if key:
        return key

    env_path = config.PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == env_var:
                return value.strip().strip('"').strip("'")

    return None


def collect_news(stock_key: str,
                 query: Optional[str] = None,
                 days_back: int = 28,
                 page_size: int = 100,
                 verbose: bool = True) -> pd.DataFrame:
    """
    Fetch recent news headlines for one stock from NewsAPI.

    Honest limitation
    -----------------
    The free NewsAPI tier only serves roughly the last 30 days and caps
    results per request. That is far too little to train on: the model
    needs years of aligned history. This function exists so the LIVE
    pipeline has a working news source, not so you can build a training
    corpus from it.

    For training-scale sentiment you need a historical news archive
    (Bloomberg, Refinitiv, Kaggle financial-news datasets, or a scraped
    corpus). Phase 1 measured the substitute currently in data/raw/
    (Reddit r/worldnews, 2008-2016) at ROC-AUC 0.47 — below chance — which
    is why sentiment is excluded from the model.
    """
    key = _load_api_key()
    if not key:
        raise ValueError(
            "No NewsAPI key found.\n"
            "Create a .env file in the project root containing:\n"
            "    NEWSAPI_KEY=your_key_here\n"
            "Get a free key at https://newsapi.org/register"
        )

    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is not installed.") from exc

    stock = config.get_stock(stock_key)
    query = query or f'"{stock.display_name}" OR "{stock.key} stock"'

    to_date = date.today()
    from_date = to_date - timedelta(days=days_back)

    if verbose:
        print(f"  {stock.key}: querying NewsAPI {from_date} .. {to_date}")

    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": query,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": key,
        },
        timeout=30,
    )

    data = response.json()
    if data.get("status") != "ok":
        raise RuntimeError(
            f"NewsAPI error: {data.get('code')} - {data.get('message')}")

    articles = [
        {
            "stock": stock.key,
            "date": a["publishedAt"][:10],
            "title": a.get("title") or "",
            "description": a.get("description") or "",
        }
        for a in data.get("articles", [])
    ]

    df = pd.DataFrame(articles)
    if verbose:
        print(f"  {stock.key}: {len(df)} articles retrieved")
        if len(df) < 100:
            print("               ! far too few for training; live use only")

    return df


def save_news(df: pd.DataFrame, stock_key: str, verbose: bool = True) -> Path:
    """Save a news frame to data/raw/<KEY>_news_raw.csv."""
    stock = config.get_stock(stock_key)
    path = config.RAW_DIR / f"{stock.key}_news_raw.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if verbose:
        print(f"  saved {len(df)} rows -> data/raw/{path.name}")
    return path


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

def registry_template(key: str, ticker: str, display_name: str,
                      start: str = "2014-01-01",
                      end: Optional[str] = None) -> str:
    """
    Print the exact code block to paste into config.STOCKS.

    Removes the last bit of guesswork from adding a stock: run this, copy
    the output, paste it into config.py.
    """
    end = end or datetime.now().strftime("%Y-%m-%d")
    return f'''    "{key.upper()}": StockConfig(
        key="{key.upper()}",
        ticker="{ticker}",
        display_name="{display_name}",
        raw_filename="{key.upper()}_raw.csv",
        start="{start}",
        end="{end}",
        sentiment_filename=None,
        notes="Added {date.today().isoformat()}.",
    ),'''

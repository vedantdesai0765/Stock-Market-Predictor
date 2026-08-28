# src/dataio.py
"""
All file reading and writing lives here.

The key job of this module is to absorb the messiness of the raw CSVs
(yfinance artefact rows, string-typed numbers, duplicate dates) so that
every downstream module receives one clean, predictable DataFrame shape:

    Date (datetime64), Open, High, Low, Close, Volume (float64)

sorted ascending by Date, with a fresh RangeIndex.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
REQUIRED_COLUMNS = ["Date"] + OHLCV_COLUMNS

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Raw price data
# ---------------------------------------------------------------------------

def load_raw_ohlcv(path: PathLike, verbose: bool = False) -> pd.DataFrame:
    """
    Load a raw yfinance CSV into a clean OHLCV DataFrame.

    Handles the yfinance multi-header artefact automatically: the second
    row of those files contains the ticker string in every column, which
    produces a NaN Date and a non-numeric Close. Both are coerced to NaN
    and dropped, so no hardcoded `.iloc[1:]` is needed.

    Raises
    ------
    FileNotFoundError : if the file does not exist
    ValueError        : if required columns are missing, or nothing survives cleaning
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            f"Expected it inside data/raw/. Check config.STOCKS[...].raw_filename."
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing required column(s): {missing}. "
            f"Found: {list(df.columns)}"
        )

    n_before = len(df)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop artefact / unusable rows
    df = df.dropna(subset=["Date", "Close"])
    df = df.drop_duplicates(subset="Date", keep="last")
    df = df.sort_values("Date").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No usable rows remained after cleaning {path.name}.")

    if verbose:
        dropped = n_before - len(df)
        print(f"  loaded {path.name}: {len(df)} rows "
              f"({dropped} dropped), "
              f"{df['Date'].min().date()} to {df['Date'].max().date()}")

    return df[REQUIRED_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Sentiment data
# ---------------------------------------------------------------------------

def load_daily_sentiment(path: PathLike, verbose: bool = False) -> pd.DataFrame:
    """
    Load an aggregated daily-sentiment CSV.

    Expects a 'date' column plus one or more score columns. Returns a
    DataFrame with a datetime 'date' column, sorted ascending, one row
    per calendar date.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sentiment file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if "date" not in df.columns:
        raise ValueError(
            f"{path.name} has no 'date' column. Found: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.groupby("date", as_index=False).mean(numeric_only=True)
    df = df.sort_values("date").reset_index(drop=True)

    if verbose:
        print(f"  loaded {path.name}: {len(df)} days, "
              f"{df['date'].min().date()} to {df['date'].max().date()}")

    return df


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save_dataset(df: pd.DataFrame, path: PathLike, verbose: bool = True) -> Path:
    """Write a processed dataset, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if verbose:
        print(f"  saved {path.name}: {df.shape[0]} rows x {df.shape[1]} cols "
              f"-> {path.parent.name}/")
    return path


def load_dataset(path: PathLike) -> pd.DataFrame:
    """Read back a processed dataset produced by save_dataset()."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}\n"
            f"Run the build step in notebooks/00_setup_and_build.ipynb first."
        )
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    return df

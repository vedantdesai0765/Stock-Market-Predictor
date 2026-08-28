# src/features.py
"""
The single source of truth for feature engineering.

Before this module existed the same indicators were computed in four
different places (notebook 03, notebook 06, notebook 07 and app.py) and
had already drifted apart. Every one of those should now call into here.

Phase 0 keeps the *same* indicator definitions as the original project so
results stay comparable, but fixes three correctness bugs:

  1. RSI divided by zero on flat/rising streaks, producing inf.
  2. The final row's Target was silently fabricated as 0, because
     `NaN > x` evaluates to False and then `.astype(int)` made it a hard
     DOWN label that dropna() could not see.
  3. Feature code was duplicated, so fixes in one copy never reached the
     others.

Phase 2 will add stationary (scale-free) versions of these features on
top of what is here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # allow both `import features` and `from src import features`
    from config import (
        EMA_SPAN, MACD_FAST, MACD_SIGNAL, MACD_SLOW,
        RSI_WINDOW, SMA_LONG, SMA_SHORT, TARGET_HORIZON,
    )
except ImportError:  # pragma: no cover
    from src.config import (
        EMA_SPAN, MACD_FAST, MACD_SIGNAL, MACD_SLOW,
        RSI_WINDOW, SMA_LONG, SMA_SHORT, TARGET_HORIZON,
    )


# ---------------------------------------------------------------------------
# Individual indicators
# ---------------------------------------------------------------------------

def add_moving_averages(df: pd.DataFrame,
                        short: int = SMA_SHORT,
                        long: int = SMA_LONG,
                        ema_span: int = EMA_SPAN) -> pd.DataFrame:
    """Simple and exponential moving averages of Close."""
    df = df.copy()
    df[f"SMA_{short}"] = df["Close"].rolling(window=short).mean()
    df[f"SMA_{long}"] = df["Close"].rolling(window=long).mean()
    df[f"EMA_{ema_span}"] = df["Close"].ewm(span=ema_span, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, window: int = RSI_WINDOW) -> pd.DataFrame:
    """
    Relative Strength Index, SMA-smoothed (matches the original project).

    BUG FIX: the original computed `avg_gain / avg_loss` directly. When a
    window contained no down-days, avg_loss was 0 and RSI became inf,
    which then poisoned tree splits and any scaler fitted on the column.

    Correct behaviour at the edges:
      - no losses, some gains -> RSI = 100
      - no gains,  some losses -> RSI = 0
      - completely flat window -> RSI = 50 (neutral by convention)
    """
    df = df.copy()
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    flat = avg_gain.eq(0) & avg_loss.eq(0)
    rsi = rsi.mask(avg_loss.eq(0) & avg_gain.gt(0), 100.0)
    rsi = rsi.mask(avg_gain.eq(0) & avg_loss.gt(0), 0.0)
    rsi = rsi.mask(flat, 50.0)

    # Preserve the warm-up NaNs so they can be dropped explicitly later
    rsi = rsi.mask(avg_gain.isna() | avg_loss.isna(), np.nan)

    df["RSI"] = rsi
    return df


def add_macd(df: pd.DataFrame,
             fast: int = MACD_FAST,
             slow: int = MACD_SLOW,
             signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """MACD line and its signal line."""
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percentage return."""
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change()
    return df


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

def add_target(df: pd.DataFrame, horizon: int = TARGET_HORIZON) -> pd.DataFrame:
    """
    Binary direction target: 1 if Close rises over the next `horizon` days.

    BUG FIX: the original wrote

        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    For the final row, Close.shift(-1) is NaN and `NaN > x` is False, so
    .astype(int) produced a hard 0. The subsequent dropna() saw no NaN and
    kept it. Every dataset therefore ended with one invented DOWN label.

    Here the unknown rows are held as NaN so they are visibly droppable.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    df = df.copy()
    future_close = df["Close"].shift(-horizon)

    target = (future_close > df["Close"]).astype(float)
    target[future_close.isna()] = np.nan     # <- the fix

    df["Target"] = target
    return df


# ---------------------------------------------------------------------------
# Optional indicator blocks
# ---------------------------------------------------------------------------

def add_ichimoku_block(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Ichimoku features using the existing src/ichimoku.py module."""
    try:
        from ichimoku import add_all_ichimoku
    except ImportError:
        from src.ichimoku import add_all_ichimoku
    return add_all_ichimoku(df)


def add_bollinger_block(df: pd.DataFrame) -> pd.DataFrame:
    """Attach Bollinger features using the existing src/bollinger.py module."""
    try:
        from bollinger import add_all_bollinger
    except ImportError:
        from src.bollinger import add_all_bollinger
    return add_all_bollinger(df)


ICHIMOKU_FEATURES = [
    "tk_cross", "price_above_cloud", "price_below_cloud",
    "cloud_bullish", "cloud_thickness", "chikou_vs_price",
]

BOLLINGER_FEATURES = [
    "bb_pct_b", "bb_bandwidth", "bb_price_vs_middle",
    "bb_above_upper", "bb_below_lower", "bb_squeeze", "bb_pct_b_delta",
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame,
                   horizon: int = TARGET_HORIZON,
                   with_ichimoku: bool = False,
                   with_bollinger: bool = False,
                   dropna: bool = True,
                   verbose: bool = False) -> pd.DataFrame:
    """
    Run the full feature pipeline on a clean OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame with Date, Open, High, Low, Close, Volume
    horizon : prediction horizon in trading days
    with_ichimoku / with_bollinger : attach the optional indicator blocks
    dropna : drop warm-up rows and the unlabelled tail rows

    Returns
    -------
    DataFrame with all indicator columns plus 'Target'.
    """
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_features is missing columns: {sorted(missing)}")

    n_start = len(df)

    df = df.copy()
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_returns(df)

    if with_ichimoku:
        df = add_ichimoku_block(df)
    if with_bollinger:
        df = add_bollinger_block(df)

    df = add_target(df, horizon=horizon)

    if dropna:
        df = df.dropna().reset_index(drop=True)

    # Target is float after the NaN fix; convert back once rows are dropped
    if df["Target"].notna().all():
        df["Target"] = df["Target"].astype(int)

    if verbose:
        print(f"  features: {n_start} -> {len(df)} rows "
              f"({n_start - len(df)} dropped as warm-up/unlabelled), "
              f"{df.shape[1]} columns")

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def audit_dataset(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Run integrity checks on a built dataset. Returns a dict of results;
    raises nothing, so it can be printed as a report.
    """
    try:
        from config import FORBIDDEN_FEATURES
    except ImportError:
        from src.config import FORBIDDEN_FEATURES

    results: dict[str, object] = {}

    results["rows"] = len(df)
    results["columns"] = df.shape[1]

    leaky = sorted(set(feature_cols) & FORBIDDEN_FEATURES)
    results["leaky_features"] = leaky or None

    numeric = df[feature_cols].select_dtypes(include="number")
    results["has_inf"] = bool(np.isinf(numeric.to_numpy()).any())
    results["nan_counts"] = {
        c: int(n) for c, n in df[feature_cols].isna().sum().items() if n > 0
    } or None

    if "Date" in df.columns:
        d = pd.to_datetime(df["Date"])
        results["dates_sorted"] = bool(d.is_monotonic_increasing)
        results["duplicate_dates"] = int(d.duplicated().sum())
        results["date_range"] = f"{d.min().date()} to {d.max().date()}"

    if "Target" in df.columns:
        vc = df["Target"].value_counts(normalize=True)
        results["target_balance"] = {int(k): round(float(v), 4)
                                     for k, v in vc.items()}
        results["majority_baseline"] = round(float(vc.max()), 4)

    return results


def print_audit(results: dict, title: str = "Dataset audit") -> None:
    """Pretty-print the output of audit_dataset()."""
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in results.items():
        flag = ""
        if k == "leaky_features" and v:
            flag = "   <-- PROBLEM"
        if k == "has_inf" and v:
            flag = "   <-- PROBLEM"
        if k == "duplicate_dates" and v:
            flag = "   <-- PROBLEM"
        if k == "dates_sorted" and v is False:
            flag = "   <-- PROBLEM"
        print(f"  {k:20s} : {v}{flag}")

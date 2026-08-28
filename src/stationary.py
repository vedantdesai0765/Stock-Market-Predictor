# src/stationary.py
"""
Stationary (scale-free) feature engineering.

The problem this solves
-----------------------
The original feature set was built from absolute price levels: SMA_20,
SMA_50, EMA_20 and MACD are all measured in rupees. A decision tree that
learns "SMA_20 > 900 -> UP" on 2014 TCS data is learning a price *level*,
not a market *pattern*. Two consequences:

  1. It does not transfer. TCS trades near 3600 and Reliance near 1100, so
     a threshold learned on one is meaningless on the other. Since the goal
     of this project is "adding a new stock should just work", this is the
     single biggest structural blocker.

  2. It does not survive time. In a trending market, price level is a
     proxy for date, so the model can partly memorise "this era went up"
     rather than learning anything predictive. A permutation test does not
     catch this, because the trend is genuinely in the data.

Every feature in this module is dimensionless: a ratio, a percentage, a
z-score or a binary flag. The same threshold means the same thing on a
1000-rupee stock and a 5000-rupee stock.

Leakage policy
--------------
Every rolling statistic uses pandas' default trailing window, so row t is
computed from rows <= t. Nothing here looks forward. The one construct to
watch is z-scoring: it uses a ROLLING mean/std, never a whole-series one,
because a whole-series z-score leaks the future distribution into the past.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, guarding division by zero."""
    return numerator / denominator.replace(0, np.nan)


def pct_deviation(series: pd.Series, reference: pd.Series) -> pd.Series:
    """
    How far `series` sits above/below `reference`, as a fraction.

        0.03  ->  3% above     -0.02  ->  2% below

    This is the workhorse: it converts any price-level pair into a
    dimensionless quantity.
    """
    return safe_ratio(series, reference) - 1.0


def rolling_zscore(series: pd.Series, window: int = 60,
                   min_periods: int | None = None) -> pd.Series:
    """
    Trailing z-score. Uses only the previous `window` observations, so it
    is leakage-free, unlike a whole-series (series - series.mean()) / std.
    """
    if min_periods is None:
        min_periods = max(window // 2, 10)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return safe_ratio(series - mean, std)


# ---------------------------------------------------------------------------
# Price-position features
# ---------------------------------------------------------------------------

def add_price_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace absolute moving averages with the price's position relative
    to them, plus the relationship between the averages themselves.

    Requires SMA_20, SMA_50, EMA_20 (produced by features.py).
    """
    df = df.copy()
    close = df["Close"]

    if "SMA_20" in df.columns:
        df["close_vs_sma20"] = pct_deviation(close, df["SMA_20"])
    if "SMA_50" in df.columns:
        df["close_vs_sma50"] = pct_deviation(close, df["SMA_50"])
    if "EMA_20" in df.columns:
        df["close_vs_ema20"] = pct_deviation(close, df["EMA_20"])
    if {"SMA_20", "SMA_50"} <= set(df.columns):
        # Classic golden/death-cross signal, expressed continuously
        df["sma20_vs_sma50"] = pct_deviation(df["SMA_20"], df["SMA_50"])

    return df


def add_normalised_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD is a difference of two EMAs, so it scales with price. Dividing by
    Close turns it into a comparable percentage across stocks.
    """
    df = df.copy()
    close = df["Close"]

    if "MACD" in df.columns:
        df["macd_norm"] = safe_ratio(df["MACD"], close)
    if {"MACD", "MACD_signal"} <= set(df.columns):
        df["macd_hist_norm"] = safe_ratio(df["MACD"] - df["MACD_signal"], close)

    return df


def add_centred_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    RSI is already bounded 0-100, so it is scale-free. Centring it on zero
    (-1 to +1) just makes the neutral point explicit and puts it on the
    same footing as the other features.
    """
    df = df.copy()
    if "RSI" in df.columns:
        df["rsi_centered"] = (df["RSI"] - 50.0) / 50.0
    return df


# ---------------------------------------------------------------------------
# Return-based features
# ---------------------------------------------------------------------------

def add_lagged_returns(df: pd.DataFrame,
                       lags: tuple[int, ...] = (1, 2, 3, 5, 10)) -> pd.DataFrame:
    """
    Yesterday's return, the day before, and so on. Returns are naturally
    stationary and are the most direct short-horizon signal available.

    ret_lag_1 at row t is the return realised BETWEEN t-1 and t, which is
    known at the close of t. No lookahead.
    """
    df = df.copy()
    ret = df["Close"].pct_change()
    for lag in lags:
        df[f"ret_lag_{lag}"] = ret.shift(lag - 1)
    return df


def add_momentum(df: pd.DataFrame,
                 windows: tuple[int, ...] = (5, 10, 20, 60)) -> pd.DataFrame:
    """Cumulative return over trailing windows — multi-scale momentum."""
    df = df.copy()
    close = df["Close"]
    for w in windows:
        df[f"momentum_{w}"] = safe_ratio(close, close.shift(w)) - 1.0
    return df


def add_return_distribution(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Shape of the recent return distribution. Skew and kurtosis capture
    asymmetry and tail behaviour that a mean/std pair misses.
    """
    df = df.copy()
    ret = df["Close"].pct_change()
    mp = max(window // 2, 5)
    df[f"ret_skew_{window}"] = ret.rolling(window, min_periods=mp).skew()
    df[f"ret_kurt_{window}"] = ret.rolling(window, min_periods=mp).kurt()
    df["ret_zscore_60"] = rolling_zscore(ret, window=60)
    return df


# ---------------------------------------------------------------------------
# Volatility features
# ---------------------------------------------------------------------------

def add_volatility(df: pd.DataFrame,
                   windows: tuple[int, ...] = (10, 20)) -> pd.DataFrame:
    """
    Three volatility estimators, all normalised by price:

      - realized_vol : rolling std of returns (the standard measure)
      - atr_norm     : Average True Range / Close (range-based, handles gaps)
      - parkinson    : high-low estimator, more efficient than close-to-close

    Volatility regime matters: the same RSI reading means something
    different in a calm market than a violent one.
    """
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]
    ret = close.pct_change()

    for w in windows:
        mp = max(w // 2, 5)
        df[f"realized_vol_{w}"] = ret.rolling(w, min_periods=mp).std()

    # Average True Range, normalised
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(14, min_periods=7).mean()
    df["atr_14_norm"] = safe_ratio(atr, close)

    # Parkinson high-low volatility estimator
    hl = np.log(safe_ratio(high, low))
    df["parkinson_vol_10"] = np.sqrt(
        (hl ** 2).rolling(10, min_periods=5).mean() / (4 * np.log(2))
    )

    # Volatility-of-volatility: is the regime itself changing?
    if "realized_vol_20" in df.columns:
        df["vol_ratio_10_20"] = safe_ratio(
            df.get("realized_vol_10", ret.rolling(10, min_periods=5).std()),
            df["realized_vol_20"],
        )

    return df


# ---------------------------------------------------------------------------
# Volume and intraday structure
# ---------------------------------------------------------------------------

def add_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Raw Volume is meaningless across stocks (TCS trades millions of shares,
    a smallcap trades thousands). Volume relative to its own recent average
    is comparable everywhere.
    """
    df = df.copy()
    vol = df["Volume"]
    mp = max(window // 2, 5)
    avg = vol.rolling(window, min_periods=mp).mean()
    df[f"volume_ratio_{window}"] = safe_ratio(vol, avg)
    df["volume_zscore_60"] = rolling_zscore(vol, window=60)
    return df


def add_candle_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intraday shape, all divided by Close so they are percentages.

      - hl_range_norm : the day's trading range
      - body_norm     : close minus open (direction and conviction)
      - gap_norm      : overnight gap from yesterday's close
      - upper/lower wick: rejection of highs or lows
    """
    df = df.copy()
    close, high, low, open_ = df["Close"], df["High"], df["Low"], df["Open"]

    df["hl_range_norm"] = safe_ratio(high - low, close)
    df["body_norm"] = safe_ratio(close - open_, close)
    df["gap_norm"] = safe_ratio(open_ - close.shift(1), close.shift(1))

    upper = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower = pd.concat([close, open_], axis=1).min(axis=1) - low
    df["upper_wick_norm"] = safe_ratio(upper, close)
    df["lower_wick_norm"] = safe_ratio(lower, close)

    # Where in the day's range did we close? 0 = at the low, 1 = at the high
    df["close_location"] = safe_ratio(close - low, (high - low))

    return df


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame,
                          date_col: str = "Date") -> pd.DataFrame:
    """
    Day-of-week and period-end flags. Cheap, and there are documented
    calendar effects (month-end rebalancing, Monday effects) worth giving
    the model the chance to find or ignore.
    """
    df = df.copy()
    d = pd.to_datetime(df[date_col])

    df["day_of_week"] = d.dt.dayofweek
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_quarter_end"] = d.dt.is_quarter_end.astype(int)

    return df


# ---------------------------------------------------------------------------
# Normalising the existing indicator blocks
# ---------------------------------------------------------------------------

def normalise_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ichimoku's binary features (tk_cross, price_above_cloud, cloud_bullish,
    price_below_cloud) are already scale-free. Its two continuous ones are
    not — they are rupee differences — so divide them by Close.
    """
    df = df.copy()
    close = df["Close"]

    if "cloud_thickness" in df.columns:
        df["cloud_thickness_norm"] = safe_ratio(df["cloud_thickness"], close)
    if "chikou_vs_price" in df.columns:
        df["chikou_vs_price_norm"] = safe_ratio(df["chikou_vs_price"], close)
    if {"Tenkan_sen", "Kijun_sen"} <= set(df.columns):
        df["tenkan_vs_kijun_norm"] = safe_ratio(
            df["Tenkan_sen"] - df["Kijun_sen"], close)
    if "Kijun_sen" in df.columns:
        df["close_vs_kijun"] = pct_deviation(close, df["Kijun_sen"])

    return df


def normalise_bollinger(df: pd.DataFrame) -> pd.DataFrame:
    """
    bb_pct_b and bb_bandwidth are already dimensionless by construction.
    bb_price_vs_middle is a rupee difference, so normalise it.
    """
    df = df.copy()
    close = df["Close"]

    if "bb_price_vs_middle" in df.columns:
        df["bb_price_vs_middle_norm"] = safe_ratio(
            df["bb_price_vs_middle"], close)
    if "bb_bandwidth" in df.columns:
        # Is the current squeeze unusual for THIS stock's own history?
        df["bb_bandwidth_zscore"] = rolling_zscore(df["bb_bandwidth"], 60)

    return df


# ---------------------------------------------------------------------------
# Feature name registries
# ---------------------------------------------------------------------------

STATIONARY_PRICE = [
    "close_vs_sma20", "close_vs_sma50", "close_vs_ema20", "sma20_vs_sma50",
    "macd_norm", "macd_hist_norm", "rsi_centered",
]

STATIONARY_RETURNS = [
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5", "ret_lag_10",
    "momentum_5", "momentum_10", "momentum_20", "momentum_60",
    "ret_skew_20", "ret_kurt_20", "ret_zscore_60",
]

STATIONARY_VOLATILITY = [
    "realized_vol_10", "realized_vol_20", "atr_14_norm",
    "parkinson_vol_10", "vol_ratio_10_20",
]

STATIONARY_VOLUME = [
    "volume_ratio_20", "volume_zscore_60",
]

STATIONARY_CANDLE = [
    "hl_range_norm", "body_norm", "gap_norm",
    "upper_wick_norm", "lower_wick_norm", "close_location",
]

STATIONARY_CALENDAR = [
    "day_of_week", "is_month_end", "is_month_start", "is_quarter_end",
]

STATIONARY_ICHIMOKU = [
    "tk_cross", "price_above_cloud", "price_below_cloud", "cloud_bullish",
    "cloud_thickness_norm", "chikou_vs_price_norm",
    "tenkan_vs_kijun_norm", "close_vs_kijun",
]

STATIONARY_BOLLINGER = [
    "bb_pct_b", "bb_bandwidth", "bb_above_upper", "bb_below_lower",
    "bb_squeeze", "bb_pct_b_delta",
    "bb_price_vs_middle_norm", "bb_bandwidth_zscore",
]

STATIONARY_CORE = (STATIONARY_PRICE + STATIONARY_RETURNS
                   + STATIONARY_VOLATILITY + STATIONARY_VOLUME
                   + STATIONARY_CANDLE + STATIONARY_CALENDAR)

STATIONARY_ALL = (STATIONARY_CORE + STATIONARY_ICHIMOKU
                  + STATIONARY_BOLLINGER)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def add_stationary_features(df: pd.DataFrame,
                            with_ichimoku: bool = True,
                            with_bollinger: bool = True,
                            with_calendar: bool = True,
                            verbose: bool = False) -> pd.DataFrame:
    """
    Add every stationary feature to a DataFrame that has already been
    through features.build_features().

    The original non-stationary columns are LEFT IN PLACE, so the two
    feature sets can be compared on identical rows and identical folds.
    Selection of which to feed a model happens at model time, not here.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"add_stationary_features is missing columns: {sorted(missing)}")

    n_before = df.shape[1]

    df = add_price_position(df)
    df = add_normalised_macd(df)
    df = add_centred_rsi(df)
    df = add_lagged_returns(df)
    df = add_momentum(df)
    df = add_return_distribution(df)
    df = add_volatility(df)
    df = add_volume_features(df)
    df = add_candle_structure(df)

    if with_calendar:
        df = add_calendar_features(df)
    if with_ichimoku:
        df = normalise_ichimoku(df)
    if with_bollinger:
        df = normalise_bollinger(df)

    # Infinities can arise from divisions on degenerate rows (zero range,
    # zero volume). Convert to NaN so they are dropped rather than poisoning
    # any scaler or split downstream.
    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan)

    if verbose:
        print(f"  stationary: {n_before} -> {df.shape[1]} columns "
              f"(+{df.shape[1] - n_before})")

    return df


def get_stationary_features(df: pd.DataFrame,
                            include_ichimoku: bool = True,
                            include_bollinger: bool = True,
                            include_calendar: bool = True) -> list[str]:
    """Return the stationary feature names actually present in `df`."""
    cols = (STATIONARY_PRICE + STATIONARY_RETURNS + STATIONARY_VOLATILITY
            + STATIONARY_VOLUME + STATIONARY_CANDLE)
    if include_calendar:
        cols += STATIONARY_CALENDAR
    if include_ichimoku:
        cols += STATIONARY_ICHIMOKU
    if include_bollinger:
        cols += STATIONARY_BOLLINGER
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def scale_report(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Show each feature's magnitude. Stationary features should have means
    near zero and small standard deviations; a feature with a mean in the
    hundreds is still carrying price level and has not been converted.
    """
    sub = df[feature_cols].select_dtypes(include="number")
    rep = pd.DataFrame({
        "mean": sub.mean(),
        "std": sub.std(),
        "min": sub.min(),
        "max": sub.max(),
        "abs_mean": sub.abs().mean(),
    })
    rep["likely_price_scale"] = rep["abs_mean"] > 10
    return rep.round(4).sort_values("abs_mean", ascending=False)


def compare_scales(df_a: pd.DataFrame, df_b: pd.DataFrame,
                   feature_cols: list[str],
                   name_a: str = "A", name_b: str = "B") -> pd.DataFrame:
    """
    Compare the same features across two stocks.

    This is the direct test of transferability: if a feature's mean differs
    by an order of magnitude between two stocks, a threshold learned on one
    cannot apply to the other.
    """
    cols = [c for c in feature_cols
            if c in df_a.columns and c in df_b.columns]
    a, b = df_a[cols].mean(), df_b[cols].mean()

    out = pd.DataFrame({f"{name_a}_mean": a, f"{name_b}_mean": b})
    denom = out[[f"{name_a}_mean", f"{name_b}_mean"]].abs().max(axis=1)
    out["abs_diff"] = (a - b).abs()
    out["relative_diff"] = out["abs_diff"] / denom.replace(0, np.nan)
    out["transferable"] = out["abs_diff"] < 1.0

    return out.round(4).sort_values("abs_diff", ascending=False)

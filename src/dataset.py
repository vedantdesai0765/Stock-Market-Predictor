# src/dataset.py
"""
End-to-end dataset construction.

    from dataset import build_dataset
    df = build_dataset("TCS")

That single call loads the raw CSV, engineers every feature, attaches the
target, optionally merges sentiment, audits the result and writes it to
data/processed/<KEY>_dataset.csv.

Adding a new stock requires no changes here: register it in config.STOCKS
and call build_dataset() with its key.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

try:
    import config
    from config import StockConfig
    from dataio import load_daily_sentiment, load_raw_ohlcv, save_dataset
    from features import (BOLLINGER_FEATURES, ICHIMOKU_FEATURES,
                          audit_dataset, build_features, print_audit)
except ImportError:  # pragma: no cover
    from src import config
    from src.config import StockConfig
    from src.dataio import load_daily_sentiment, load_raw_ohlcv, save_dataset
    from src.features import (BOLLINGER_FEATURES, ICHIMOKU_FEATURES,
                              audit_dataset, build_features, print_audit)


# ---------------------------------------------------------------------------
# Sentiment merge
# ---------------------------------------------------------------------------

def merge_sentiment(features_df: pd.DataFrame,
                    sentiment_df: pd.DataFrame,
                    lag_days: int = 0,
                    trim_to_news: bool = True,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Left-join daily sentiment onto the feature frame.

    Parameters
    ----------
    lag_days : int
        Shift sentiment forward by this many days before joining, so that
        day t only ever sees news published on day t - lag_days.
        Phase 0 default is 0 to preserve the original behaviour exactly.
        Phase 5 will switch this to 1.
    trim_to_news : bool
        Restrict the price rows to the news date range. This is what
        collapses the TCS dataset from ~2400 rows to ~565.

    Unlike the original merge_sentiment_features(), this never writes a
    file as a side effect and never silently returns an empty frame.
    """
    features_df = features_df.copy()
    sentiment_df = sentiment_df.copy()

    features_df["Date"] = pd.to_datetime(features_df["Date"])
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

    if lag_days:
        sentiment_df["date"] = sentiment_df["date"] + pd.Timedelta(days=lag_days)

    score_cols = [c for c in sentiment_df.columns if c != "date"]

    if trim_to_news and not sentiment_df.empty:
        lo, hi = sentiment_df["date"].min(), sentiment_df["date"].max()
        before = len(features_df)
        features_df = features_df[
            (features_df["Date"] >= lo) & (features_df["Date"] <= hi)
        ]
        if verbose:
            print(f"  trimmed to news window {lo.date()}..{hi.date()}: "
                  f"{before} -> {len(features_df)} rows")

        if features_df.empty:
            raise ValueError(
                "No overlap between price dates and news dates. "
                "Either widen the news range or set trim_to_news=False."
            )

    merged = features_df.merge(
        sentiment_df, left_on="Date", right_on="date", how="left"
    ).drop(columns=["date"], errors="ignore")

    # Forward-fill across weekends/holidays, then neutral-fill the head
    for col in score_cols:
        n_missing = int(merged[col].isna().sum())
        merged[col] = merged[col].ffill().fillna(0.0)
        if verbose and n_missing:
            print(f"  {col}: {n_missing} missing day(s) forward-filled")

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dataset(stock_key: str,
                  horizon: Optional[int] = None,
                  with_ichimoku: bool = True,
                  with_bollinger: bool = True,
                  with_sentiment: bool = True,
                  sentiment_lag_days: int = 0,
                  trim_to_news: bool = True,
                  save: bool = True,
                  verbose: bool = True) -> pd.DataFrame:
    """
    Build (and optionally save) the full modelling dataset for one stock.

    Returns the DataFrame. The saved file path is stock.features_path.
    """
    stock: StockConfig = config.get_stock(stock_key)
    horizon = config.TARGET_HORIZON if horizon is None else horizon

    if verbose:
        print(f"\n=== Building dataset: {stock.key} ({stock.display_name}) ===")

    # 1. Load raw prices
    raw = load_raw_ohlcv(stock.raw_path, verbose=verbose)

    # 2. Engineer features + target
    df = build_features(
        raw,
        horizon=horizon,
        with_ichimoku=with_ichimoku,
        with_bollinger=with_bollinger,
        dropna=True,
        verbose=verbose,
    )

    # 3. Merge sentiment where available
    used_sentiment = False
    if with_sentiment and stock.has_sentiment:
        sent = load_daily_sentiment(stock.sentiment_path, verbose=verbose)
        df = merge_sentiment(
            df, sent,
            lag_days=sentiment_lag_days,
            trim_to_news=trim_to_news,
            verbose=verbose,
        )
        used_sentiment = True
    elif with_sentiment and verbose:
        print("  no sentiment file registered; skipping sentiment merge")

    # 4. Assemble the feature list actually present
    feature_cols = list(config.BASELINE_FEATURES)
    if with_ichimoku:
        feature_cols += ICHIMOKU_FEATURES
    if with_bollinger:
        feature_cols += BOLLINGER_FEATURES
    if used_sentiment:
        feature_cols += [c for c in config.SENTIMENT_FEATURES if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # 5. Audit
    if verbose:
        results = audit_dataset(df, feature_cols)
        results["sentiment_merged"] = used_sentiment
        results["n_features"] = len(feature_cols)
        print_audit(results, title=f"Audit: {stock.key}")

    # 6. Save
    if save:
        save_dataset(df, stock.features_path, verbose=verbose)

    return df


def build_all(verbose: bool = True, **kwargs) -> dict[str, pd.DataFrame]:
    """Build datasets for every stock in the registry."""
    out: dict[str, pd.DataFrame] = {}
    for key in config.list_stocks():
        try:
            out[key] = build_dataset(key, verbose=verbose, **kwargs)
        except Exception as exc:  # keep going so one bad file doesn't stop all
            print(f"\n  !! {key} failed: {type(exc).__name__}: {exc}")
    return out


def get_feature_columns(df: pd.DataFrame,
                        include_sentiment: bool = True,
                        include_ichimoku: bool = True,
                        include_bollinger: bool = True) -> list[str]:
    """
    Return the model-input columns present in `df`, excluding anything in
    config.FORBIDDEN_FEATURES. Use this instead of select_dtypes(), which
    silently swept raw OHLCV into the original models.
    """
    cols = list(config.BASELINE_FEATURES)
    if include_ichimoku:
        cols += ICHIMOKU_FEATURES
    if include_bollinger:
        cols += BOLLINGER_FEATURES
    if include_sentiment:
        cols += config.SENTIMENT_FEATURES
    return [c for c in cols
            if c in df.columns and c not in config.FORBIDDEN_FEATURES]


if __name__ == "__main__":
    config.ensure_dirs()
    build_all()

# src/bollinger.py
"""
Bollinger Bands — reusable module.

Uses the `ta` library (BollingerBands) for core calculations and adds
ML-safe derived features free of look-ahead / data leakage.

Bollinger Band Theory
---------------------
Developed by John Bollinger (1980s):
  - Middle Band  : N-period SMA of Close          (default N=20)
  - Upper Band   : Middle + (k × N-period std)    (default k=2)
  - Lower Band   : Middle - (k × N-period std)    (default k=2)

Key properties:
  - Bands widen during high volatility, contract during low volatility.
  - %B locates price relative to the bands (0 = lower, 1 = upper band).
  - Bandwidth measures the width as a % of the middle band (volatility proxy).
  - A "squeeze" (very narrow bandwidth) often precedes a breakout.

All features use only past data — no look-ahead leakage.
"""

import pandas as pd
from ta.volatility import BollingerBands


# ---------------------------------------------------------------------------
# Core Bollinger Band computation
# ---------------------------------------------------------------------------

def compute_bollinger(df: pd.DataFrame,
                      window: int = 20,
                      window_dev: float = 2.0) -> pd.DataFrame:
    """
    Add raw Bollinger Band columns to *df* and return it.

    Expects column: 'Close'
    Returns the same DataFrame with new columns:
        - bb_upper      : Upper band  (SMA + 2σ)
        - bb_middle     : Middle band (SMA)
        - bb_lower      : Lower band  (SMA - 2σ)
        - bb_pct_b      : %B  = (Close - Lower) / (Upper - Lower)
        - bb_bandwidth  : Bandwidth = (Upper - Lower) / Middle × 100
    """
    df = df.copy()

    indicator = BollingerBands(
        close=df['Close'],
        window=window,
        window_dev=window_dev,
        fillna=False
    )

    df['bb_upper']     = indicator.bollinger_hband()
    df['bb_middle']    = indicator.bollinger_mavg()
    df['bb_lower']     = indicator.bollinger_lband()
    df['bb_pct_b']     = indicator.bollinger_pband()   # %B  (0–1 inside bands)
    df['bb_bandwidth'] = indicator.bollinger_wband()   # Bandwidth %

    return df


# ---------------------------------------------------------------------------
# ML-safe derived features
# ---------------------------------------------------------------------------

def add_bollinger_ml_features(df: pd.DataFrame,
                               squeeze_pct: float = 10.0) -> pd.DataFrame:
    """
    Add six ML-safe features derived from the raw Bollinger Bands.

    Pre-condition: compute_bollinger() must have been called first.

    Parameters
    ----------
    squeeze_pct : float
        Bandwidth percentile threshold below which we flag a squeeze.
        Default 10 = bottom 10th percentile of historical bandwidth.

    New columns
    -----------
        - bb_price_vs_middle  : Close − Middle band  (trend position)
        - bb_above_upper      : 1 if Close > Upper band  (overbought / breakout)
        - bb_below_lower      : 1 if Close < Lower band  (oversold  / breakdown)
        - bb_pct_b            : already added by compute_bollinger (kept as-is)
        - bb_bandwidth        : already added by compute_bollinger (kept as-is)
        - bb_squeeze          : 1 when bandwidth is in its bottom squeeze_pct %ile
                               (low volatility → potential explosive move)
        - bb_pct_b_delta      : 1-period change in %B  (momentum of band position)
    """
    df = df.copy()

    required = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_pct_b',
                'bb_bandwidth', 'Close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing Bollinger columns: {missing}. "
            "Run compute_bollinger() first."
        )

    df['bb_price_vs_middle'] = df['Close'] - df['bb_middle']
    df['bb_above_upper']     = (df['Close'] > df['bb_upper']).astype(int)
    df['bb_below_lower']     = (df['Close'] < df['bb_lower']).astype(int)

    # Squeeze: bandwidth in the bottom squeeze_pct-th percentile
    squeeze_threshold        = df['bb_bandwidth'].quantile(squeeze_pct / 100)
    df['bb_squeeze']         = (df['bb_bandwidth'] <= squeeze_threshold).astype(int)

    # Rate of change of %B (1-period delta)
    df['bb_pct_b_delta']     = df['bb_pct_b'].diff()

    return df


# ---------------------------------------------------------------------------
# Convenience: compute everything in one call
# ---------------------------------------------------------------------------

def add_all_bollinger(df: pd.DataFrame,
                      window: int = 20,
                      window_dev: float = 2.0,
                      squeeze_pct: float = 10.0) -> pd.DataFrame:
    """
    Run compute_bollinger() then add_bollinger_ml_features() in one shot.
    Returns the enriched DataFrame.
    """
    df = compute_bollinger(df, window=window, window_dev=window_dev)
    df = add_bollinger_ml_features(df, squeeze_pct=squeeze_pct)
    return df


# ---------------------------------------------------------------------------
# Quick sanity check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/TCS_raw.csv"

    raw = pd.read_csv(data_path)
    raw = raw.iloc[1:].reset_index(drop=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values('Date').reset_index(drop=True)

    result = add_all_bollinger(raw)

    bb_cols = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_pct_b',
               'bb_bandwidth', 'bb_price_vs_middle', 'bb_above_upper',
               'bb_below_lower', 'bb_squeeze', 'bb_pct_b_delta']

    print("=== Bollinger Band Columns (last 5 rows) ===")
    print(result[['Date'] + bb_cols].tail().to_string())

    print("\n=== NaN counts ===")
    print(result[bb_cols].isna().sum().to_string())

    print("\n=== Data types ===")
    print(result[bb_cols].dtypes.to_string())

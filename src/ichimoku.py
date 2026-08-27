# src/ichimoku.py
"""
Ichimoku Kinko Hyo — reusable module.

Uses the `ta` library (IchimokuIndicator) for core calculations and adds
ML-safe derived features that are guaranteed free of look-ahead / data leakage.

Leakage note
------------
The ta library's ichimoku_a() and ichimoku_b() return the Senkou Spans
*already displaced* +26 bars forward (chart convention).  That means the
value stored at index t was computed from data ending at t-26, which is
perfectly safe for ML — it is purely historical information at time t.
We therefore use these columns as-is for ML features.

The Chikou Span (closing price shifted 26 bars into the PAST) is also
look-back safe; we compute it as  Close.shift(26)  so row t contains
the close price from 26 trading days ago.
"""

import pandas as pd
from ta.trend import IchimokuIndicator


# ---------------------------------------------------------------------------
# Core Ichimoku computation
# ---------------------------------------------------------------------------

def compute_ichimoku(df: pd.DataFrame,
                     window1: int = 9,
                     window2: int = 26,
                     window3: int = 52) -> pd.DataFrame:
    """
    Add the five raw Ichimoku components to *df* and return it.

    Expects columns: 'High', 'Low', 'Close'
    Returns the same DataFrame with new columns added:
        - Tenkan_sen      : Conversion line  (window1-period midpoint)
        - Kijun_sen       : Base line        (window2-period midpoint)
        - Senkou_Span_A   : Leading Span A   (displaced +window2 forward by ta)
        - Senkou_Span_B   : Leading Span B   (displaced +window2 forward by ta)
        - Chikou_Span     : Lagging Span     (close shifted window2 bars back)
    """
    df = df.copy()

    indicator = IchimokuIndicator(
        high=df['High'],
        low=df['Low'],
        window1=window1,
        window2=window2,
        window3=window3,
        fillna=False
    )

    df['Tenkan_sen']    = indicator.ichimoku_conversion_line()
    df['Kijun_sen']     = indicator.ichimoku_base_line()
    df['Senkou_Span_A'] = indicator.ichimoku_a()   # already +26 displaced
    df['Senkou_Span_B'] = indicator.ichimoku_b()   # already +26 displaced

    # Chikou Span: current close plotted 26 bars in the past.
    # shift(26) gives us the close from 26 bars ago at the current row.
    df['Chikou_Span'] = df['Close'].shift(window2)

    return df


# ---------------------------------------------------------------------------
# ML-safe derived features
# ---------------------------------------------------------------------------

def add_ichimoku_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add six ML-safe derived features from the raw Ichimoku components.

    Pre-condition: compute_ichimoku() must have been called on *df* first.

    New columns:
        - tk_cross            : 1 if Tenkan > Kijun (bullish cross signal)
        - price_above_cloud   : 1 if Close > max(Span_A, Span_B)
        - price_below_cloud   : 1 if Close < min(Span_A, Span_B)
        - cloud_bullish       : 1 if Span_A > Span_B  (green cloud)
        - cloud_thickness     : abs(Span_A - Span_B)  (momentum proxy)
        - chikou_vs_price     : Chikou_Span - Close    (momentum proxy)
    """
    df = df.copy()

    required = ['Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A',
                'Senkou_Span_B', 'Chikou_Span', 'Close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing Ichimoku columns: {missing}. "
            "Run compute_ichimoku() first."
        )

    cloud_upper = df[['Senkou_Span_A', 'Senkou_Span_B']].max(axis=1)
    cloud_lower = df[['Senkou_Span_A', 'Senkou_Span_B']].min(axis=1)

    df['tk_cross']          = (df['Tenkan_sen'] > df['Kijun_sen']).astype(int)
    df['price_above_cloud'] = (df['Close'] > cloud_upper).astype(int)
    df['price_below_cloud'] = (df['Close'] < cloud_lower).astype(int)
    df['cloud_bullish']     = (df['Senkou_Span_A'] > df['Senkou_Span_B']).astype(int)
    df['cloud_thickness']   = (df['Senkou_Span_A'] - df['Senkou_Span_B']).abs()
    df['chikou_vs_price']   = df['Chikou_Span'] - df['Close']

    return df


# ---------------------------------------------------------------------------
# Convenience: compute everything in one call
# ---------------------------------------------------------------------------

def add_all_ichimoku(df: pd.DataFrame,
                     window1: int = 9,
                     window2: int = 26,
                     window3: int = 52) -> pd.DataFrame:
    """
    Run compute_ichimoku() then add_ichimoku_ml_features() in one shot.
    Returns the enriched DataFrame.
    """
    df = compute_ichimoku(df, window1=window1, window2=window2, window3=window3)
    df = add_ichimoku_ml_features(df)
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

    result = add_all_ichimoku(raw)

    ichimoku_cols = ['Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A',
                     'Senkou_Span_B', 'Chikou_Span',
                     'tk_cross', 'price_above_cloud', 'price_below_cloud',
                     'cloud_bullish', 'cloud_thickness', 'chikou_vs_price']

    print("=== Ichimoku Columns Sample (last 5 rows) ===")
    print(result[['Date'] + ichimoku_cols].tail())

    print("\n=== NaN counts ===")
    print(result[ichimoku_cols].isna().sum())

    print("\n=== Data types ===")
    print(result[ichimoku_cols].dtypes)

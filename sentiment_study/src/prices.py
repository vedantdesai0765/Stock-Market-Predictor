"""Price loading + strictly-lagged technical features."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import numpy as np, pandas as pd

PRICE_FILE = price_file("yahoo_daily_prices_long")
_cache = {}

def load_prices(ticker):
    if "all" not in _cache:
        _cache["all"] = (pd.read_parquet(PRICE_FILE) if PRICE_FILE.endswith(".parquet") else pd.read_csv(PRICE_FILE))
        _cache["all"]["date"] = pd.to_datetime(_cache["all"]["date"])
    p = _cache["all"]
    d = p[p.ticker == ticker].set_index("date").sort_index()
    d = d[~d.index.duplicated()]
    # drop stale rows (Yahoo repeats a close on exchange holidays for some series)
    d = d[d.close.notna()]
    d = d[(d.close.diff() != 0) | (d.high != d.low)]
    return d[["open", "high", "low", "close", "adjclose", "volume"]]


def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn)


def technical_features(d):
    """Every feature at row t uses data up to and including close(t) only."""
    c = d.adjclose if d.adjclose.notna().all() else d.close
    r = np.log(c).diff()
    f = pd.DataFrame(index=d.index)
    for k in range(1, 6):
        f[f"ret_l{k}"] = r.shift(k-1)                    # r_t, r_{t-1}, ...
    f["ret_5d"] = r.rolling(5).sum()
    f["ret_20d"] = r.rolling(20).sum()
    f["vol_5"] = r.rolling(5).std()
    f["vol_20"] = r.rolling(20).std()
    f["vol_ratio"] = f.vol_5 / f.vol_20
    f["rsi_14"] = rsi(c)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    f["dist_sma20"] = c / c.rolling(20).mean() - 1
    f["dist_sma50"] = c / c.rolling(50).mean() - 1
    f["dist_sma200"] = c / c.rolling(200).mean() - 1
    sd = c.rolling(20).std(); m = c.rolling(20).mean()
    f["boll_b"] = (c - (m - 2*sd)) / (4*sd)
    if {"high", "low", "open"} <= set(d.columns) and d.high.notna().all():
        f["hl_range"] = (d.high - d.low) / d.close
        f["oc_ret"] = np.log(d.close / d.open)
        f["gap_ret"] = np.log(d.open / d.close.shift(1))
    v = d.volume.replace(0, np.nan)
    if v.notna().mean() > 0.9:
        lv = np.log(v)
        f["vol_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    f["dow"] = d.index.dayofweek
    return f, r


def targets(d, r):
    """Targets at row t refer to the NEXT session (t+1)."""
    t = pd.DataFrame(index=d.index)
    t["r_next"] = r.shift(-1)
    t["up_next"] = (t.r_next > 0).astype(float)
    t.loc[t.r_next.isna(), "up_next"] = np.nan
    t["absr_next"] = t.r_next.abs()
    # 'big move' = |r_{t+1}| above trailing 1-year 70th percentile of |r| (known at t)
    thr = r.abs().rolling(250, min_periods=120).quantile(0.70)
    t["big_next"] = (t.absr_next > thr).astype(float)
    t.loc[t.r_next.isna() | thr.isna(), "big_next"] = np.nan
    # same-day (contemporaneous) target, used ONLY for the leakage demonstration
    t["up_today"] = (r > 0).astype(float)
    # 5-day forward realised volatility
    t["rv5_next"] = r[::-1].rolling(5).std()[::-1].shift(-1)
    return t

"""
Feature construction for the Morning Call model.

This is a self-contained copy of feature set "B" from the research study
(sentiment_study/src/company_study.py -> prices.technical_features, experiment._global_india,
method1.asia_gaps). scripts/train_morning_model.py re-runs the study's walk-forward test with
these functions and must reproduce the published accuracy, which is how we know the live model
sees exactly what the tested model saw.

Row t is an Indian trading day. Every feature on row t is known before the next session opens:
  * the stock's own indicators up to close(t)
  * Nifty indicators up to close(t)
  * the US session(s) that closed after close(t) and before the next Indian open
  * the opening gaps of Japan, Korea, Taiwan and Hong Kong on the morning of the next session
Targets (next session):  gapup_next = open(t+1) > close(t);  up_next = close(t+1) > close(t)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASIA = ["^N225", "^KS11", "^TWII", "^HSI"]
NIFTY, SP500, VIX = "^NSEI", "^GSPC", "^VIX"
MARKET_TICKERS = [NIFTY, SP500, VIX] + ASIA
WARMUP_ROWS = 200          # rows dropped at the start of each stock (200-day average needs them)
PRICE_COLS = ["open", "high", "low", "close", "adjclose", "volume"]


def clean(d: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate and drop stale rows (Yahoo repeats a close on exchange holidays)."""
    d = d.sort_index()
    d = d[~d.index.duplicated()]
    d = d[d.close.notna()]
    d = d[(d.close.diff() != 0) | (d.high != d.low)]
    return d[PRICE_COLS]


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def technical_features(d: pd.DataFrame):
    """Every feature at row t uses data up to and including close(t) only."""
    c = d.adjclose if d.adjclose.notna().all() else d.close
    r = np.log(c).diff()
    f = pd.DataFrame(index=d.index)
    for k in range(1, 6):
        f[f"ret_l{k}"] = r.shift(k - 1)
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
    sd = c.rolling(20).std()
    m = c.rolling(20).mean()
    f["boll_b"] = (c - (m - 2 * sd)) / (4 * sd)
    f["hl_range"] = (d.high - d.low) / d.close
    f["oc_ret"] = np.log(d.close / d.open)
    f["gap_ret"] = np.log(d.open / d.close.shift(1))
    v = d.volume.replace(0, np.nan)
    lv = np.log(v)
    f["vol_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    return f, r


def nifty_features(tdays: pd.DatetimeIndex, nifty: pd.DataFrame) -> pd.DataFrame:
    nr = np.log(nifty.close).diff()
    return pd.DataFrame({
        "nifty_r1": nr,
        "nifty_r5": nr.rolling(5).sum(),
        "nifty_vol20": nr.rolling(20).std(),
        "nifty_rsi": rsi(nifty.close),
    }).reindex(tdays)


def us_features(tdays: pd.DatetimeIndex, sp: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """US sessions dated in [date(t), date(t+1)) close at 01:30/02:30 IST, before the next Indian open."""
    r = np.log(sp.close).diff()
    pos = tdays.searchsorted(sp.index, side="right") - 1
    ok = pos >= 0
    g = pd.Series(r.values[ok], index=tdays[pos[ok]]).groupby(level=0).sum()
    vpos = tdays.searchsorted(vix.index, side="right") - 1
    vok = vpos >= 0
    v = pd.Series(np.log(vix.close.values[vok]), index=tdays[vpos[vok]]).groupby(level=0).last()
    out = pd.DataFrame(index=tdays)
    out["us_ret_overnight"] = g.reindex(tdays).fillna(0)
    out["us_ret_5d"] = out.us_ret_overnight.rolling(5).sum()
    out["vix_log"] = v.reindex(tdays).ffill()
    out["vix_chg"] = out.vix_log.diff()
    return out


def asia_gaps(tdays: pd.DatetimeIndex, px: dict, next_day=None) -> pd.DataFrame:
    """For row t: the opening gap of each Asian market on the calendar date of the next Indian session.

    next_day is the date of the session after the last row (the day being forecast). Without it the
    last row has no "next session" and its gaps are 0 with the open-flag 0, as in the study.
    """
    last = pd.Timestamp(next_day) if next_day is not None else pd.NaT
    nxt = pd.Series(tdays[1:].append(pd.DatetimeIndex([last])), index=tdays)
    out = pd.DataFrame(index=tdays)
    cols = []
    for tk in ASIA:
        d = px[tk]
        g = np.log(d.open / d.close.shift(1))
        nm = "asia_" + tk.strip("^").lower()
        out[nm] = nxt.map(g).values
        out[nm + "_open"] = out[nm].notna().astype(float)
        out[nm] = out[nm].fillna(0.0)
        cols.append(nm)
    out["asia_mean"] = out[cols].mean(axis=1)
    return out


def market_frame(tdays: pd.DatetimeIndex, px: dict, next_day=None) -> pd.DataFrame:
    return (nifty_features(tdays, px[NIFTY])
            .join(us_features(tdays, px[SP500], px[VIX]))
            .join(asia_gaps(tdays, px, next_day)))


def stock_frame(d: pd.DataFrame, px: dict, next_day=None) -> pd.DataFrame:
    """Features and next-session targets for one stock. d = cleaned prices of that stock."""
    f, r = technical_features(d)
    x = f.join(market_frame(d.index, px, next_day))
    gap = np.log(d.open.shift(-1) / d.close)
    rn = r.shift(-1)
    x["gapup_next"] = (gap > 0).astype(float).where(gap.notna() & (gap.abs() > 1e-6))
    x["up_next"] = (rn > 0).astype(float).where(rn.notna())
    return x


TECH = ["ret_l1", "ret_l2", "ret_l3", "ret_l4", "ret_l5", "ret_5d", "ret_20d", "vol_5", "vol_20",
        "vol_ratio", "rsi_14", "macd_hist", "dist_sma20", "dist_sma50", "dist_sma200", "boll_b",
        "hl_range", "oc_ret", "gap_ret", "vol_z"]
MARKET = ["nifty_r1", "nifty_r5", "nifty_vol20", "nifty_rsi",
          "us_ret_overnight", "us_ret_5d", "vix_log", "vix_chg",
          "asia_n225", "asia_n225_open", "asia_ks11", "asia_ks11_open",
          "asia_twii", "asia_twii_open", "asia_hsi", "asia_hsi_open", "asia_mean"]
FEATURES = TECH + MARKET
TARGETS = {"open": "gapup_next", "close": "up_next"}

# Plain-language groups used to explain a forecast
GROUPS = {
    "us": ["us_ret_overnight", "us_ret_5d"],
    "vix": ["vix_log", "vix_chg"],
    "asia": [c for c in MARKET if c.startswith("asia_")],
    "nifty": [c for c in MARKET if c.startswith("nifty_")],
    "stock": TECH,
}

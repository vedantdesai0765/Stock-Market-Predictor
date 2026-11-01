"""
Aggregate scored headlines into one row per TRADING DAY, with explicit,
auditable information-set cut-offs.

Row t of every output describes information available at a stated time and
is paired with the target r_{t+1} = log(close_{t+1}/close_t), unless the
mode is 'sameday' (leakage demonstration only).

Modes
-----
ToI (date-only stamps, IST):
  'forecast' : news dated in [date(t), date(t+1))  -> known by midnight IST
               of day t, i.e. before the next session opens (09:15 IST).
NYT (minute stamps, converted to US/Eastern):
  'close'    : news stamped in (16:00 ET of t-1, 16:00 ET of t]  (strictest)
  'preopen'  : news stamped in (09:00 ET of t,  09:00 ET of t+1]  -- before
               the 09:30 open of the session whose return is the target
  'sameday'  : news with ET calendar date == date(t) (incl. after-close
               write-ups of day t) -- LEAKY when paired with r_t.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import numpy as np, pandas as pd

IN = str(INTERIM)


def _assign_last_le(keys, tdays):
    """map calendar dates -> last trading day <= key"""
    pos = tdays.searchsorted(keys, side="right") - 1
    out = pd.Series(pd.NaT, index=range(len(keys)), dtype="datetime64[ns]")
    ok = pos >= 0
    out[ok] = tdays[pos[ok]]
    return out.values


def _assign_first_ge(keys, tdays):
    pos = tdays.searchsorted(keys, side="left")
    out = pd.Series(pd.NaT, index=range(len(keys)), dtype="datetime64[ns]")
    ok = pos < len(tdays)
    out[ok] = tdays[pos[ok]]
    return out.values


MKT_RE = r"(?i)\b(sensex|nifty|stocks?|shares|wall st|wall street|dow|s\.?&\s?p|nasdaq|markets?|investors?|rally|sell-?off|bull|bear)\b"


def _aggregate(df, key, has_fb, textcol="text"):
    df = df.assign(row=key).dropna(subset=["row"])
    g1 = _agg_core(df, has_fb)
    mk = df[df[textcol].str.contains(MKT_RE, regex=True)]
    g2 = _agg_core(mk, has_fb).add_prefix("m_")
    return g1.join(g2, how="left")


def _agg_core(df, has_fb):
    df = df.copy()
    df["lm_net"] = (df.lm_pos - df.lm_neg) / (df.lm_pos + df.lm_neg + 1)
    agg = {"vader": "mean", "lm_net": "mean"}
    if has_fb:
        df["fb_net"] = df.fb_pos - df.fb_neg
        df["fb_negflag"] = np.where(df.fb_neg.isna(), np.nan, (df.fb_neg > 0.5).astype(float))
        df["fb_posflag"] = np.where(df.fb_pos.isna(), np.nan, (df.fb_pos > 0.5).astype(float))
        agg.update({"fb_net": "mean", "fb_negflag": "mean", "fb_posflag": "mean"})
    g = df.groupby("row").agg(agg)
    g["n_news"] = df.groupby("row").size()
    return g


def add_dynamics(s, cols):
    """smoothed levels, changes and 'surprise' z-scores -- all backward-looking"""
    out = s.copy()
    for c in cols:
        x = s[c]
        out[f"{c}_ema3"] = x.ewm(span=3).mean()
        out[f"{c}_ema10"] = x.ewm(span=10).mean()
        m, sd = x.rolling(60, min_periods=20).mean(), x.rolling(60, min_periods=20).std()
        out[f"{c}_z60"] = (x - m) / sd
        out[f"{c}_chg"] = x - x.shift(1)
    out["n_news_log"] = np.log1p(s.n_news)
    out["n_news_z60"] = (out.n_news_log - out.n_news_log.rolling(60, min_periods=20).mean()) / out.n_news_log.rolling(60, min_periods=20).std()
    return out.drop(columns=["n_news"])


def toi_daily(tdays, has_fb=True, subset="market"):
    f = first(f"{IN}/toi_market_scored_fb.parquet" if has_fb else f"{IN}/toi_market_scored.parquet",
              PROCESSED / "toi_market_headlines_scored.parquet")
    d = pd.read_parquet(f)
    if subset == "business":
        d = d[d.is_business_cat]
    key = _assign_last_le(d.date.values, tdays)
    return _finish(_aggregate(d, key, has_fb), tdays)


def _finish(g, tdays):
    g = g.reindex(tdays)
    for c in ["n_news", "m_n_news"]:
        g[c] = g[c].fillna(0)
    base = [c for c in g.columns if c not in ("n_news", "m_n_news")]
    g[base] = g[base].fillna(0.0)
    out = add_dynamics(g, base)
    out["m_n_news_log"] = np.log1p(g.m_n_news)
    return out.drop(columns=["m_n_news"])


def nyt_daily(tdays, mode="preopen", has_fb=True, text="full"):
    f = first(f"{IN}/nyt_fin_scored_fb.parquet" if has_fb else f"{IN}/nyt_fin_scored.parquet",
              PROCESSED / "nyt_finance_headlines_scored.parquet")
    d = pd.read_parquet(f)
    if "ts_et" not in d:
        d["ts_et"] = d.ts_utc.dt.tz_convert("America/New_York")
    if text == "headline":
        d["vader"] = d.vader_head
    et = d.ts_et.dt.tz_localize(None)
    if mode == "close":
        k = (et + pd.Timedelta(hours=8)).dt.normalize()      # after 16:00 -> next day
        key = _assign_first_ge(k.values, tdays)
    elif mode == "preopen":
        k = (et - pd.Timedelta(hours=9)).dt.normalize()      # before 09:00 -> previous day
        key = _assign_last_le(k.values, tdays)
    elif mode == "sameday":
        key = _assign_first_ge(et.dt.normalize().values, tdays)
    else:
        raise ValueError(mode)
    d["text"] = d.headline
    return _finish(_aggregate(d, key, has_fb), tdays)

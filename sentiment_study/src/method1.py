"""
Method 1: forecast the Indian index OPENING GAP at 09:00 IST.

Row t (an Indian trading day) -> target: open(t+1) > close(t).
Information available at 09:00 IST on day t+1:
  T  technical features up to close(t)
  G  US session dated t (closes 01:30/02:30 IST), its 5-day sum, VIX
  A  opening gaps of Asian markets on date(t+1): Nikkei/KOSPI (open 05:30 IST),
     Taiex (06:30 IST), Hang Seng (07:00 IST) = log(open / previous close)
  S  overnight news sentiment: Times of India headlines dated in [date(t), date(t+1))
     and NYT headlines stamped after the Indian close (10:00 UTC) and before 09:00 IST
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import prices
prices.PRICE_FILE = price_file("yahoo_prices_v2"); prices._cache.clear()
from prices import load_prices, technical_features
import experiment as E
from sentiment_daily import _aggregate, _finish, _assign_last_le, toi_daily

IN = str(INTERIM)
ASIA = ["^N225", "^KS11", "^TWII", "^HSI"]


def asia_gaps(tdays):
    """for row t, the Asian opening gap on the calendar date of t+1"""
    nxt = pd.Series(tdays[1:].append(pd.DatetimeIndex([pd.NaT])), index=tdays)
    out = pd.DataFrame(index=tdays)
    for tk in ASIA:
        d = load_prices(tk); g = np.log(d.open / d.close.shift(1))
        nm = "asia_" + tk.strip("^").lower()
        out[nm] = nxt.map(g).values
        out[nm + "_open"] = out[nm].notna().astype(float)
        out[nm] = out[nm].fillna(0.0)
    cols = [f"asia_{t.strip('^').lower()}" for t in ASIA]
    out["asia_mean"] = out[cols].mean(axis=1)
    return out


def nyt_overnight_india(tdays):
    d = pd.read_parquet(first(f"{IN}/nyt_fin_scored_fb.parquet", PROCESSED / "nyt_finance_headlines_scored.parquet"))
    u = d.ts_utc.dt.tz_localize(None)
    h = u.dt.hour + u.dt.minute / 60
    day = u.dt.normalize()
    key = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    key[h >= 10] = day[h >= 10]                                   # after the Indian close
    key[h < 3.5] = day[h < 3.5] - pd.Timedelta(days=1)            # before 09:00 IST next morning
    d = d[key.notna()]; key = key[key.notna()]
    d["text"] = d.headline
    rows = _assign_last_le(key.values, tdays)
    return _finish(_aggregate(d, rows, True), tdays).add_prefix("nyt_")


def build(index="^BSESN", start="2001-01-02", end="2022-03-30", use_toi=True):
    d = load_prices(index).loc[start:end]
    f, r = technical_features(d)
    g = E._global_india(d.index)
    a = asia_gaps(d.index)
    sn = nyt_overnight_india(d.index)
    parts = [f, g, a, sn]
    s_cols = [c for c in sn.columns if any(c == "nyt_" + k for k in E.SENT_COLS)]
    if use_toi:
        st = toi_daily(d.index).add_prefix("toi_"); parts.append(st)
        s_cols += [c for c in st.columns if any(c == "toi_" + k for k in E.SENT_COLS)]
    df = pd.concat(parts, axis=1)
    gap = np.log(d.open.shift(-1) / d.close)
    df["gapup_next"] = (gap > 0).astype(float).where(gap.notna() & (gap.abs() > 1e-6))
    df = df.iloc[200:]
    return df, list(f.columns), list(g.columns), list(a.columns), s_cols


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    tag, index, start, end, use_toi, fty = cfg.get("tag", "M1S"), cfg.get("index", "^BSESN"), cfg.get("start", "2001-01-02"), cfg.get("end", "2022-03-30"), cfg.get("use_toi", True), cfg.get("fty", 2009)
    df, T, G, A, S = build(index, start, end, use_toi)
    print(tag, index, df.shape, "T", len(T), "G", len(G), "A", len(A), "S", len(S), flush=True)
    sets = {"T": T, "T+G": T + G, "T+G+S": T + G + S, "T+G+A": T + G + A, "T+G+A+S": T + G + A + S}
    for m in cfg.get("models", ["LogReg", "RandomForest", "GradBoost", "XGBoost", "LightGBM"]):
        for fs, feats in sets.items():
            o = E.walk_forward(df, feats, "gapup_next", E.clf_zoo()[m], fty)
            k = f"{tag}__gapup_next__{m}__{fs}"; o.to_parquet(f"{E.OUT}/{k}.parquet")
            print(f"{k:45s} n={len(o)} acc={((o.p > .5) == (o.y > .5)).mean():.4f} base={o.y.mean():.4f}", flush=True)

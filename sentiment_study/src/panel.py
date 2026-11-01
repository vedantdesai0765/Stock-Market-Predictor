"""
Company-level panel study (FNSPID Nasdaq-100 news, 38 stocks, 2012-2023).

Row (stock i, day t): technical features of i up to close t, market features
up to close t, and company news dated in [date(t), date(t+1)) -- i.e. known by
midnight ET of day t, before the next session opens. Targets refer to t+1,
except up_today (leakage demonstration).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import sys, os, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from prices import technical_features, rsi
from sentiment_daily import _assign_last_le, add_dynamics
from experiment import clf_zoo, OUT

IN = str(INTERIM)
PF = price_file("yahoo_ndx_prices_long")


def load(tk, P):
    d = P[P.ticker == tk].set_index("date").sort_index()
    d = d[~d.index.duplicated()].dropna(subset=["close"])
    return d[["open", "high", "low", "close", "adjclose", "volume"]]


def company_sent(news, tdays):
    n = news.copy()
    n["row"] = _assign_last_le(n.date.values, tdays)
    n = n.dropna(subset=["row"])
    n["lm_net"] = (n.lm_pos - n.lm_neg) / (n.lm_pos + n.lm_neg + 1)
    n["fb_net"] = n.fb_pos - n.fb_neg
    n["fb_negflag"] = (n.fb_neg > .5).astype(float); n["fb_posflag"] = (n.fb_pos > .5).astype(float)
    g = n.groupby("row").agg(fb_net=("fb_net", "mean"), fb_negflag=("fb_negflag", "mean"),
                             fb_posflag=("fb_posflag", "mean"), vader=("vader", "mean"),
                             lm_net=("lm_net", "mean"), n_news=("fb_net", "size"),
                             fb_min=("fb_net", "min"), fb_max=("fb_net", "max"))
    g = g.reindex(tdays); g["n_news"] = g.n_news.fillna(0)
    base = ["fb_net", "fb_negflag", "fb_posflag", "vader", "lm_net", "fb_min", "fb_max"]
    g[base] = g[base].fillna(0.0)
    out = add_dynamics(g, ["fb_net", "vader", "lm_net"])
    out["has_news"] = (g.n_news > 0).astype(float).values
    for c in ["fb_negflag", "fb_posflag", "fb_min", "fb_max"]: out[c] = g[c].values
    return out


def build():
    P = pd.read_parquet(PF) if PF.endswith(".parquet") else pd.read_csv(PF)
    P["date"] = pd.to_datetime(P["date"])
    news = pd.read_parquet(first(f"{IN}/fnspid_scored_fb.parquet", PROCESSED / "fnspid_ndx_titles_scored.parquet"))
    mk = load("QQQ", P); mr = np.log(mk.adjclose).diff()
    mkt = pd.DataFrame({"mkt_r1": mr, "mkt_r5": mr.rolling(5).sum(), "mkt_vol20": mr.rolling(20).std(),
                        "mkt_rsi": rsi(mk.adjclose)})
    rows = []
    for tk, nw in news.groupby("ticker"):
        if tk not in set(P.ticker): continue
        d = load(tk, P).loc["2010-01-01":"2023-12-31"]
        f, r = technical_features(d)
        f = f.drop(columns=["dow"])
        s = company_sent(nw, d.index)
        t = pd.DataFrame(index=d.index)
        t["r_next"] = r.shift(-1)
        t["up_next"] = (t.r_next > 0).astype(float)
        ex = r - mr.reindex(d.index)
        t["upx_next"] = (ex.shift(-1) > 0).astype(float)                     # beats the market tomorrow
        gap = np.log(d.open.shift(-1) / d.close)
        t["gapup_next"] = (gap > 0).astype(float)
        thr = r.abs().rolling(250, min_periods=120).quantile(0.70)
        t["big_next"] = (r.shift(-1).abs() > thr).astype(float)
        t["up_today"] = (r > 0).astype(float)
        for c in ["up_next", "upx_next", "gapup_next", "big_next"]:
            t.loc[t.r_next.isna(), c] = np.nan
        x = f.join(mkt.reindex(d.index)).join(s).join(t)
        x["ticker"] = tk
        rows.append(x.loc["2012-03-01":])
    df = pd.concat(rows).reset_index().rename(columns={"index": "date"})
    tech = list(f.columns); glob_ = list(mkt.columns)
    sent = [c for c in s.columns]
    return df, tech, glob_, sent


def walk(df, feats, target, fn, years, embargo_days=5):
    d = df.dropna(subset=[target]); preds = []
    for y in years:
        cut = pd.Timestamp(f"{y}-01-01")
        tr = d[d.date < cut - pd.Timedelta(days=embargo_days)]
        te = d[d.date.dt.year == y]
        m = fn(); m.fit(tr[feats], tr[target].astype(int))
        p = m.predict_proba(te[feats])[:, 1]
        preds.append(pd.DataFrame({"y": te[target].values, "p": p, "maj": float(tr[target].mean() > .5),
                                   "has_news": te.has_news.values, "ticker": te.ticker.values},
                                  index=pd.MultiIndex.from_arrays([te.date, te.ticker])))
    return pd.concat(preds)


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    df, tech, glob_, sent = build()
    tl = [c + "_lag1" for c in tech]; gl = [c + "_lag1" for c in glob_]
    df[tl + gl] = df.groupby("ticker")[tech + glob_].shift(1).values
    print("panel rows", len(df), "news-day share", df.has_news.mean().round(3), flush=True)
    zoo = clf_zoo()
    models = cfg.get("models", ["LogReg", "LightGBM", "XGBoost", "RandomForest", "GradBoost"])
    targets_ = cfg.get("targets", ["up_next", "upx_next", "gapup_next", "big_next", "up_today"])
    years = list(range(2016, 2024))
    for tg in targets_:
        for m in models:
            for fs in ["T+G", "T+G+S"]:
                if tg == "up_today":
                    feats = tl + gl + (sent if fs == "T+G+S" else [])
                else:
                    feats = tech + glob_ + (sent if fs == "T+G+S" else [])
                t0 = time.time()
                o = walk(df, feats, tg, zoo[m], years)
                key = f"PANEL__{tg}__{m}__{fs}"
                o.reset_index(drop=True).set_index(pd.RangeIndex(len(o))).to_parquet(f"{OUT}/{key}.parquet")
                acc = ((o.p > .5) == (o.y > .5)).mean(); nd = o.has_news > 0
                accn = ((o.p[nd] > .5) == (o.y[nd] > .5)).mean()
                print(f"{key:45s} n={len(o)} acc={acc:.4f} acc_newsdays={accn:.4f} {time.time()-t0:.0f}s", flush=True)

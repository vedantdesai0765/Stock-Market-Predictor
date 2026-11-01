"""
Company-targeted sentiment study (10 project stocks).

Study TOI  : Times of India headlines that name the company (2001-2022-03), date-only stamps.
             headline dated d -> row t = last trading day <= d  (all of it is known before the t+1 open)
Study NSE  : the company's own NSE filings (2008-2025), exact timestamps.
             filing -> row t if disseminated after 15:30 IST of t and before 09:00 IST of t+1
Targets (on t+1): gapup_next = open(t+1) > close(t);  up_next = close(t+1) > close(t)
Feature sets
  B      stock technicals + market (Nifty technicals, S&P 500 overnight, VIX, Asian opens)
  M      market-wide news sentiment (ToI market headlines, FinBERT) - the old, non-targeted feature
  Cfb    company headlines scored with plain FinBERT (tone of the whole headline)
  Cent   company headlines scored with the entity-level model (sentiment TOWARDS that company)
  Cplc   placebo: Cent shifted by 252 trading days within each stock
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, ROOT, first, price_file

import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import prices
prices.PRICE_FILE = price_file("yahoo_prices_v2"); prices._cache.clear()
from prices import load_prices, technical_features, rsi
from sentiment_daily import _assign_last_le, toi_daily
import experiment as E
from method1 import asia_gaps
from method4 import overnight_window, walk
from analyze import block_boot, _acc, _auc
from scipy.stats import binomtest

IN = str(INTERIM); REP = str(REPORTS)
OUT = str(REPORTS / "oof_company"); os.makedirs(OUT, exist_ok=True)
RAW13 = str(ROOT.parent / "data" / "raw")      # the main project's raw price files
STOCKS = {"RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS", "HDFCBANK": "HDFCBANK.NS",
          "SBI": "SBIN.NS", "ITC": "ITC.NS", "ADANIENT": "ADANIENT.NS", "VEDL": "VEDL.NS",
          "ADANIGREEN": "ADANIGREEN_raw.csv", "ETERNAL": "ETERNAL_raw.csv"}
NSE_SYM = {"SBI": "SBIN"}
START = {"TCS": "2004-08-25", "ADANIENT": "2004-07-01", "ADANIGREEN": "2018-06-18", "ETERNAL": "2021-07-23"}


def stock_prices(k):
    src = STOCKS[k]
    if src.endswith(".csv"):
        d = pd.read_csv(f"{RAW13}/{src}", parse_dates=["Date"]).set_index("Date").sort_index()
        d.columns = [c.lower() for c in d.columns]; d["adjclose"] = d.close
        d = d[(d.close > 0) & (d.open > 0)]
    else:
        d = load_prices(src)
    return d.loc[START.get(k, "2000-01-01"):]


def market_frame(tdays):
    nf = load_prices("^NSEI"); nr = np.log(nf.close).diff()
    mkt = pd.DataFrame({"nifty_r1": nr, "nifty_r5": nr.rolling(5).sum(), "nifty_vol20": nr.rolling(20).std(),
                        "nifty_rsi": rsi(nf.close)}).reindex(tdays)
    return mkt.join(E._global_india(tdays)).join(asia_gaps(tdays))


def daily(items, tdays, pre, neg, pos):
    """aggregate scored items (column 'row' = trading day) into daily company features"""
    it = items.dropna(subset=["row"]).copy()
    it["net"] = it[pos] - it[neg]
    g = it.groupby("row")
    f = pd.DataFrame({"n": g.size(), "net": g.net.mean(), "net_sum": g.net.sum(), "net_min": g.net.min(),
                      "neg_sh": g[neg].apply(lambda x: (x > .5).mean()),
                      "pos_sh": g[pos].apply(lambda x: (x > .5).mean())}).reindex(tdays).fillna(0.0)
    f["n_log"] = np.log1p(f.pop("n"))
    f["net_ema5"] = f.net_sum.ewm(halflife=5).mean()
    f["net_ema20"] = f.net_sum.ewm(halflife=20).mean()
    f["net_chg"] = f.net_ema5 - f.net_ema5.shift(5)
    return f.add_prefix(pre + "_")


def build(study):
    toi = pd.read_parquet(first(f"{IN}/toi_company_scored.parquet", PROCESSED / "toi_company_scored.parquet"))
    fil = pd.read_parquet(first(f"{IN}/nse_company_scored.parquet", PROCESSED / "nse_company_scored.parquet"))
    nf = load_prices("^NSEI")
    frames = []; cols = {}
    msent = toi_daily(nf.index, has_fb=True)          # market-wide sentiment (non-targeted)
    mcols = [c for c in msent.columns if c in E.SENT_COLS]
    for k in STOCKS:
        d = stock_prices(k)
        if study == "TOI": d = d.loc[:"2022-03-31"]
        else: d = d.loc["2008-06-01":"2025-12-31"]
        if len(d) < 300: continue
        f, r = technical_features(d); f = f.drop(columns=["dow"], errors="ignore")
        T = [c for c in f.columns]
        x = f.join(market_frame(d.index))
        if study == "TOI":
            it = toi[toi.stock == k].copy(); it["row"] = _assign_last_le(it.date.values, d.index)
            ce = daily(it, d.index, "ce", "ent_neg", "ent_pos"); cf = daily(it, d.index, "cf", "fbh_neg", "fbh_pos")
            x = x.join(msent[mcols].reindex(d.index))
        else:
            it = fil[fil.symbol == NSE_SYM.get(k, k)].copy(); it["row"] = overnight_window(it, d.index)
            ce = daily(it, d.index, "ce", "ent_neg", "ent_pos"); cf = daily(it, d.index, "cf", "fb_neg", "fb_pos")
        cp = ce.shift(252).fillna(0.0); cp.columns = [c.replace("ce_", "cp_") for c in cp.columns]
        x = x.join(ce).join(cf).join(cp)
        x["news_day"] = (x.ce_n_log > 0).astype(float)
        gap = np.log(d.open.shift(-1) / d.close); rn = r.shift(-1)
        x["gapup_next"] = (gap > 0).astype(float).where(gap.notna() & (gap.abs() > 1e-6))
        x["up_next"] = (rn > 0).astype(float).where(rn.notna())
        x["ticker"] = k
        frames.append(x.iloc[200:])                       # warm-up for SMA200
        cols = {"T": T, "MKT": list(market_frame(d.index[:5]).columns), "M": mcols,
                "Cent": list(ce.columns), "Cfb": list(cf.columns), "Cplc": list(cp.columns)}
    df = pd.concat(frames).reset_index().rename(columns={"index": "date", "Date": "date"})
    return df, cols


def paired(a, b, mask=None):
    if mask is not None: a, b = a[mask], b[mask]
    y = a.y.values; pa, pb = a.p.values, b.p.values
    ca = (pa > .5) == (y > .5); cb = (pb > .5) == (y > .5)
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    bacc = block_boot(pa, pb, y, _acc, n=1000); bauc = block_boot(pa, pb, y, _auc, n=500)
    return {"n": len(y), "acc_a": ca.mean(), "acc_b": cb.mean(), "d_acc": cb.mean() - ca.mean(),
            "d_acc_lo": np.quantile(bacc, .025), "d_acc_hi": np.quantile(bacc, .975),
            "p_mcnemar": binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0,
            "auc_a": _auc(pa, y), "auc_b": _auc(pb, y), "d_auc": _auc(pb, y) - _auc(pa, y),
            "d_auc_lo": np.nanquantile(bauc, .025), "d_auc_hi": np.nanquantile(bauc, .975),
            "p_boot_auc": (bauc <= 0).mean()}


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    study = cfg.get("study", "TOI")
    df, C = build(study)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df.to_parquet(f"{IN}/company_panel_{study}.parquet")
    B = C["T"] + C["MKT"]
    sets = {"B": B, "B+Cfb": B + C["Cfb"], "B+Cent": B + C["Cent"], "B+Cplc": B + C["Cplc"]}
    if study == "TOI":
        sets.update({"B+M": B + C["M"], "B+M+Cent": B + C["M"] + C["Cent"]})
        years = list(range(2007, 2023))
    else:
        years = list(range(2013, 2026))
    print(study, df.shape, "news-day share", df.news_day.mean().round(3),
          df.groupby("ticker").news_day.mean().round(3).to_dict(), flush=True)
    zoo = E.clf_zoo()
    for tg in cfg.get("targets", ["gapup_next", "up_next"]):
        for m in cfg.get("models", ["LogReg", "XGBoost", "LightGBM", "RandomForest"]):
            for fs, feats in sets.items():
                k = f"{study}__{tg}__{m}__{fs}"
                if os.path.exists(f"{OUT}/{k}.parquet"): continue
                o = walk(df, feats, tg, zoo[m], years)
                o["news_day"] = df.dropna(subset=[tg]).set_index(["date", "ticker"]).news_day.reindex(
                    pd.MultiIndex.from_arrays([o.date, o.ticker])).values
                o.to_parquet(f"{OUT}/{k}.parquet")
                nd = o.news_day > 0
                print(f"{k:42s} n={len(o)} acc={((o.p > .5) == (o.y > .5)).mean():.4f} "
                      f"news-days acc={((o.p[nd] > .5) == (o.y[nd] > .5)).mean():.4f} (n={nd.sum()})", flush=True)

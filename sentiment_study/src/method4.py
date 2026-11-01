"""
Method 4: company event days from official NSE corporate announcements (exact timestamps).

Row (stock i, day t). Overnight window = filings disseminated after the 15:30 IST close of day t
and up to 09:00 IST of day t+1 (plus weekends/holidays in between). These are not in close(t).
Event rows = rows whose overnight window holds at least one NON-routine filing.
Targets on day t+1: opening gap up, close-to-close up, beats Nifty (excess return up).
Feature sets:
  B  = stock technicals (T) + market/global (US overnight return, VIX, Asian opens, Nifty technicals)
  E  = event metadata: counts of filings by type (results, press release, rating, M&A, news clarification ...)
  S  = sentiment of the overnight filings (FinBERT, VADER, Loughran-McDonald)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import sys, os, re, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import prices
prices.PRICE_FILE = price_file("yahoo_prices_v2"); prices._cache.clear()
from prices import load_prices, technical_features, rsi
from sentiment_daily import _assign_last_le
import experiment as E
from method1 import asia_gaps

IN = str(INTERIM)
YH = {"M&M": "M&M.NS", "BAJAJ-AUTO": "BAJAJ-AUTO.NS"}

CATS = {
    "results": r"financial result|outcome of board meeting|results",
    "press": r"press release|media release",
    "clarif": r"clarification|news verification|spurt|price movement",
    "rating": r"credit rating",
    "deal": r"acquisition|amalgamation|merger|scheme of arrangement|joint venture|disinvestment|stake",
    "capital": r"buyback|buy back|dividend|bonus|split|rights issue|fund raising|qip|preferential",
    "mgmt": r"resignation|appointment|change in director|cessation|ceo|managing director",
    "legal": r"litigation|dispute|penalty|order|fraud|default|investigation|raid|sebi order",
}
ROUTINE = r"analyst|institutional investor|con\. call|conference call|loss of share|investor presentation|schedule of|insider trading|related party|issue of securities|transfer agent|registrar|postal ballot|e-voting|annual report|secretarial|esop|esps|allotment of|trading window|certificate|shareholders meeting|agm|investor complaint|compliance|newspaper|loss of share|duplicate share|record date|book closure|copy of|intimation of|updates on|closure of|statement of deviation|disclosure under sebi \(substantial|regulation 74|regulation 40|regulation 7|xbrl"


def load_announcements():
    a = pd.read_parquet(first(f"{IN}/nse_ann_scored.parquet", PROCESSED / "nse_announcements_scored.parquet"))
    return a


def prep_raw(path):
    j = json.load(open(path))
    a = pd.DataFrame(j).drop_duplicates(subset=["symbol", "an_dt", "desc", "text"])
    raw = a.dis.where(a.dis.notna() & ~a.dis.isin(["", "-"]), a.an_dt)
    ts = pd.to_datetime(raw, format="mixed", dayfirst=True, errors="coerce")
    a["ts"] = ts
    a = a[a.ts.notna() & (a.ts.dt.year >= 2008)]
    a = a[(a.ts.dt.hour + a.ts.dt.minute) > 0]            # drop date-only stamps
    low = (a.desc.fillna("") + " " + a.text.fillna("")).str.lower()
    for k, pat in CATS.items():
        a[f"cat_{k}"] = low.str.contains(pat, regex=True).astype(float)
    a["routine"] = (low.str.contains(ROUTINE, regex=True) & (a[[f"cat_{k}" for k in CATS]].sum(axis=1) == 0)).astype(float)
    # informative part: quoted title if any, else the text after the boilerplate opener
    q = a.text.fillna("").str.extract(r'["“](.+?)["”]')[0]
    body = a.text.fillna("").str.replace(r"^.*?has informed the Exchange (?:regarding|that|about)?\s*", "", regex=True)
    a["clean"] = np.where(q.notna() & (q.str.len() > 12), q, body).astype(str)
    a["clean"] = (a.desc.fillna("") + ": " + a["clean"]).str.slice(0, 400)
    return a.reset_index(drop=True)


def score(a):
    a = a[a.routine == 0].reset_index(drop=True)
    from score_lexicon import run
    a["vader"], a["lm_pos"], a["lm_neg"] = run(a.clean.tolist())
    from score_finbert import score as fb
    p = fb(a.clean.tolist(), f"{IN}/finbert_nse.npy")
    a["fb_pos"], a["fb_neg"] = p[:, 0], p[:, 1]
    return a


def overnight_window(a, tdays):
    """filing -> row t: disseminated in (15:30 IST of t, 09:00 IST of t+1]; intraday filings -> None"""
    ts = a.ts; h = ts.dt.hour + ts.dt.minute / 60
    key = pd.Series(pd.NaT, index=a.index, dtype="datetime64[ns]")
    after = h >= 15.5; before = h < 9.0
    key[after] = ts[after].dt.normalize()
    key[before] = ts[before].dt.normalize() - pd.Timedelta(days=1)
    # weekend / holiday filings at any hour: belong to the last trading day before them
    is_td = ts.dt.normalize().isin(tdays)
    key[~is_td] = ts[~is_td].dt.normalize()
    m = key.notna()
    rows = pd.Series(pd.NaT, index=a.index, dtype="datetime64[ns]")
    rows[m] = _assign_last_le(key[m].values, tdays)
    return rows


def stock_frame(sym, a, mkt):
    tk = YH.get(sym, sym + ".NS")
    d = load_prices(tk).loc["2007-01-01":"2025-12-31"]
    if len(d) < 500: return None
    f, r = technical_features(d); f = f.drop(columns=["dow"])
    aa = a[a.symbol == sym].copy()
    aa["row"] = overnight_window(aa, d.index)
    aa = aa.dropna(subset=["row"])
    inf = aa[aa.routine == 0].copy()
    inf["lm_net"] = (inf.lm_pos - inf.lm_neg) / (inf.lm_pos + inf.lm_neg + 1)
    inf["fb_net"] = inf.fb_pos - inf.fb_neg
    g = inf.groupby("row")
    ev = pd.DataFrame({"n_filings": g.size()})
    for k in CATS: ev[f"n_{k}"] = g[f"cat_{k}"].sum()
    sen = pd.DataFrame({"fb_net": g.fb_net.mean(), "fb_min": g.fb_net.min(), "fb_max": g.fb_net.max(),
                        "fb_neg_share": g.fb_neg.apply(lambda x: (x > .5).mean()),
                        "fb_pos_share": g.fb_pos.apply(lambda x: (x > .5).mean()),
                        "vader": g.vader.mean(), "lm_net": g.lm_net.mean()})
    # sentiment of results-type filings and of press releases separately
    for k in ["results", "press", "clarif"]:
        sub = inf[inf[f"cat_{k}"] > 0]
        sen[f"fb_net_{k}"] = sub.groupby("row").fb_net.mean()
    x = f.join(mkt.reindex(d.index)).join(ev.reindex(d.index)).join(sen.reindex(d.index))
    evc = list(ev.columns); sc = list(sen.columns)
    x[evc] = x[evc].fillna(0); x[sc] = x[sc].fillna(0)
    x["event"] = (x.n_filings > 0).astype(float)
    gap = np.log(d.open.shift(-1) / d.close); rn = r.shift(-1)
    nifty_r = mkt["nifty_r1"].shift(-1).reindex(d.index)
    x["gapup_next"] = (gap > 0).astype(float).where(gap.notna() & (gap.abs() > 1e-6))
    x["up_next"] = (rn > 0).astype(float).where(rn.notna())
    x["upx_next"] = ((rn - nifty_r) > 0).astype(float).where(rn.notna() & nifty_r.notna())
    x["absgap_next"] = gap.abs()
    x["ticker"] = sym
    return x.loc["2008-06-01":], list(f.columns), evc, sc


def build():
    a = load_announcements()
    nf = load_prices("^NSEI"); nr = np.log(nf.close).diff()
    mkt = pd.DataFrame({"nifty_r1": nr, "nifty_r5": nr.rolling(5).sum(), "nifty_vol20": nr.rolling(20).std(), "nifty_rsi": rsi(nf.close)})
    gl = E._global_india(nf.index); asia = asia_gaps(nf.index)
    mkt = mkt.join(gl).join(asia)
    frames = []; T = EV = SC = None
    for sym in sorted(a.symbol.unique()):
        out = stock_frame(sym, a, mkt)
        if out is None: continue
        x, T, EV, SC = out; frames.append(x)
    df = pd.concat(frames).reset_index().rename(columns={"index": "date", "Date": "date"})
    return df, T, list(mkt.columns), EV, SC


def walk(df, feats, target, fn, years, mask=None):
    d = df.dropna(subset=[target]); preds = []
    for y in years:
        tr = d[d.date < pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=7)]
        te = d[d.date.dt.year == y]
        if mask is not None:
            tr = tr[mask(tr)]; te = te[mask(te)]
        if len(te) == 0 or tr[target].nunique() < 2: continue
        m = fn(); m.fit(tr[feats], tr[target].astype(int))
        preds.append(pd.DataFrame({"y": te[target].values, "p": m.predict_proba(te[feats])[:, 1],
                                   "maj": float(tr[target].mean() > .5), "ticker": te.ticker.values,
                                   "date": te.date.values}))
    o = pd.concat(preds, ignore_index=True)
    return o


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    if cfg.get("prep"):
        a = prep_raw(cfg["prep"]); print("filings", len(a), "routine share", a.routine.mean().round(3), flush=True)
        a = score(a); a.to_parquet(f"{IN}/nse_ann_scored.parquet"); print("scored", flush=True); sys.exit()
    df, T, G, EV, SC = build()
    print("panel", df.shape, "event share", df.event.mean().round(3), "stocks", df.ticker.nunique(), flush=True)
    df.to_parquet(f"{IN}/method4_panel.parquet")
    years = list(range(2013, 2026))
    ev_mask = lambda z: z.event > 0
    res_mask = lambda z: z.n_results > 0
    sets = {"B": T + G, "B+E": T + G + EV, "B+E+S": T + G + EV + SC}
    for subset, mk in [("event", ev_mask), ("results", res_mask)]:
        for tg in ["gapup_next", "up_next", "upx_next"]:
            for m in cfg.get("models", ["LogReg", "RandomForest", "GradBoost", "XGBoost", "LightGBM"]):
                for fs, feats in sets.items():
                    o = walk(df, feats, tg, E.clf_zoo()[m], years, mk)
                    k = f"M4{subset}__{tg}__{m}__{fs}"; o.to_parquet(f"{E.OUT}/{k}.parquet")
                    print(f"{k:45s} n={len(o)} acc={((o.p > .5) == (o.y > .5)).mean():.4f} base={o.y.mean():.4f}", flush=True)

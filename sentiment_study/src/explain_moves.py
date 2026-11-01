"""
Explaining price moves with news sentiment (explanatory, not predictive).

Question: on the day a stock (or the Sensex) moves, does the tone of that day's news line up with the move?

News -> trading day it can explain
  Times of India headline dated d    -> first trading day >= d   (weekend news explains Monday)
  NSE filing at time ts              -> that day if before the 15:30 IST close, else the next trading day
Price-report headlines ("Sensex down 200 points", "Infosys shares fall 3%") describe the move instead of
explaining it, so every statistic is also reported with those headlines removed.

Outputs (read by the dashboard's "Why did it move?" page):
  ../data/processed/news/<KEY>_news_items.csv   every headline / filing with its company-targeted score
  ../reports/explain/lead_lag.csv               correlation of sentiment with returns k days before/after
  ../reports/explain/quintiles.csv              mean return by sentiment quintile, same day vs next day
  ../reports/explain/event_study.csv            cumulative abnormal return around strong news days
  ../reports/explain/big_moves_coverage.csv     share of the largest moves with matching / opposite / no news
  ../reports/explain/big_moves_top.csv          the largest moves with their top headline
  ../reports/explain/same_day_accuracy.csv      up/down accuracy of the SAME day, with vs without sentiment
  ../reports/explain/summary.csv                one row per scope, for the paper's main table

    python src/explain_moves.py
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, ROOT, first, price_file
import sys, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, binomtest
import prices
prices.PRICE_FILE = price_file("yahoo_prices_v2"); prices._cache.clear()
from prices import load_prices, technical_features
from sentiment_daily import _assign_first_ge
from method4 import walk
import experiment as E

MAIN = ROOT.parent
sys.path.insert(0, str(MAIN / "src"))
import config                                   # the main project's stock registry

OUT = REPORTS / "explain"; OUT.mkdir(parents=True, exist_ok=True)
NEWS = MAIN / "data" / "processed" / "news"; NEWS.mkdir(parents=True, exist_ok=True)
NSE_SYM = {"SBI": "SBIN"}
START, END = "2001-01-01", "2022-03-31"         # Times of India archive

PRICE_REPORT = (r"\b(sensex|nifty|bse|nse)\b|\bpoints?\b|"
                r"\b(stocks?|shares?|scrip|m-?cap|market cap\w*|markets?|indices|index)\b.{0,30}\b("
                r"fall|falls|fell|rise|rises|rose|gain|gains|gained|drop|drops|dropped|slip|slips|slump|slumps|"
                r"surge|surges|surged|rally|rallies|plunge|plunges|tumble|tumbles|crash|soar|soars|jump|jumps|"
                r"climb|climbs|end|ends|close|closes|open|opens|hit|hits|high|low|up|down|crashes|crashed|plunged|"
                r"plummets?|tanks?|tanked|sinks?|sank|slides?|slid|zooms?|skyrockets?|nosedives?|dips?|dipped|"
                r"rallied|soared|jumped|climbed|tumbled|slumped|surged|spurts?|skids?)\b|"
                r"\b(stocks?|shares?|scrip)\b.{0,40}\d+(\.\d+)?\s?(%|per ?cent)")


def is_price_report(text):
    return text.fillna("").str.lower().str.contains(PRICE_REPORT, regex=True)


def stock_prices(key):
    s = config.get_stock(key)
    d = pd.read_csv(s.raw_path, parse_dates=["Date"]).set_index("Date").sort_index()
    d = d[(d.Close > 0)]
    r = np.log(d.Close).diff() * 100
    return d, r


def market_return():
    s = load_prices("^BSESN"); return np.log(s.close).diff() * 100


def filing_trading_day(ts, tdays):
    """filing before 15:30 IST on a trading day explains that day; otherwise the next trading day"""
    day = ts.dt.normalize(); after = (ts.dt.hour + ts.dt.minute / 60) >= 15.5
    key = day + pd.to_timedelta(after.astype(int), unit="D")
    return _assign_first_ge(key.values, tdays)


# ----------------------------------------------------------------------------- news items per stock
def build_news_items():
    toi = pd.read_parquet(first(INTERIM / "toi_company_scored.parquet", PROCESSED / "toi_company_scored.parquet"))
    fil = pd.read_parquet(first(INTERIM / "nse_company_scored.parquet", PROCESSED / "nse_company_scored.parquet"))
    items = {}
    for key in config.STOCKS:
        d, r = stock_prices(key)
        t = toi[toi.stock == key].copy()
        t["trade_date"] = _assign_first_ge(t.date.values, d.index)
        t = t.assign(source="Times of India headline", published=t.date.dt.strftime("%Y-%m-%d"),
                     targeted=t.ent_pos - t.ent_neg, plain=t.fbh_pos - t.fbh_neg)
        f = fil[fil.symbol == NSE_SYM.get(key, key)].copy()
        f["trade_date"] = filing_trading_day(f.ts, d.index)
        f = f.assign(source="NSE filing", published=f.ts.dt.strftime("%Y-%m-%d %H:%M"), text=f.clean,
                     targeted=f.ent_pos - f.ent_neg, plain=f.fb_pos - f.fb_neg)
        cols = ["trade_date", "published", "source", "entity", "text", "targeted", "plain", "ent_neg", "ent_neu", "ent_pos"]
        x = pd.concat([t[cols], f[cols]], ignore_index=True).dropna(subset=["trade_date"])
        x["label"] = np.array(["negative", "neutral", "positive"])[x[["ent_neg", "ent_neu", "ent_pos"]].values.argmax(1)]
        x["price_report"] = is_price_report(x.text)
        x["day_return_pct"] = r.reindex(pd.to_datetime(x.trade_date)).values
        x = x.drop(columns=["ent_neg", "ent_neu", "ent_pos"]).sort_values(["trade_date", "source"])
        x["text"] = x.text.str.slice(0, 300)
        for c in ["targeted", "plain", "day_return_pct"]: x[c] = x[c].round(4)
        x.to_csv(NEWS / f"{key}_news_items.csv", index=False)
        items[key] = x
        print(f"  {key:10s} {len(x):6d} items ({(x.source == 'NSE filing').sum()} filings), "
              f"price-report headlines {x.price_report.mean():.1%}", flush=True)
    return items


# ----------------------------------------------------------------------------- statistics
def lead_lag(s, r, scope, variant, lags=range(-5, 6)):
    rows = []
    for k in lags:
        x = pd.concat([s, r.shift(-k)], axis=1).dropna()
        rho, p = spearmanr(x.iloc[:, 0], x.iloc[:, 1]) if len(x) > 30 else (np.nan, np.nan)
        rows.append({"scope": scope, "variant": variant, "lag": k, "rho": rho, "p": p, "n": len(x)})
    return rows


def quintiles(s, r, scope, variant):
    x = pd.concat([s.rename("s"), r.rename("same"), r.shift(-1).rename("next")], axis=1).dropna()
    if len(x) < 50: return []
    x["q"] = pd.qcut(x.s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    g = x.groupby("q").agg(same_day=("same", "mean"), next_day=("next", "mean"), n=("s", "size")).reset_index()
    return [dict(scope=scope, variant=variant, **row) for row in g.to_dict("records")]


def event_study(s, ar, scope, pos_thr, neg_thr, win=5):
    rows = []
    idx = ar.index
    for grp, mask in [("Positive news", s >= pos_thr), ("Negative news", s <= neg_thr)]:
        ev = s.index[mask.reindex(s.index).fillna(False)]
        paths = []
        for d in ev:
            i = idx.get_indexer([d])[0]
            if i - win < 0 or i + win >= len(idx): continue
            seg = ar.iloc[i - win:i + win + 1].values
            if np.isnan(seg).any(): continue
            paths.append(np.cumsum(seg))
        if not paths: continue
        P = np.vstack(paths); m = P.mean(0); se = P.std(0, ddof=1) / np.sqrt(len(P))
        for j, k in enumerate(range(-win, win + 1)):
            rows.append({"scope": scope, "group": grp, "day": k, "car": m[j], "lo": m[j] - 1.96 * se[j],
                         "hi": m[j] + 1.96 * se[j], "n_events": len(P)})
    return rows


def big_moves(s_all, r, scope, top_headline=None, q=0.95, neutral=0.1, n_top=25):
    rr = r.loc[START:END].dropna()
    big = rr[rr.abs() >= rr.abs().quantile(q)]
    sv = s_all.reindex(big.index)
    cat = np.where(sv.isna(), "No news that day",
          np.where(sv.abs() < neutral, "Neutral news",
          np.where(np.sign(sv) == np.sign(big), "News sentiment matches the move", "News sentiment opposite to the move")))
    cov = pd.Series(cat).value_counts()
    rows = [{"scope": scope, "category": c, "count": int(cov.get(c, 0)), "share": cov.get(c, 0) / len(big)}
            for c in ["News sentiment matches the move", "Neutral news", "News sentiment opposite to the move", "No news that day"]]
    top = []
    if top_headline is not None:
        for d in big.abs().sort_values(ascending=False).index[:n_top]:
            h = top_headline.get(d)
            top.append({"scope": scope, "date": d.strftime("%Y-%m-%d"), "return_pct": round(big[d], 2),
                        "sentiment": None if pd.isna(sv.get(d)) else round(sv.get(d), 3),
                        "headline": h[0] if h else "", "headline_score": h[1] if h else None})
    return rows, top


def mcnemar(a, b):
    ca = (a.p > .5) == (a.y > .5); cb = (b.p > .5) == (b.y > .5)
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    return binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0


def same_day_accuracy(df, base, sets, scope, years, models=("LogReg", "LightGBM"), mask_col=None):
    rows = []; zoo = E.clf_zoo()
    for m in models:
        ref = walk(df, base, "up_today", zoo[m], years)
        for name, feats in sets.items():
            o = walk(df, base + feats, "up_today", zoo[m], years)
            for sub in ["all days"] + (["news days"] if mask_col else []):
                if sub == "news days":
                    nd = df.dropna(subset=["up_today"]).set_index(["date", "ticker"])[mask_col]
                    k = nd.reindex(pd.MultiIndex.from_arrays([o.date, o.ticker])).values > 0
                    a, b = ref[k], o[k]
                else:
                    a, b = ref, o
                rows.append({"scope": scope, "model": m, "sentiment": name, "rows": sub, "n": len(a),
                             "acc_without": ((a.p > .5) == (a.y > .5)).mean(),
                             "acc_with": ((b.p > .5) == (b.y > .5)).mean(),
                             "majority": max(a.y.mean(), 1 - a.y.mean()), "p_mcnemar": mcnemar(a, b)})
                rows[-1]["gain_pts"] = 100 * (rows[-1]["acc_with"] - rows[-1]["acc_without"])
                print(f"  same-day {scope:8s} {m:8s} {name:32s} {sub:10s} "
                      f"{rows[-1]['acc_without']:.4f} -> {rows[-1]['acc_with']:.4f} p={rows[-1]['p_mcnemar']:.4f}", flush=True)
    return rows


def lagged_base(d_ohlc, r):
    """technical features known before day t opens (computed to close t-1)"""
    dd = d_ohlc.rename(columns=str.lower).copy()
    if "adjclose" not in dd: dd["adjclose"] = dd["close"]
    f, _ = technical_features(dd)
    f = f.drop(columns=["dow", "oc_ret", "hl_range", "gap_ret"], errors="ignore").shift(1)
    return f


def main():
    print("news items", flush=True)
    items = build_news_items()
    mkt = market_return()
    LL, QQ, EV, BC, BT, SD, SUM = [], [], [], [], [], [], []

    # ---------------- Sensex: market-wide headlines
    sx = load_prices("^BSESN").loc["1999-01-01":END]; rs = np.log(sx.close).diff() * 100
    h = pd.read_parquet(first(INTERIM / "toi_market_scored_fb.parquet", PROCESSED / "toi_market_headlines_scored.parquet"))
    h["net"] = h.fb_pos - h.fb_neg; h["pr"] = is_price_report(h.text)
    h["trade_date"] = _assign_first_ge(h.date.values, sx.index)
    daily = {}
    for variant, sub in [("all headlines", h), ("excluding price reports", h[~h.pr])]:
        s = sub.dropna(subset=["trade_date"]).groupby("trade_date").net.mean().reindex(sx.index)
        z = ((s - s.rolling(250, min_periods=60).mean()) / s.rolling(250, min_periods=60).std()).loc[START:END]
        daily[variant] = z
        LL += lead_lag(z, rs, "SENSEX", variant); QQ += quintiles(z, rs, "SENSEX", variant)
    z = daily["excluding price reports"]
    EV += event_study(z, rs.reindex(z.index), "SENSEX", z.quantile(.9), z.quantile(.1))
    # (no single "top headline" for the market: ~35 headlines a day, so none is singled out)
    c, t = big_moves(z, rs, "SENSEX"); BC += c; BT += t
    # same-day accuracy, Sensex
    base = lagged_base(sx[["open", "high", "low", "close", "volume"]].rename(columns=str.title), rs)
    g = E._global_india(sx.index).shift(1)
    df = base.join(g).loc[START:END]
    df["up_today"] = (rs > 0).astype(float).where(rs.notna()).reindex(df.index)
    for v, zz in daily.items():
        df[f"s_{v}"] = zz; df[f"s_{v}_chg"] = zz - zz.shift(1)
    df = df.reset_index().rename(columns={"index": "date", "Date": "date"}); df["ticker"] = "SENSEX"
    B = list(base.columns) + list(g.columns)
    SD += same_day_accuracy(df, B, {"sentiment (all headlines)": ["s_all headlines", "s_all headlines_chg"],
                                     "sentiment (no price reports)": ["s_excluding price reports", "s_excluding price reports_chg"]},
                            "SENSEX", list(range(2007, 2023)))

    # ---------------- each stock + pooled: company-targeted sentiment of headlines naming it
    panel = []
    for key, x in items.items():
        d, r = stock_prices(key)
        ar = (r - mkt.reindex(r.index)).dropna()            # abnormal = stock minus Sensex
        hx = x[x.source == "Times of India headline"]
        dd = {}
        for variant, sub in [("all headlines", hx), ("excluding price reports", hx[~hx.price_report])]:
            s = sub.groupby(pd.to_datetime(sub.trade_date)).targeted.mean().reindex(d.index).loc[START:END]
            dd[variant] = s
            LL += lead_lag(s, r, key, variant); QQ += quintiles(s, r, key, variant)
        s = dd["excluding price reports"]
        EV += event_study(s.dropna(), ar, key, 0.5, -0.5)
        tops = {}
        for dt, gg in hx[~hx.price_report].groupby(pd.to_datetime(hx[~hx.price_report].trade_date)):
            i = gg.targeted.abs().idxmax(); tops[dt] = (gg.text[i], round(gg.targeted[i], 3))
        c, t = big_moves(s, r.loc[d.index[0]:], key, tops); BC += c; BT += t
        base = lagged_base(d[["Open", "High", "Low", "Close", "Volume"]], r)
        p = base.join(E._global_india(d.index).shift(1)).join(mkt.shift(1).rename("sensex_r_lag"))
        for v, ss in dd.items():
            sub = hx[~hx.price_report] if v.startswith("excl") else hx
            p[f"s_{v}"] = ss.fillna(0.0)
            p[f"n_{v}"] = np.log1p(sub.groupby(pd.to_datetime(sub.trade_date)).size()).reindex(d.index).fillna(0.0)
        p["news_day"] = (p["n_all headlines"] > 0).astype(float)
        p["up_today"] = (r > 0).astype(float).where(r.notna())
        p = p.loc[max(START, str(d.index[0].date())):END].iloc[200:]
        p = p.reset_index().rename(columns={"index": "date", "Date": "date"}); p["ticker"] = key
        panel.append(p)
    P = pd.concat(panel, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    # pooled lead-lag across the 10 stocks (news days only)
    pooled_ll = []
    for variant in ["all headlines", "excluding price reports"]:
        for k in range(-5, 6):
            vals = []
            for key in items:
                d, r = stock_prices(key)
                hx = items[key]; hx = hx[hx.source == "Times of India headline"]
                if variant.startswith("excl"): hx = hx[~hx.price_report]
                s = hx.groupby(pd.to_datetime(hx.trade_date)).targeted.mean().reindex(d.index).loc[START:END]
                vals.append(pd.concat([s, r.shift(-k)], axis=1).dropna().values)
            v = np.vstack([a for a in vals if len(a)])
            rho, pv = spearmanr(v[:, 0], v[:, 1])
            pooled_ll.append({"scope": "ALL10", "variant": variant, "lag": k, "rho": rho, "p": pv, "n": len(v)})
    LL += pooled_ll
    B = [c for c in P.columns if c not in ("date", "ticker", "up_today", "news_day") and not c.startswith(("s_", "n_"))]
    SD += same_day_accuracy(P, B, {"targeted sentiment (all headlines)": ["s_all headlines", "n_all headlines"],
                                    "targeted sentiment (no price reports)": ["s_excluding price reports", "n_excluding price reports"]},
                            "ALL10", list(range(2007, 2023)), mask_col="news_day")

    ll = pd.DataFrame(LL); qq = pd.DataFrame(QQ); ev = pd.DataFrame(EV); bc = pd.DataFrame(BC); bt = pd.DataFrame(BT); sd = pd.DataFrame(SD)
    for name, t in [("lead_lag", ll), ("quintiles", qq), ("event_study", ev), ("big_moves_coverage", bc),
                    ("big_moves_top", bt), ("same_day_accuracy", sd)]:
        t.round(5).to_csv(OUT / f"{name}.csv", index=False)
    # summary per scope (price reports excluded = the defensible numbers)
    for scope in ll.scope.unique():
        a = ll[(ll.scope == scope) & (ll.variant == "excluding price reports")].set_index("lag")
        q = qq[(qq.scope == scope) & (qq.variant == "excluding price reports")].set_index("q")
        b = bc[bc.scope == scope].set_index("category")
        SUM.append({"scope": scope, "same_day_rho": a.rho.get(0), "same_day_p": a.p.get(0), "next_day_rho": a.rho.get(1),
                    "news_days": a.n.get(0),
                    "q5_minus_q1_same_day_pct": (q.same_day.get(5) - q.same_day.get(1)) if len(q) else np.nan,
                    "q5_minus_q1_next_day_pct": (q.next_day.get(5) - q.next_day.get(1)) if len(q) else np.nan,
                    "big_moves_matching_news": b["share"].get("News sentiment matches the move", np.nan) if len(b) else np.nan,
                    "big_moves_opposite_news": b["share"].get("News sentiment opposite to the move", np.nan) if len(b) else np.nan,
                    "big_moves_no_news": b["share"].get("No news that day", np.nan) if len(b) else np.nan})
    sm = pd.DataFrame(SUM); sm.round(4).to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(sm.round(3).to_string(index=False)); print(sd.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

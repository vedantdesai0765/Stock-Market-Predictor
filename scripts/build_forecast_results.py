"""
Build the tables behind the dashboard's "Next-morning forecast" view.

Reads the out-of-fold (walk-forward, never-seen-year) predictions saved by the sentiment study and
summarises them stock by stock. Nothing is retrained here.

Inputs  (sentiment_study/reports/)
  oof_company_all.parquet   10 stocks; base features = the stock's own price indicators + market
                            (Nifty, S&P 500 overnight return, VIX, Asian opening gaps) known before 09:15 IST;
                            "+ news" = company-targeted sentiment (headlines or NSE filings)
  oof_all_1.parquet, oof_all_4.parquet   Sensex and US market-level runs
Outputs (reports/forecast/)
  stock_summary.csv   one row per period x target x model x stock
  stock_by_year.csv   the same, per test year
  market_summary.csv  Sensex / US context rows

    python scripts/build_forecast_results.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "sentiment_study" / "reports"
OUT = ROOT / "reports" / "forecast"; OUT.mkdir(parents=True, exist_ok=True)
NAME = {"SBI": "SBI"}
PERIOD = {"NSE": "2013–2025 (with company announcements)", "TOI": "2007–2022 (with newspaper headlines)"}
TARGET = {"gapup_next": "Next morning's opening price", "up_next": "Next day's closing price"}


def correct(p, y):
    return (np.asarray(p) > .5) == (np.asarray(y) > .5)


def mcnemar(ca, cb):
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    return binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0


def stocks():
    oof = pd.read_parquet(REP / "oof_company_all.parquet")
    oof["date"] = pd.to_datetime(oof["date"])
    parts = oof["key"].str.split("__", expand=True)
    oof["study"], oof["target"], oof["model"], oof["fs"] = parts[0], parts[1], parts[2], parts[3]
    rows, yrs = [], []
    for (study, target, model), g in oof.groupby(["study", "target", "model"]):
        if not {"B", "B+Cent"} <= set(g.fs):
            continue
        a = g[g.fs == "B"].set_index(["date", "ticker"]).sort_index()
        b = g[g.fs == "B+Cent"].set_index(["date", "ticker"]).sort_index().reindex(a.index)
        a = a.assign(cm=correct(a.p, a.y), cn=correct(b.p, a.y), cg=correct(a.maj, a.y)).reset_index()
        for scope, s in [("ALL", a)] + list(a.groupby("ticker")):
            cm, cn, cg = s.cm.values, s.cn.values, s.cg.values
            rows.append({"period": PERIOD[study], "target": TARGET[target], "model": model, "ticker": scope,
                         "n": len(s), "first_year": int(s.date.dt.year.min()), "last_year": int(s.date.dt.year.max()),
                         "acc_guess": cg.mean(), "acc_model": cm.mean(), "acc_news": cn.mean(),
                         "edge": cm.mean() - cg.mean(), "p_vs_guess": mcnemar(cg, cm),
                         "news_effect": cn.mean() - cm.mean(), "p_news": mcnemar(cm, cn)})
            for yr, t in s.groupby(s.date.dt.year):
                yrs.append({"period": PERIOD[study], "target": TARGET[target], "model": model, "ticker": scope,
                            "year": int(yr), "n": len(t), "acc_guess": t.cg.mean(), "acc_model": t.cm.mean(),
                            "acc_news": t.cn.mean()})
    pd.DataFrame(rows).round(5).to_csv(OUT / "stock_summary.csv", index=False)
    pd.DataFrame(yrs).round(5).to_csv(OUT / "stock_by_year.csv", index=False)
    print("stocks:", len(rows), "rows")


def market():
    specs = [  # (file, key without / with news, label)
        ("oof_all_4", "M1S__gapup_next__LogReg__T+G+A", "M1S__gapup_next__LogReg__T+G+A+S", "Sensex: next morning's opening price (2009–2022)"),
        ("oof_all_1", "IN__up_next__LogReg__T+G", "IN__up_next__LogReg__T+G+S", "Sensex: next day's closing price (2009–2022)"),
        ("oof_all_4", "IN70__volup_next__LogReg__T+G", "IN70__volup_next__LogReg__T+G+S", "Sensex: will tomorrow be more volatile than today?"),
        ("oof_all_1", "USgap_DJI__gapup_next__LogReg__T+G", "USgap_DJI__gapup_next__LogReg__T+G+S", "US Dow Jones: next opening price (news to 09:00 New York)"),
    ]
    cache, rows = {}, []
    for f, ka, kb, label in specs:
        if f not in cache:
            cache[f] = pd.read_parquet(REP / f"{f}.parquet")
        d = cache[f]
        a = d[d.key == ka].reset_index(drop=True); b = d[d.key == kb].reset_index(drop=True)
        if a.empty or len(a) != len(b):
            continue
        cm, cn, cg = correct(a.p, a.y), correct(b.p, a.y), correct(a.maj, a.y)
        rows.append({"task": label, "n": len(a), "acc_guess": cg.mean(), "acc_model": cm.mean(), "acc_news": cn.mean(),
                     "edge": cm.mean() - cg.mean(), "p_vs_guess": mcnemar(cg, cm),
                     "news_effect": cn.mean() - cm.mean(), "p_news": mcnemar(cm, cn)})
    pd.DataFrame(rows).round(5).to_csv(OUT / "market_summary.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    stocks(); market()

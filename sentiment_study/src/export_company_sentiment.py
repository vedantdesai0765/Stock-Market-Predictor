"""
Write one company-targeted daily sentiment file per stock for the main project:
  data/processed/<STOCK>_company_sentiment.csv
columns: date, finbert_sentiment, vader_sentiment, company_news_count, filings_sentiment, filings_count
  finbert_sentiment  = entity-level FinBERT score towards THIS company (ToI headlines naming it),
                       smoothed (EMA, half-life 3 trading days; fades to 0 without news) so every trading day has a value
  vader_sentiment    = VADER of the same headlines, same decay (kept for compatibility)
  filings_sentiment  = entity-level score of the company's own NSE filings (overnight window), same decay
One row per trading day; news dated d is attached to the last trading day <= d.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, ROOT, first, price_file

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from company_study import STOCKS, NSE_SYM, stock_prices, IN
from sentiment_daily import _assign_last_le
from method4 import overnight_window

DEST = sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "data" / "processed")


def decayed(items, tdays, val):
    """daily mean score (0 on days without news), smoothed with an EMA (half-life 3 trading days)
    so the signal fades back to neutral when there is no fresh news"""
    it = items.dropna(subset=["row"])
    m = it.groupby("row")[val].mean().reindex(tdays).fillna(0.0)
    n = it.groupby("row").size().reindex(tdays).fillna(0.0)
    return m.ewm(halflife=3, adjust=False).mean(), n


if __name__ == "__main__":
    toi = pd.read_parquet(first(f"{IN}/toi_company_scored.parquet", PROCESSED / "toi_company_scored.parquet")); fil = pd.read_parquet(first(f"{IN}/nse_company_scored.parquet", PROCESSED / "nse_company_scored.parquet"))
    toi["ent_net"] = toi.ent_pos - toi.ent_neg; fil["ent_net"] = fil.ent_pos - fil.ent_neg
    summ = []
    for k in STOCKS:
        td = stock_prices(k).index
        t = toi[toi.stock == k].copy(); t["row"] = _assign_last_le(t.date.values, td)
        f = fil[fil.symbol == NSE_SYM.get(k, k)].copy(); f["row"] = overnight_window(f, td)
        fb, n = decayed(t, td, "ent_net"); vd, _ = decayed(t, td, "vader")
        fs, fn = decayed(f, td, "ent_net")
        out = pd.DataFrame({"date": td.strftime("%Y-%m-%d"), "finbert_sentiment": fb.round(5).values,
                            "vader_sentiment": vd.round(5).values, "company_news_count": n.astype(int).values,
                            "filings_sentiment": fs.round(5).values, "filings_count": fn.astype(int).values})
        # headlines exist only 2001..2022-03; outside that window the headline columns are 0 (neutral)
        out.loc[pd.to_datetime(out.date) > "2022-03-31", ["finbert_sentiment", "vader_sentiment"]] = 0.0
        out.to_csv(f"{DEST}/{k}_company_sentiment.csv", index=False)
        summ.append({"stock": k, "rows": len(out), "headline_days": int((out.company_news_count > 0).sum()),
                     "headlines": int(out.company_news_count.sum()), "filing_days": int((out.filings_count > 0).sum()),
                     "filings": int(out.filings_count.sum())})
    print(pd.DataFrame(summ).to_string(index=False))

"""
Bridge to the main pipeline: write market-news sentiment in the format src/dataset.py expects
(data/processed/<file>.csv with columns date, vader_sentiment, finbert_sentiment).

    python sentiment_study/src/export_to_main.py

Source: Times of India market/business headlines (2001-01-02 .. 2022-03-31), scored with VADER
and FinBERT. Row date = Indian trading day t; it averages headlines dated from t up to (not
including) the next trading day, i.e. news known by midnight of t. For a strictly close-time
forecast, merge with lag_days=1 in build_dataset(..., sentiment_lag_days=1).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from paths import ROOT
import pandas as pd
from prices import load_prices
from sentiment_daily import toi_daily

MAIN_PROCESSED = ROOT.parent / "data" / "processed"
OUT = MAIN_PROCESSED / "IN_market_news_sentiment.csv"

if __name__ == "__main__":
    d = load_prices("^BSESN").loc["2001-01-02":"2022-03-31"]
    s = toi_daily(d.index)
    out = pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"),
                        "vader_sentiment": s["vader"].round(4).values,
                        "finbert_sentiment": s["fb_net"].round(4).values})
    MAIN_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} trading days, {out.date.min()} .. {out.date.max()})")

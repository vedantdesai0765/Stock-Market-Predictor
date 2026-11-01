"""Company-level news (FNSPID Nasdaq-100 subset): clean, score with VADER/LM/FinBERT."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
IN = str(INTERIM)
TICKS = "GILD,NVDA,INTC,TSLA,GOOG,MU,AAPL,AMD,QCOM,MSFT,CMCSA,SBUX,EBAY,COST,PEP,PANW,BIIB,TXN,PYPL,ADBE,CSX,AMGN,VRTX,AMZN,DLTR,TMUS,SIRI,AMAT,AVGO,ADI,MNST,ALGN,WDAY,MRVL,REGN,JD,ORLY,PAYX".split(",")

if __name__ == "__main__":
    d = pd.read_parquet(str(RAW / "fnspid_ndx100_slim.parquet"))
    d = d[d.Stock_symbol.isin(TICKS)].copy()
    ts = pd.to_datetime(d.Date, utc=True)
    midnight = (ts.dt.hour == 0) & (ts.dt.minute == 0)
    d["date"] = np.where(midnight, ts.dt.tz_localize(None).dt.normalize(),
                         ts.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize())
    d["date"] = pd.to_datetime(d.date)
    d["has_time"] = ~midnight
    d = d[d.date >= "2012-01-01"]
    d = d.rename(columns={"Article_title": "text", "Stock_symbol": "ticker"}).dropna(subset=["text"])
    d = d.drop_duplicates(subset=["ticker", "date", "text"]).reset_index(drop=True)
    print("rows", len(d), "with real time", d.has_time.mean().round(3), flush=True)
    from score_lexicon import run
    d["vader"], d["lm_pos"], d["lm_neg"] = run(d.text.tolist())
    d[["ticker", "date", "has_time", "text", "vader", "lm_pos", "lm_neg"]].to_parquet(f"{IN}/fnspid_scored.parquet")
    from score_finbert import score
    p = score(d.text.tolist(), f"{IN}/finbert_fnspid.npy")
    d["fb_pos"], d["fb_neg"], d["fb_neu"] = p[:, 0], p[:, 1], p[:, 2]
    d[["ticker", "date", "has_time", "text", "vader", "lm_pos", "lm_neg", "fb_pos", "fb_neg"]].to_parquet(f"{IN}/fnspid_scored_fb.parquet")
    print("DONE FNSPID", flush=True)

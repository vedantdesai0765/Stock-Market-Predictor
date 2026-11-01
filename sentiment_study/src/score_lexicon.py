"""VADER + Loughran-McDonald scores for every headline in both corpora."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import sys, time
import numpy as np, pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pysentiment2 as ps
from multiprocessing import Pool

IN = str(INTERIM)
_v = SentimentIntensityAnalyzer(); _lm = ps.LM()

def score(t):
    c = _v.polarity_scores(t)["compound"]
    s = _lm.get_score(_lm.tokenize(t))
    return c, s["Positive"], s["Negative"]

def run(texts):
    with Pool(2) as p:
        r = p.map(score, texts, chunksize=2000)
    a = np.array(r, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2]

if __name__ == "__main__":
    t0 = time.time()
    toi = pd.read_parquet(f"{IN}/toi_all.parquet")
    toi = toi[toi.is_market].reset_index(drop=True)
    toi["vader"], toi["lm_pos"], toi["lm_neg"] = run(toi.text.tolist())
    toi.to_parquet(f"{IN}/toi_market_scored.parquet"); print("toi", len(toi), time.time()-t0, flush=True)
    nyt = pd.read_parquet(f"{IN}/nyt_fin.parquet")
    txt = (nyt.headline + ". " + nyt.abstract).tolist()
    nyt["vader"], nyt["lm_pos"], nyt["lm_neg"] = run(txt)
    nyt["vader_head"] = run(nyt.headline.tolist())[0]
    nyt.to_parquet(f"{IN}/nyt_fin_scored.parquet"); print("nyt", len(nyt), time.time()-t0, flush=True)

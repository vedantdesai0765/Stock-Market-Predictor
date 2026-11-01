"""FinBERT on the market-related NYT subset only (CPU budget)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from score_finbert import score, IN
from sentiment_daily import MKT_RE
nyt = pd.read_parquet(f"{IN}/nyt_fin_scored.parquet")
mask = (nyt.headline + " " + nyt.abstract).str.contains(MKT_RE, regex=True).values
idx = np.where(mask)[0]
p = score(nyt.headline.values[idx].tolist(), f"{IN}/finbert_nyt_mkt.npy")
for k, c in enumerate(["fb_pos", "fb_neg", "fb_neu"]):
    nyt[c] = np.nan; nyt.loc[nyt.index[idx], c] = p[:, k]
nyt.to_parquet(f"{IN}/nyt_fin_scored_fb.parquet"); print("DONE NYT", len(idx), flush=True)

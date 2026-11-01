"""FinBERT (ProsusAI/finbert) scores, checkpointed, int8-quantised on CPU."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import sys, os, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_num_threads(2)
MD = str(FINBERT); IN = str(INTERIM)
tok = AutoTokenizer.from_pretrained(MD)
m = AutoModelForSequenceClassification.from_pretrained(MD).eval()
m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)

def score(texts, out, bs=128):
    if os.path.exists(out):
        return np.load(out)
    n = len(texts); res = np.full((n, 3), np.nan, dtype=np.float32)
    ckpt = out + ".partial.npy"
    if os.path.exists(ckpt): res = np.load(ckpt)
    order = np.argsort([len(t) for t in texts])
    t0 = time.time(); done = 0
    for i in range(0, n, bs):
        idx = order[i:i+bs]
        if not np.isnan(res[idx[0], 0]): continue
        x = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.inference_mode():
            res[idx] = m(**x).logits.softmax(-1).numpy()
        done += len(idx)
        if (i // bs) % 200 == 0:
            np.save(ckpt, res); print(out, i, n, round(done/(time.time()-t0), 1), "/s", flush=True)
    np.save(out, res); return res

if __name__ == "__main__":
    toi = pd.read_parquet(f"{IN}/toi_market_scored.parquet")
    p = score(toi.text.tolist(), f"{IN}/finbert_toi.npy")
    toi["fb_pos"], toi["fb_neg"], toi["fb_neu"] = p[:, 0], p[:, 1], p[:, 2]
    toi.to_parquet(f"{IN}/toi_market_scored_fb.parquet")
    nyt = pd.read_parquet(f"{IN}/nyt_fin_scored.parquet")
    p = score(nyt.headline.tolist(), f"{IN}/finbert_nyt.npy")
    nyt["fb_pos"], nyt["fb_neg"], nyt["fb_neu"] = p[:, 0], p[:, 1], p[:, 2]
    nyt.to_parquet(f"{IN}/nyt_fin_scored_fb.parquet")
    print("DONE", flush=True)

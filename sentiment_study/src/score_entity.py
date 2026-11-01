"""
Score (company, text) pairs.
  ent_*  : entity-level FinBERT fine-tuned on SEntFiN  -> sentiment TOWARDS the named company
  fbh_*  : plain FinBERT on the whole headline          -> headline tone, not targeted
Outputs probabilities neg / neu / pos.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, ROOT, first, price_file

import sys, os, time, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(2)
IN = str(INTERIM)
ENT = str(RAW / "finbert_entity"); BASE = str(FINBERT)


def _load(path):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).eval()
    # no int8 quantisation: it flattens the fine-tuned entity model (checked on SEntFiN test)
    return tok, m


@torch.no_grad()
def run(tok, m, a, b=None, bs=64, order=None, tag=""):
    out = []; t0 = time.time()
    for i in range(0, len(a), bs):
        enc = tok(a[i:i + bs], b[i:i + bs] if b is not None else None, truncation=True,
                  max_length=64 if b is not None else 96, padding=True, return_tensors="pt")
        p = torch.softmax(m(**enc).logits, -1).numpy()
        out.append(p[:, order] if order is not None else p)
        if (i // bs) % 50 == 0: print(tag, i, len(a), f"{time.time() - t0:.0f}s", flush=True)
    return np.vstack(out)


def entity_label(sym, sm_name):
    short = {"SBIN": "SBI", "INFY": "Infosys", "TCS": "TCS", "ITC": "ITC", "HDFCBANK": "HDFC Bank",
             "RELIANCE": "Reliance Industries", "ADANIENT": "Adani Enterprises", "ADANIGREEN": "Adani Green",
             "VEDL": "Vedanta", "ETERNAL": "Zomato"}
    return short.get(sym, sm_name)


if __name__ == "__main__":
    toi = pd.read_parquet(f"{IN}/toi_company_matches.parquet").reset_index(drop=True)
    # filings for the 10 stocks (8 from the big file + ADANIGREEN/ETERNAL)
    keep = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "ITC", "ADANIENT", "VEDL"]
    f8 = pd.read_parquet(first(f"{IN}/nse_ann_scored.parquet", PROCESSED / "nse_announcements_scored.parquet")); f8 = f8[f8.symbol.isin(keep)]
    f2 = pd.read_parquet(f"{IN}/nse_ann_new2_prepped.parquet"); f2 = f2[f2.routine == 0]
    fil = pd.concat([f8, f2], ignore_index=True)
    fil["entity"] = [entity_label(s, n) for s, n in zip(fil.symbol, fil.sm_name)]
    print("toi pairs", len(toi), "filings", len(fil), flush=True)

    tok, m = _load(ENT)
    p = run(tok, m, toi.entity.tolist(), toi.text.tolist(), tag="ent-toi")
    toi[["ent_neg", "ent_neu", "ent_pos"]] = p
    p = run(tok, m, fil.entity.tolist(), fil.clean.tolist(), tag="ent-fil")
    fil[["ent_neg", "ent_neu", "ent_pos"]] = p
    tok, m = _load(BASE)
    p = run(tok, m, toi.text.tolist(), order=[1, 2, 0], tag="fb-toi")
    toi[["fbh_neg", "fbh_neu", "fbh_pos"]] = p
    miss = pd.Series(True, index=fil.index)                 # rescore all filings in fp32 (fair comparison)
    if miss.any():
        p = run(tok, m, fil.loc[miss, "clean"].tolist(), order=[1, 2, 0], tag="fb-fil")
        fil.loc[miss, "fb_neg"] = p[:, 0]; fil.loc[miss, "fb_pos"] = p[:, 2]
    from score_lexicon import run as lex
    for d, col in [(toi, "text"), (fil, "clean")]:
        miss = d.get("vader", pd.Series(np.nan, index=d.index)).isna()
        v, lp, ln = lex(d.loc[miss, col].tolist())
        d.loc[miss, "vader"] = v; d.loc[miss, "lm_pos"] = lp; d.loc[miss, "lm_neg"] = ln
    toi.to_parquet(f"{IN}/toi_company_scored.parquet"); fil.to_parquet(f"{IN}/nse_company_scored.parquet")
    print("done", flush=True)

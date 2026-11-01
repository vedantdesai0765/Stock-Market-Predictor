"""Legitimate routes towards ~70%: (A) India opening gap, (B) selective prediction, (C) volatility direction."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import experiment as E
from prices import load_prices
from analyze import compare

rows = []
def log(**k): rows.append(k); print(k, flush=True)

M = {m: E.clf_zoo()[m] for m in ["LogReg", "LightGBM", "RandomForest"]}

# ---------- A: Sensex opening gap (prediction made 02:30-09:00 IST, after the US close, before the 09:15 open)
df, tech, glob_, sent = E.build("IN")
d = load_prices("^BSESN").reindex(df.index)
gap = np.log(d.open.shift(-1) / d.close)
flat = (gap.abs() < 1e-6)
print("share of flat Sensex opens by year:\n", flat.groupby(gap.index.year).mean().round(3).to_string())
df["gapup_next"] = (gap > 0).astype(float).where(gap.notna() & ~flat)
for m, fn in M.items():
    for fs, feats in [("T", tech), ("T+S", tech + sent), ("T+G", tech + glob_), ("T+G+S", tech + glob_ + sent)]:
        o = E.walk_forward(df, feats, "gapup_next", fn, 2009)
        k = f"IN70__gapup_next__{m}__{fs}"; o.to_parquet(f"{E.OUT}/{k}.parquet")
        log(test="A India gap", model=m, set=fs, n=len(o), acc=round(((o.p > .5) == (o.y > .5)).mean(), 4), base=round(o.y.mean(), 4))

# ---------- C: volatility direction: will next-5-day realised vol exceed the past-20-day vol?
for mk in ["IN", "US"]:
    dfm, t2, g2, s2 = E.build(mk)
    r = dfm.ret_l1
    rv5n = r[::-1].rolling(5).std()[::-1].shift(-1)
    dfm["volup_next"] = (rv5n > dfm.vol_20).astype(float).where(rv5n.notna())
    fty = 2009 if mk == "IN" else 1998; blk = 1 if mk == "IN" else 2
    for m, fn in M.items():
        for fs, feats in [("T+G", t2 + g2), ("T+G+S", t2 + g2 + s2)]:
            o = E.walk_forward(dfm, feats, "volup_next", fn, fty, embargo=10, block=blk)
            k = f"{mk}70__volup_next__{m}__{fs}"; o.to_parquet(f"{E.OUT}/{k}.parquet")
            log(test=f"C {mk} vol up", model=m, set=fs, n=len(o), acc=round(((o.p > .5) == (o.y > .5)).mean(), 4), base=round(max(o.y.mean(), 1 - o.y.mean()), 4))

pd.DataFrame(rows).to_csv(REPORTS / "seventy.csv", index=False)

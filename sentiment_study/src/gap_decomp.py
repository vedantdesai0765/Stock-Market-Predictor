"""Where does the US pre-open sentiment gain come from: the overnight gap or the next trading day?"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import experiment as E
from prices import load_prices

for tk, real_open, first_test in [("^DJI", "1996", 2003), ("^GSPC", "2008", 2014)]:
    df, tech, glob_, sent = E.build("US", ticker=tk, nyt_mode="preopen")
    df = df.loc[real_open:]
    d = load_prices(tk).reindex(df.index)
    gap = np.log(d.open.shift(-1) / d.close); oc = np.log(d.close.shift(-1) / d.open.shift(-1))
    df["gapup_next"] = (gap > 0).astype(float).where(gap.notna())
    df["ocup_next"] = (oc > 0).astype(float).where(oc.notna())
    tag = "_" + tk.strip("^")
    for tg in ["gapup_next", "ocup_next", "up_next"]:
        for m in ["LogReg", "LightGBM"]:
            for fs, feats in [("T+G", tech + glob_), ("T+G+S", tech + glob_ + sent)]:
                o = E.walk_forward(df, feats, tg, E.clf_zoo()[m], first_test, block=2)
                key = f"USgap{tag}__{tg}__{m}__{fs}"
                o.to_parquet(f"{E.OUT}/{key}.parquet")
                print(key, len(o), round(((o.p > .5) == (o.y > .5)).mean(), 4), flush=True)

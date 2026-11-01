"""Consolidated with-vs-without-sentiment table for the paper (reports/comparison_all.csv)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from analyze import table, OOF

R = str(REPORTS)
M = ["LogReg", "RandomForest", "GradBoost", "XGBoost", "LightGBM", "LSTM", "GRU"]
SPEC = [("IN", "up_next"), ("IN", "big_next"), ("IN", "up_today"),
        ("US", "up_next"), ("US", "big_next"), ("US", "up_today"),
        ("US_close", "up_next"), ("US_close", "big_next"),
        ("USgap_DJI", "gapup_next"), ("USgap_DJI", "ocup_next"), ("USgap_DJI", "up_next"),
        ("USgap_GSPC", "gapup_next"), ("USgap_GSPC", "ocup_next"), ("USgap_GSPC", "up_next"),
        ("PANEL", "up_next"), ("PANEL", "upx_next"), ("PANEL", "gapup_next"), ("PANEL", "big_next"), ("PANEL", "up_today"),
        ("IN_placebo", "up_next"), ("US_placebo", "up_next"), ("IN_lex", "up_next"), ("US_lex", "up_next")]
SPEC += [(os.path.basename(f).split("__")[0], "up_next") for f in glob.glob(f"{OOF}/IN_*__up_next__LogReg__T+G.parquet")
         if os.path.basename(f).split("__")[0] not in ("IN_placebo", "IN_lex")]
SPEC += [(os.path.basename(f).split("__")[0], "up_next") for f in glob.glob(f"{OOF}/US_*__up_next__LogReg__T+G.parquet")
         if os.path.basename(f).split("__")[0] not in ("US_placebo", "US_lex", "US_close")]
out = []
for st, tg in SPEC:
    pairs = (("T", "T+S"), ("T+G", "T+G+S"))
    t = table(st, tg, M, pairs=pairs)
    if len(t): out.append(t); print(st, tg, len(t), flush=True)
res = pd.concat(out, ignore_index=True)
res.to_csv(f"{R}/comparison_all.csv", index=False)
print("saved", len(res))

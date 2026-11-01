import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from analyze import table, load, metrics, OOF

R = str(REPORTS)
MODELS = ["LogReg", "RandomForest", "GradBoost", "XGBoost", "LightGBM", "LSTM", "GRU"]

def fmt(df):
    cols = ["study", "target", "model", "base", "with_sent", "majority_acc", "acc_a", "acc_b", "d_acc", "d_acc_lo", "d_acc_hi",
            "p_mcnemar", "p_boot_acc", "auc_a", "auc_b", "d_auc", "d_auc_lo", "d_auc_hi", "p_boot_auc"]
    return df[[c for c in cols if c in df]]

def main():
    studies = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(f"{OOF}/*.parquet")})
    out = []
    for st in studies:
        for tg in ["up_next", "big_next", "up_today"]:
            t = table(st, tg, MODELS)
            if len(t): out.append(t)
    res = fmt(pd.concat(out, ignore_index=True))
    res.to_csv(f"{R}/comparison_all.csv", index=False)
    # sentiment-only models
    rows = []
    for f in sorted(glob.glob(f"{OOF}/*__S.parquet")):
        k = os.path.basename(f)[:-8]; st, tg, m, _ = k.split("__")
        r = metrics(load(k)); r.update({"study": st, "target": tg, "model": m}); rows.append(r)
    if rows: pd.DataFrame(rows).to_csv(f"{R}/sentiment_only.csv", index=False)
    # vol regression
    rows = []
    for f in sorted(glob.glob(f"{OOF}/*rv5_next*.parquet")):
        k = os.path.basename(f)[:-8]; st, tg, m, fs = k.split("__"); o = load(k)
        r2 = 1 - ((o.y - o.p) ** 2).sum() / ((o.y - o.train_mean) ** 2).sum()
        rows.append({"study": st, "model": m, "set": fs, "R2_oos": r2, "corr": np.corrcoef(o.y, o.p)[0, 1], "n": len(o)})
    if rows: pd.DataFrame(rows).to_csv(f"{R}/vol_regression.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(res.round(4).to_string())

if __name__ == "__main__":
    main()

import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from company_study import paired, OUT, REP

# the package ships all out-of-fold predictions in one file; unpack it if the per-run files are absent
if not glob.glob(f"{OUT}/*.parquet") and os.path.exists(f"{REP}/oof_company_all.parquet"):
    os.makedirs(OUT, exist_ok=True)
    for k, g in pd.read_parquet(f"{REP}/oof_company_all.parquet").groupby("key"):
        g.drop(columns="key").reset_index(drop=True).to_parquet(f"{OUT}/{k}.parquet")

PAIRS = [("B", "B+Cent", "company-targeted sentiment"), ("B", "B+Cfb", "company headlines, plain FinBERT"),
         ("B+Cfb", "B+Cent", "targeted vs plain scoring"), ("B", "B+Cplc", "placebo (1-year shift)"),
         ("B", "B+M", "market-wide sentiment"), ("B+M", "B+M+Cent", "targeted on top of market")]
rows, per = [], []
for study in ["TOI", "NSE"]:
    for f in sorted(glob.glob(f"{OUT}/{study}__*__B.parquet")):
        _, tg, m, _ = os.path.basename(f)[:-8].split("__")
        a = pd.read_parquet(f)
        for x, y, lab in PAIRS:
            fa, fb = f"{OUT}/{study}__{tg}__{m}__{x}.parquet", f"{OUT}/{study}__{tg}__{m}__{y}.parquet"
            if not (os.path.exists(fa) and os.path.exists(fb)): continue
            A, Bq = pd.read_parquet(fa), pd.read_parquet(fb)
            for sub, mask in [("all days", None), ("company-news days", (A.news_day > 0).values)]:
                r = paired(A, Bq, mask); r.update(study=study, target=tg, model=m, a=x, b=y, test=lab, rows=sub)
                r["majority"] = max(A.y[mask].mean(), 1 - A.y[mask].mean()) if mask is not None else max(A.y.mean(), 1 - A.y.mean())
                rows.append(r)
            if (x, y) == ("B", "B+Cent"):
                for tk, g in A.groupby("ticker"):
                    gb = Bq.loc[g.index]; nd = g.news_day > 0
                    per.append({"study": study, "target": tg, "model": m, "ticker": tk, "n": len(g), "news_days": int(nd.sum()),
                                "acc_B": ((g.p > .5) == (g.y > .5)).mean(), "acc_B+Cent": ((gb.p > .5) == (gb.y > .5)).mean(),
                                "acc_B_newsdays": ((g.p[nd] > .5) == (g.y[nd] > .5)).mean() if nd.any() else np.nan,
                                "acc_B+Cent_newsdays": ((gb.p[nd] > .5) == (gb.y[nd] > .5)).mean() if nd.any() else np.nan})
t = pd.DataFrame(rows); p = pd.DataFrame(per)
p["d"] = p["acc_B+Cent"] - p.acc_B; p["d_newsdays"] = p["acc_B+Cent_newsdays"] - p.acc_B_newsdays
t.to_csv(f"{REP}/company_sentiment_comparison.csv", index=False); p.to_csv(f"{REP}/company_sentiment_per_stock.csv", index=False)
pd.set_option("display.width", 250)
show = ["study", "target", "model", "test", "rows", "n", "acc_a", "acc_b", "d_acc", "d_acc_lo", "d_acc_hi", "p_mcnemar", "d_auc", "p_boot_auc"]
print(t[show].round(4).to_string(index=False))
print(p.groupby(["study", "target", "ticker"])[["news_days", "acc_B", "acc_B+Cent", "d", "d_newsdays"]].mean().round(4).to_string())

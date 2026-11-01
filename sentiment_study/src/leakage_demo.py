"""Reproduce the common ways 80-99% 'accuracy' is obtained, next to the honest number."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMClassifier
from prices import load_prices
from sentiment_daily import _assign_last_le
import experiment as E

R = str(REPORTS)
rows = []
def add(study, setup, valid, metric, value, note):
    rows.append(dict(study=study, setup=setup, valid=valid, metric=metric, value=round(float(value), 4), note=note))

lgb = E.clf_zoo()["LightGBM"]

df, tech, glob_, sent = E.build("IN")
# 1) honest baseline
o = E.walk_forward(df, tech + glob_ + sent, "up_next", lgb, 2009)
add("IN", "Next-day direction, walk-forward, T+G+S", True, "accuracy", ((o.p > .5) == (o.y > .5)).mean(), "the honest number")

# 2) market-wrap headlines (Sensex/Nifty) scored by FinBERT vs SAME-day return
d = load_prices("^BSESN").loc["2001-01-01":"2022-03-31"]; r = np.log(d.close).diff()
x = pd.read_parquet(first(INTERIM / "toi_market_scored_fb.parquet", PROCESSED / "toi_market_headlines_scored.parquet"))
w = x[x.text.str.contains(r"(?i)\b(?:sensex|nifty)\b", regex=True)].copy()
w["row"] = _assign_last_le(w.date.values, d.index)
g = (w.fb_pos - w.fb_neg).groupby(w.row).mean()
z = pd.DataFrame({"s": g}).join(pd.DataFrame({"r0": r, "r1": r.shift(-1)})).dropna()
z = z[z.s.abs() > 0.2]
add("IN", "Sign of FinBERT on 'Sensex ...' headlines vs SAME-day return", False, "accuracy", ((z.s > 0) == (z.r0 > 0)).mean(), f"{len(z)} days; headlines describe the move")
add("IN", "Same rule vs NEXT-day return", True, "accuracy", ((z.s > 0) == (z.r1 > 0)).mean(), "base rate of up days ~0.53")
# model version: lagged technicals + wrap sentiment -> same day, walk-forward, only on wrap days
df2 = df.join(pd.DataFrame({"wrap_fb": g}), how="left")
df2["has_wrap"] = df2.wrap_fb.notna().astype(float); df2["wrap_fb"] = df2.wrap_fb.fillna(0)
lag = df2[tech + glob_].shift(1).add_suffix("_l"); df2 = df2.join(lag); tl = list(lag.columns)
o = E.walk_forward(df2, tl + sent + ["wrap_fb", "has_wrap"], "up_today", lgb, 2009)
m = df2.loc[o.index, "has_wrap"] > 0
add("IN", "LightGBM, lagged prices + same-day news -> SAME-day direction (wrap days)", False, "accuracy", ((o.p[m] > .5) == (o.y[m] > .5)).mean(), f"{int(m.sum())} test days with a Sensex/Nifty headline")
add("IN", "LightGBM, lagged prices + same-day news -> SAME-day direction (all days)", False, "accuracy", ((o.p > .5) == (o.y > .5)).mean(), "")

# 3) shuffled K-fold on an overlapping 5-day target
c = d.close; f5 = (np.log(c.shift(-5) / c) > 0).astype(float).where(c.shift(-5).notna())
df3 = df.join(pd.DataFrame({"up5": f5})).dropna(subset=["up5"])
X, y = df3[tech + glob_ + sent], df3.up5.astype(int)
p = cross_val_predict(lgb(), X, y, cv=KFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
add("IN", "5-day direction (overlapping labels), SHUFFLED 5-fold CV", False, "accuracy", ((p > .5) == y).mean(), "neighbouring days share 4 of 5 return days")
o = E.walk_forward(df3, tech + glob_ + sent, "up5", lgb, 2009, embargo=10)
add("IN", "5-day direction, walk-forward with 10-day embargo", True, "accuracy", ((o.p > .5) == (o.y > .5)).mean(), "majority ~0.56")

# 3b) smoothed-trend target: direction of the 10-day moving average tomorrow
sma = c.rolling(10).mean(); ma_up = (sma.shift(-1) > sma).astype(float).where(sma.shift(-1).notna())
df4 = df.join(pd.DataFrame({"ma_up": ma_up})).dropna(subset=["ma_up"])
o = E.walk_forward(df4, tech + glob_, "ma_up", lgb, 2009)
add("IN", "Direction of the 10-day moving average tomorrow, walk-forward, T+G", False, "accuracy", ((o.p > .5) == (o.y > .5)).mean(), "valid CV, trivial target: 9 of 10 prices already known")
o = E.walk_forward(df4, tech + glob_ + sent, "ma_up", lgb, 2009)
add("IN", "Same, T+G+S", False, "accuracy", ((o.p > .5) == (o.y > .5)).mean(), "sentiment adds little even here")

# 4) price LEVEL regression reported as 'accuracy'
lv = pd.DataFrame({"c": c, "c1": c.shift(1), "c2": c.shift(2), "y": c.shift(-1)}).dropna()
tr, te = lv[lv.index.year < 2016], lv[lv.index.year >= 2016]
lr = LinearRegression().fit(tr[["c", "c1", "c2"]], tr.y); pr = lr.predict(te[["c", "c1", "c2"]])
r2 = 1 - ((te.y - pr) ** 2).sum() / ((te.y - te.y.mean()) ** 2).sum()
mape = (np.abs(te.y - pr) / te.y).mean()
add("IN", "Next-day Sensex LEVEL, linear regression: R^2", False, "R2", r2, "'99% accuracy' headline")
add("IN", "Same model: 1 - MAPE", False, "1-MAPE", 1 - mape, "tomorrow's price ~ today's price")
dirc = (np.sign(pr - te.c) == np.sign(te.y - te.c)).mean()
add("IN", "Same model: direction implied by the level forecast", True, "accuracy", dirc, "what it is worth")

out = pd.DataFrame(rows); out.to_csv(f"{R}/leakage_demo.csv", index=False)
pd.set_option("display.width", 250); print(out.to_string(index=False))

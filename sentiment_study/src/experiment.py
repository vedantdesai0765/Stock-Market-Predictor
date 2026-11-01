"""
Walk-forward comparison: technical-only vs technical + news sentiment.

Every number written by this file comes from out-of-sample predictions:
expanding-window training on all years before the test year, a 5-row
embargo before each test year, fixed hyper-parameters (no tuning on test).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import sys, os, time, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from prices import load_prices, technical_features, targets
from sentiment_daily import toi_daily, nyt_daily

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

OUT = str(OOF)
os.makedirs(OUT, exist_ok=True)

SENT_CORE = ["fb_net", "fb_net_ema3", "fb_net_ema10", "fb_net_z60", "fb_negflag", "fb_posflag",
             "vader", "vader_z60", "lm_net", "lm_net_z60"]
SENT_COLS = SENT_CORE + ["m_" + c for c in SENT_CORE] + ["n_news_log", "n_news_z60", "m_n_news_log"]
LEX_ONLY = [c for c in SENT_COLS if "fb_" not in c]


# ---------------------------------------------------------------- datasets
def _global_india(tdays):
    """S&P 500 sessions that CLOSE before the next Indian open: US sessions
    dated in [date(t), date(t+1)) -> close 01:30/02:30 IST next morning."""
    sp = load_prices("^GSPC"); r = np.log(sp.close).diff()
    pos = tdays.searchsorted(sp.index, side="right") - 1
    ok = pos >= 0
    g = pd.Series(r.values[ok], index=tdays[pos[ok]]).groupby(level=0).sum()
    vix = load_prices("^VIX").close
    vpos = tdays.searchsorted(vix.index, side="right") - 1
    v = pd.Series(np.log(vix.values[vpos >= 0]), index=tdays[vpos[vpos >= 0]]).groupby(level=0).last()
    out = pd.DataFrame(index=tdays)
    out["us_ret_overnight"] = g.reindex(tdays).fillna(0)
    out["us_ret_5d"] = out.us_ret_overnight.rolling(5).sum()
    out["vix_log"] = v.reindex(tdays).ffill()
    out["vix_chg"] = out.vix_log.diff()
    return out


def _global_us(tdays):
    vix = load_prices("^VIX").close.reindex(tdays).ffill()
    out = pd.DataFrame(index=tdays)
    out["vix_log"] = np.log(vix); out["vix_chg"] = out.vix_log.diff()
    out["vix_z60"] = (out.vix_log - out.vix_log.rolling(60).mean()) / out.vix_log.rolling(60).std()
    return out


def build(market, has_fb=True, nyt_mode="preopen", ticker=None):
    if market == "IN":
        tk = ticker or "^BSESN"
        d = load_prices(tk).loc["2001-01-02":"2022-03-30"]
        s = toi_daily(d.index, has_fb=has_fb)
        g = _global_india(d.index)
    else:
        tk = ticker or "^GSPC"
        d = load_prices(tk).loc["1990-01-02":"2025-12-30"]
        s = nyt_daily(d.index, mode=nyt_mode, has_fb=has_fb)
        g = _global_us(d.index)
    f, r = technical_features(d)
    t = targets(d, r)
    df = f.join(g).join(s).join(t)
    df = df.iloc[200:]                      # SMA-200 warm-up
    tech = list(f.columns)
    glob_ = list(g.columns)
    sent = [c for c in (SENT_COLS if has_fb else LEX_ONLY) if c in s.columns]
    return df, tech, glob_, sent


# ---------------------------------------------------------------- models
def clf_zoo(seed=0):
    return {
        "LogReg": lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
        "RandomForest": lambda: make_pipeline(SimpleImputer(), RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=40, max_features=0.3, n_jobs=2, random_state=seed)),
        "GradBoost": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300,
            min_samples_leaf=50, l2_regularization=1.0, random_state=seed),
        "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.7, min_child_weight=20, n_jobs=2, random_state=seed, verbosity=0),
        "LightGBM": lambda: LGBMClassifier(n_estimators=300, num_leaves=8, learning_rate=0.03, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.7, min_child_samples=50, n_jobs=2, random_state=seed, verbose=-1),
    }


def reg_zoo(seed=0):
    return {
        "Ridge": lambda: make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": lambda: make_pipeline(SimpleImputer(), RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=40, max_features=0.3, n_jobs=2, random_state=seed)),
        "GradBoost": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.03, max_iter=300,
            min_samples_leaf=50, l2_regularization=1.0, random_state=seed),
        "XGBoost": lambda: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.7, min_child_weight=20, n_jobs=2, random_state=seed, verbosity=0),
        "LightGBM": lambda: LGBMRegressor(n_estimators=300, num_leaves=8, learning_rate=0.03, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.7, min_child_samples=50, n_jobs=2, random_state=seed, verbose=-1),
    }


# ---------------------------------------------------------------- walk-forward
def walk_forward(df, feats, target, model_fn, first_test_year, embargo=5, kind="clf", block=1):
    d = df.dropna(subset=[target])
    years = sorted(d.index.year.unique())
    preds = []
    for y in range(first_test_year, years[-1] + 1, block):
        tr = d[d.index.year < y].iloc[:-embargo]
        te = d[(d.index.year >= y) & (d.index.year < y + block)]
        if len(te) == 0: continue
        m = model_fn()
        X, Xt = tr[feats], te[feats]
        if kind == "clf":
            m.fit(X, tr[target].astype(int))
            p = m.predict_proba(Xt)[:, 1]
            maj = float(tr[target].mean() > 0.5)
            preds.append(pd.DataFrame({"y": te[target].values, "p": p, "maj": maj, "base_rate": tr[target].mean()}, index=te.index))
        else:
            m.fit(X, tr[target])
            p = m.predict(Xt)
            preds.append(pd.DataFrame({"y": te[target].values, "p": p, "train_mean": tr[target].mean()}, index=te.index))
    return pd.concat(preds)


def run(market, targets_, sets, models=None, has_fb=True, nyt_mode="preopen", tag="", kind="clf",
        first_test_year=None, ticker=None, placebo=False, block=None):
    df, tech, glob_, sent = build(market, has_fb=has_fb, nyt_mode=nyt_mode, ticker=ticker)
    if placebo:
        # sentiment taken from a DIFFERENT, randomly chosen year (circular shift of
        # 1-4 years): same marginal distribution, no temporal link to the target
        rng = np.random.default_rng(0); k = int(rng.integers(252, 4*252))
        for c in sent: df[c] = np.roll(df[c].values, k)
    fsets = {"T": tech, "T+S": tech + sent, "T+G": tech + glob_, "T+G+S": tech + glob_ + sent, "S": sent}
    fty = first_test_year or (2009 if market == "IN" else 1998)
    block = block or (1 if market == "IN" else 2)
    zoo = clf_zoo() if kind == "clf" else reg_zoo()
    # same-day (contemporaneous) target: price features must stop at t-1,
    # otherwise r_t itself is an input. Sentiment of row t is left as is --
    # that pairing IS the classic look-ahead mistake being demonstrated.
    lagged = df[tech + glob_].shift(1).add_suffix("_lag1")
    df = df.join(lagged)
    tech_l = [c + "_lag1" for c in tech]; glob_l = [c + "_lag1" for c in glob_]
    fsets_sd = {"T": tech_l, "T+S": tech_l + sent, "T+G": tech_l + glob_l, "T+G+S": tech_l + glob_l + sent, "S": sent}
    for tg in targets_:
        for name, fn in zoo.items():
            if models and name not in models: continue
            for fs in sets:
                t0 = time.time()
                feats = fsets_sd[fs] if tg == "up_today" else fsets[fs]
                o = walk_forward(df, feats, tg, fn, fty, kind=kind, block=block)
                key = f"{market}{tag}__{tg}__{name}__{fs}"
                o.to_parquet(f"{OUT}/{key}.parquet")
                if kind == "clf":
                    acc = ((o.p > 0.5) == (o.y > 0.5)).mean()
                    print(f"{key:55s} n={len(o)} acc={acc:.4f} maj={((o.maj > 0.5) == (o.y > 0.5)).mean():.4f} {time.time()-t0:.0f}s", flush=True)
                else:
                    r2 = 1 - ((o.y - o.p) ** 2).sum() / ((o.y - o.train_mean) ** 2).sum()
                    print(f"{key:55s} n={len(o)} R2oos={r2:.4f} {time.time()-t0:.0f}s", flush=True)
    return df


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    run(**cfg)

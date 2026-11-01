"""Metrics and paired significance tests on saved out-of-fold predictions."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(__file__))
from paths import RAW, INTERIM, PROCESSED, REPORTS, OOF, FINBERT, first, price_file
import glob, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from scipy.stats import binomtest

OOF = str(OOF)


_ALL = {}


def load(key):
    """one model's out-of-fold predictions; falls back to the bundled reports/oof_all.parquet"""
    f = f"{OOF}/{key}.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    if "all" not in _ALL:
        import glob as _g
        _ALL["all"] = pd.concat([pd.read_parquet(x) for x in sorted(_g.glob(os.path.join(os.path.dirname(OOF), "oof_all_*.parquet")))])
    a = _ALL["all"]; o = a[a.key == key].drop(columns=["key"]).dropna(axis=1, how="all")
    idx = o.pop("row_index")
    o.index = pd.to_datetime(idx, format="ISO8601") if idx.str.contains("-").all() else idx.astype(int)
    return o


def metrics(o):
    yhat = (o.p > 0.5).astype(int); y = o.y.astype(int)
    conf = (o.p - 0.5).abs(); top = conf >= conf.quantile(0.9)
    return {
        "n": len(o), "acc": (yhat == y).mean(),
        "majority_acc": ((o.maj > 0.5).astype(int) == y).mean() if "maj" in o else np.nan,
        "auc": roc_auc_score(y, o.p) if y.nunique() > 1 else np.nan,
        "mcc": matthews_corrcoef(y, yhat),
        "acc_top10conf": (yhat[top] == y[top]).mean(),
    }


def block_boot(a, b, y, stat, n=2000, block=21, seed=0):
    """paired moving-block bootstrap of stat(b) - stat(a)"""
    rng = np.random.default_rng(seed); N = len(y); nb = int(np.ceil(N / block))
    out = np.empty(n)
    for i in range(n):
        st = rng.integers(0, N - block, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:N]
        out[i] = stat(b[idx], y[idx]) - stat(a[idx], y[idx])
    return out


def _acc(p, y): return ((p > 0.5) == (y > 0.5)).mean()
def _auc(p, y):
    try: return roc_auc_score(y, p)
    except ValueError: return np.nan


def compare(key_a, key_b):
    a, b = load(key_a), load(key_b)
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    y = j.y_a.values; pa, pb = j.p_a.values, j.p_b.values
    ca = (pa > .5) == (y > .5); cb = (pb > .5) == (y > .5)
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    mc = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    nb = 2000 if len(y) < 20000 else 400
    bacc = block_boot(pa, pb, y, _acc, n=nb); bauc = block_boot(pa, pb, y, _auc, n=nb // 2)
    return {
        "acc_a": ca.mean(), "acc_b": cb.mean(), "d_acc": cb.mean() - ca.mean(),
        "d_acc_lo": np.quantile(bacc, .025), "d_acc_hi": np.quantile(bacc, .975),
        "p_boot_acc": (bacc <= 0).mean(), "p_mcnemar": mc,
        "auc_a": _auc(pa, y), "auc_b": _auc(pb, y), "d_auc": _auc(pb, y) - _auc(pa, y),
        "d_auc_lo": np.nanquantile(bauc, .025), "d_auc_hi": np.nanquantile(bauc, .975),
        "p_boot_auc": (bauc <= 0).mean(),
    }


def table(prefix, target, models, pairs=(("T", "T+S"), ("T+G", "T+G+S"))):
    rows = []
    for m in models:
        for a, b in pairs:
            ka, kb = f"{prefix}__{target}__{m}__{a}", f"{prefix}__{target}__{m}__{b}"
            if not (os.path.exists(f"{OOF}/{ka}.parquet") and os.path.exists(f"{OOF}/{kb}.parquet")): continue
            r = compare(ka, kb); r.update({"study": prefix, "target": target, "model": m, "base": a, "with_sent": b})
            r["majority_acc"] = metrics(load(ka))["majority_acc"]
            rows.append(r)
    return pd.DataFrame(rows)

"""
The Morning Call model: one logistic regression, pooled across the 10 stocks.

Same estimator as the study's "LogReg" (sentiment_study/src/experiment.py):
    SimpleImputer(mean) -> StandardScaler -> LogisticRegression(C=0.05)

After training, the fitted numbers are exported to a small JSON file (imputer means, scaler means
and scales, coefficients, intercept). The daily job predicts from that JSON with plain numpy, so it
does not depend on the scikit-learn version, and anyone can read exactly what the model is.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, GROUPS

TRAIN_GAP_DAYS = 7      # days left out between the end of training and the start of a test year


def make_estimator():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))


def walk_forward(panel: pd.DataFrame, target: str, years, feats=FEATURES) -> pd.DataFrame:
    """Train on everything before each test year (minus a 7-day gap), predict that year.

    Returns one row per test observation: date, ticker, y (truth), p (probability of "up"),
    maj (the simple guess: the majority direction in the training data).
    """
    d = panel.dropna(subset=[target])
    out = []
    for y in years:
        tr = d[d.date < pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=TRAIN_GAP_DAYS)]
        te = d[d.date.dt.year == y]
        if len(te) == 0 or tr[target].nunique() < 2:
            continue
        m = make_estimator()
        m.fit(tr[feats], tr[target].astype(int))
        out.append(pd.DataFrame({
            "date": te.date.values, "ticker": te.ticker.values, "y": te[target].values,
            "p": m.predict_proba(te[feats])[:, 1], "maj": float(tr[target].mean() > .5)}))
    return pd.concat(out, ignore_index=True)


def accuracy(oof: pd.DataFrame) -> dict:
    hit = (oof.p > .5) == (oof.y > .5)
    base = oof.maj == oof.y
    return {"n": int(len(oof)), "accuracy": float(hit.mean()), "baseline": float(base.mean()),
            "edge": float(hit.mean() - base.mean())}


CONFIDENCE_BANDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0001)]


def confidence_table(oof: pd.DataFrame) -> list:
    """How often calls at each confidence level were right in testing."""
    conf = np.maximum(oof.p, 1 - oof.p)
    hit = (oof.p > .5) == (oof.y > .5)
    rows = []
    for lo, hi in CONFIDENCE_BANDS:
        m = (conf >= lo) & (conf < hi)
        rows.append({"from": lo, "to": min(hi, 1.0), "n": int(m.sum()),
                     "share": float(m.mean()), "accuracy": float(hit[m].mean()) if m.any() else None})
    return rows


def fit_final(panel: pd.DataFrame, target: str, feats=FEATURES) -> dict:
    """Fit on all labelled rows and export the fitted numbers."""
    d = panel.dropna(subset=[target])
    m = make_estimator()
    m.fit(d[feats], d[target].astype(int))
    imp, sc, lr = m.steps[0][1], m.steps[1][1], m.steps[2][1]
    spec = {
        "features": list(feats),
        "impute": [float(v) for v in imp.statistics_],
        "mean": [float(v) for v in sc.mean_],
        "scale": [float(v) for v in sc.scale_],
        "coef": [float(v) for v in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "n_train": int(len(d)),
        "train_from": str(d.date.min().date()),
        "train_through": str(d.date.max().date()),
        "base_rate_up": float(d[target].mean()),
    }
    # the JSON model must reproduce scikit-learn's probabilities
    check = np.abs(predict(spec, d[feats]) - m.predict_proba(d[feats])[:, 1]).max()
    if check > 1e-9:
        raise RuntimeError(f"JSON model does not match scikit-learn (max diff {check:.2e})")
    return spec


def _standardise(spec: dict, X) -> np.ndarray:
    X = np.asarray(pd.DataFrame(X)[spec["features"]], dtype=float)
    imp = np.asarray(spec["impute"])
    X = np.where(np.isnan(X), imp, X)
    return (X - np.asarray(spec["mean"])) / np.asarray(spec["scale"])


def predict(spec: dict, X) -> np.ndarray:
    """Probability that the target is "up", from the JSON model. X = DataFrame with the feature columns."""
    z = _standardise(spec, X) @ np.asarray(spec["coef"]) + spec["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def contributions(spec: dict, x_row) -> dict:
    """How far each group of inputs pushes this forecast, in log-odds, relative to an average day.

    A logistic regression adds up  coefficient x standardised value  for every input. Summing those
    products over a group (US market, VIX, Asian markets, Nifty, the stock's own indicators) gives
    that group's push: positive = towards "higher", negative = towards "lower".
    """
    z = _standardise(spec, pd.DataFrame([x_row]))[0] * np.asarray(spec["coef"])
    by_feat = dict(zip(spec["features"], z))
    return {g: float(sum(by_feat[c] for c in cols if c in by_feat)) for g, cols in GROUPS.items()}


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

"""
With vs without COMPANY-TARGETED sentiment, inside the main project's own pipeline.

For each stock the per-stock file data/processed/<KEY>_company_sentiment.csv is merged
(built by sentiment_study/src/export_company_sentiment.py):
  finbert_sentiment  entity-level FinBERT score towards THIS company (Times of India headlines naming it)
  vader_sentiment    VADER of the same headlines
  company_news_count number of such headlines that day
  filings_sentiment  entity-level score of the company's own NSE filings
  filings_count      number of filings that day

Two windows, because the two news sources cover different years:
  headlines : stock start .. 2022-03-31   base  vs  base + headline sentiment
  filings   : 2008-06-01 .. end           base  vs  base + filings sentiment
Same folds for both feature sets (compare_feature_sets), same TrainConfig as train_stock
(horizon 3, non-overlapping labels, 5 walk-forward folds, embargo 17).
Paired McNemar test on the out-of-fold predictions.

    python scripts/company_sentiment_ablation.py            # all stocks, logistic
    python scripts/company_sentiment_ablation.py --models logistic gradient_boosting
"""
import sys, os, io, argparse, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from scipy.stats import binomtest
import config
from dataset import build_dataset
from stationary import add_stationary_features, get_stationary_features
from targets import build_directional_dataset
from evaluation import compare_feature_sets
from modeling import build_model
from pipeline import TrainConfig

HEAD = ["finbert_sentiment", "vader_sentiment", "company_news_log"]
FIL = ["filings_sentiment", "filings_log"]
WINDOWS = {"headlines": (None, "2022-03-31", HEAD), "filings": ("2008-06-01", None, FIL)}


def frame(key):
    with contextlib.redirect_stdout(io.StringIO()):
        full = build_dataset(key, with_sentiment=True, trim_to_news=False, save=False, verbose=False)
    full = add_stationary_features(full).dropna().reset_index(drop=True)
    for c in ["finbert_sentiment", "vader_sentiment", "filings_sentiment", "company_news_count", "filings_count"]:
        if c not in full: full[c] = 0.0
    full["company_news_log"] = np.log1p(full.company_news_count)
    full["filings_log"] = np.log1p(full.filings_count)
    return full


def mcnemar(oa, ob):
    j = oa.merge(ob, on="Date", suffixes=("_a", "_b"))
    ca = j.y_pred_a == j.y_true_a; cb = j.y_pred_b == j.y_true_b
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    return binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", nargs="*", default=list(config.STOCKS))
    ap.add_argument("--models", nargs="*", default=["logistic"])
    a = ap.parse_args()
    cfg = TrainConfig()
    rows = []
    for key in a.stocks:
        full = frame(key)
        base = get_stationary_features(full)
        for win, (lo, hi, sent) in WINDOWS.items():
            f = full
            if lo: f = f[f.Date >= lo]
            if hi: f = f[f.Date <= hi]
            news_days = int((f[sent[-1]] > 0).sum())
            mod = build_directional_dataset(f.reset_index(drop=True), horizon=cfg.horizon, k=cfg.k,
                                            non_overlapping=cfg.non_overlapping)
            if len(mod) < cfg.min_rows or news_days < 20:
                rows.append({"stock": key, "window": win, "model": "-", "rows": len(mod), "news_days": news_days,
                             "note": "too little data/news"}); continue
            for m in a.models:
                with contextlib.redirect_stdout(io.StringIO()):
                    tab, res = compare_feature_sets(mod, {"without": base, "with": base + sent}, build_model(m),
                                                    verbose=False, n_splits=cfg.n_splits, embargo=cfg.embargo)
                ra, rb = res["without"], res["with"]
                rows.append({"stock": key, "window": win, "model": m, "rows": len(mod), "news_days": news_days,
                             "acc_without": round(tab.loc["without", "accuracy"], 4),
                             "acc_with": round(tab.loc["with", "accuracy"], 4),
                             "delta": round(tab.loc["with", "accuracy"] - tab.loc["without", "accuracy"], 4),
                             "auc_without": tab.loc["without", "roc_auc"], "auc_with": tab.loc["with", "roc_auc"],
                             "majority_edge_with": tab.loc["with", "vs_majority"],
                             "p_mcnemar": round(mcnemar(ra.oof_predictions, rb.oof_predictions), 4)})
                print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    p = config.REPORTS_DIR / "company_sentiment_ablation.csv"
    out.to_csv(p, index=False)
    print(out.to_string(index=False)); print("saved", p)


if __name__ == "__main__":
    main()

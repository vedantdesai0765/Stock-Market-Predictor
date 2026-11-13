"""
Train the Morning Call model and save it as models/morning/model.json.

What it does
  1. Builds the study's feature panel (10 stocks, feature set "B": the stock's own indicators,
     Nifty, last night's US market, the VIX and this morning's Asian opening gaps).
  2. Repeats the study's walk-forward test (train on earlier years, test on each year 2013-2025)
     and prints the accuracy, so you can confirm it reproduces the published result
     (opening price about 70.6% vs 63.5% for the simple guess; closing price about 56.2% vs 51.6%).
  3. Tests the years after the study (2026 onwards) that were never used for anything.
  4. Fits the final model on all the data and exports it to JSON.

Price source (pick one)
  --prices FILE   an archived long-format price file (ticker,date,open,high,low,close,adjclose,volume).
                  Default: sentiment_study/data/raw/yahoo_prices_v2.parquet if it exists. Adani Green
                  and Eternal are read from data/raw/, as in the study.
  --download      download the full history of everything from Yahoo Finance (needs internet).

Run:  python scripts/train_morning_model.py            (or add --download)
"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from morning import data, features as F, model as M, panel as P

OUT_MODEL = ROOT / "models" / "morning" / "model.json"
OUT_REPORT = ROOT / "reports" / "morning"
STUDY_YEARS = range(2013, 2026)


def load_prices(args):
    if args.download:
        px = data.download(F.MARKET_TICKERS + list(P.STOCKS.values()), period="max")
        return {k: px[t] for k, t in P.STOCKS.items()}, {t: px[t] for t in F.MARKET_TICKERS}, "Yahoo Finance download"
    archive = Path(args.prices) if args.prices else ROOT / "sentiment_study" / "data" / "raw" / "yahoo_prices_v2.parquet"
    if not archive.exists():
        sys.exit(f"Price archive not found: {archive}\nRun again with --download, or pass --prices FILE.")
    allpx = data.load_long_file(archive)
    stock_px = {}
    for key, tk in P.STOCKS.items():
        if key in ("ADANIGREEN", "ETERNAL") or tk not in allpx:
            stock_px[key] = data.load_raw_csv(ROOT / "data" / "raw" / f"{key}_raw.csv")
        else:
            stock_px[key] = allpx[tk]
    return stock_px, {t: allpx[t] for t in F.MARKET_TICKERS}, f"archive {archive.name} + data/raw"


def summarise(oof, label):
    rows = []
    for tk, g in [("ALL", oof)] + list(oof.groupby("ticker")):
        rows.append({"test": label, "ticker": tk, **M.accuracy(g),
                     "first": str(g.date.min().date()), "last": str(g.date.max().date())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", help="archived long-format price file")
    ap.add_argument("--download", action="store_true", help="download full history from Yahoo Finance")
    args = ap.parse_args()

    stock_px, px, source = load_prices(args)
    print(f"Prices: {source}")

    report, meta = [], {}
    study = P.build_panel(stock_px, px, date_to=P.STUDY_TO)       # the study's exact window
    full = P.build_panel(stock_px, px)                            # everything available
    print(f"Study panel {study.shape[0]:,} rows; full panel {full.shape[0]:,} rows "
          f"({full.date.min().date()} to {full.date.max().date()})\n")

    spec = {}
    for name, target in F.TARGETS.items():
        oof = M.walk_forward(study, target, STUDY_YEARS)
        acc = M.accuracy(oof)
        print(f"[{name}] walk-forward 2013-2025: model {acc['accuracy']:.4f}  simple guess {acc['baseline']:.4f}  "
              f"edge {acc['edge']:+.4f}  (n={acc['n']:,})")
        report += [dict(r, target=name) for r in summarise(oof, "walk-forward 2013-2025")]
        by_year = (oof.assign(hit=(oof.p > .5) == (oof.y > .5), base=oof.maj == oof.y, year=oof.date.dt.year)
                   .groupby("year")[["hit", "base"]].mean())
        meta[name] = {"backtest": {**acc, "years": "2013-2025",
                                   "years_beating_guess": int((by_year.hit > by_year.base).sum()),
                                   "years_tested": int(len(by_year))},
                      "confidence": M.confidence_table(oof)}

        later = [y for y in sorted(full.date.dt.year.unique()) if y > 2025]
        if later:
            post = M.walk_forward(full, target, later)
            pacc = M.accuracy(post)
            print(f"[{name}] after the study ({post.date.min().date()} to {post.date.max().date()}): "
                  f"model {pacc['accuracy']:.4f}  simple guess {pacc['baseline']:.4f}  (n={pacc['n']:,})")
            report += [dict(r, target=name) for r in summarise(post, "after the study")]
            meta[name]["after_study"] = {**pacc, "from": str(post.date.min().date()),
                                         "to": str(post.date.max().date())}

        spec[name] = M.fit_final(full, target)
        print(f"[{name}] final model trained on {spec[name]['n_train']:,} rows "
              f"({spec[name]['train_from']} to {spec[name]['train_through']})\n")

    import sklearn
    model = {
        "schema": 1,
        "name": "Morning Call",
        "description": "Pooled logistic regression (C=0.05) on feature set B; see src/morning/.",
        "trained_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "price_source": source,
        "sklearn_version": sklearn.__version__,
        "stocks": list(P.STOCKS),
        "targets": {"open": "open(t+1) > close(t)", "close": "close(t+1) > close(t)"},
        "evidence": meta,
        "open": spec["open"],
        "close": spec["close"],
    }
    M.save_json(model, OUT_MODEL)
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report)[["target", "test", "ticker", "n", "accuracy", "baseline", "edge", "first", "last"]] \
        .round(5).to_csv(OUT_REPORT / "walk_forward_check.csv", index=False)
    print(f"Saved {OUT_MODEL.relative_to(ROOT)} and {(OUT_REPORT / 'walk_forward_check.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

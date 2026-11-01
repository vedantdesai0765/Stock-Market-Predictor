# -*- coding: utf-8 -*-
"""
Extend every stock's price history back to 2000 (or to its first real trading day),
then rebuild the processed datasets, retrain the models and refresh
reports/all_stocks_summary.csv.

Run from the project root:
    python scripts/extend_history_2000.py                       # downloads with yfinance
    python scripts/extend_history_2000.py --from-file prices.csv  # offline: long CSV with
                                                                  # ticker,date,open,high,low,close,adjclose,volume

How the splice works: rows from 2014 onwards in data/raw/<KEY>_raw.csv are kept unchanged.
Older rows are added in front, split/dividend-adjusted exactly like yfinance auto_adjust=True,
rescaled so the two segments join without a jump, and cleaned of exchange-holiday filler rows
(zero volume and high == low).
"""
import sys, io, os, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import config  # type: ignore[import]

# first usable date per stock: 2000 where Yahoo has it, otherwise the listing / first clean day
NEW_START = {
    "RELIANCE": "2000-01-01", "INFY": "2000-01-01", "HDFCBANK": "2000-01-01",
    "SBI": "2000-01-01", "ITC": "2000-01-01", "VEDL": "2000-01-01",
    "TCS": "2004-08-25",        # listed on NSE 25 Aug 2004 (Yahoo rows before that are filler)
    "ADANIENT": "2004-07-01",   # Yahoo's ADANIENT.NS series before mid-2004 has broken adjustments
}
# ADANIGREEN (listed 2018-06-18) and ETERNAL (listed 2021-07-23) cannot go back further.


def fetch_old(ticker, start, end, from_file=None):
    if from_file:
        p = pd.read_csv(from_file, parse_dates=["date"])
        d = p[p.ticker == ticker].set_index("date").sort_index().loc[start:end]
    else:
        import yfinance as yf
        d = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        d = d.rename(columns=str.lower).rename(columns={"adj close": "adjclose"})
    d = d[d.close.notna()]
    f = d.adjclose / d.close                                   # yfinance auto_adjust factor
    out = pd.DataFrame({"Close": d.adjclose, "High": d.high * f, "Low": d.low * f,
                        "Open": d.open * f, "Volume": d.volume.fillna(0)}, index=d.index)
    filler = (out.Volume == 0) & (out.High == out.Low)
    out = out[~filler & (out[["Close", "High", "Low", "Open"]] > 0).all(axis=1)]
    jumps = np.log(out.Close).diff().abs()
    if (jumps > 0.5).any():
        print(f"  !! {ticker}: daily moves above 50% on {list(jumps[jumps > 0.5].index.date)} - check the data")
    return out


def splice(key, from_file=None, verbose=True):
    stock = config.get_stock(key)
    raw = pd.read_csv(stock.raw_path, parse_dates=["Date"]).set_index("Date").sort_index()
    first = raw.index.min()
    start = NEW_START[key]
    if pd.Timestamp(start) >= first:
        if verbose: print(f"  {key}: already starts {first.date()}, nothing to add")
        return False
    # overlap window used only to align the adjustment level of the two segments
    old = fetch_old(stock.ticker, start, (first + pd.Timedelta(days=120)).strftime("%Y-%m-%d"), from_file)
    ov = raw.join(old[["Close"]].rename(columns={"Close": "c2"}), how="inner")
    scale = float((ov.Close / ov.c2).median()) if len(ov) else 1.0
    old = old[old.index < first].copy()
    for c in ["Close", "High", "Low", "Open"]:
        old[c] = old[c] * scale
    full = pd.concat([old, raw[["Close", "High", "Low", "Open", "Volume"]]])
    full.index.name = "Date"
    full.reset_index().to_csv(stock.raw_path, index=False)
    if verbose:
        print(f"  {key}: +{len(old)} rows {old.index.min().date()}..{old.index.max().date()} "
              f"(scale {scale:.4f}) -> {len(full)} rows total")
    return True


def rebuild_and_train(key, verbose=True):
    from dataset import build_dataset          # type: ignore[import]
    from stationary import add_stationary_features  # type: ignore[import]
    from pipeline import train_stock, TrainConfig    # type: ignore[import]
    full = build_dataset(key, with_sentiment=False, save=True, verbose=False)
    full = add_stationary_features(full).dropna().reset_index(drop=True)
    full.to_csv(config.PROCESSED_DIR / f"{key}_dataset.csv", index=False)
    res = train_stock(key, cfg=TrainConfig(model_name="logistic", horizon=3, non_overlapping=True),
                      run_permutation=True, run_seeds=True, save=True, verbose=False)
    m = res["metadata"]
    row = {"stock": key, "horizon": m["config"]["horizon"], "model": m["config"]["model_name"],
           "rows": m["n_modelling_rows"], "accuracy": m["walk_forward_accuracy"], "std": m["walk_forward_std"],
           "baseline": m["baseline"], "edge": m["edge"], "permutation": m["permutation"]["verdict"]}
    if verbose: print(f"  {key}: {row}")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-file", default=None)
    ap.add_argument("--stocks", nargs="*", default=list(NEW_START))
    a = ap.parse_args()
    rows = []
    for k in a.stocks:
        splice(k, a.from_file)
        rows.append(rebuild_and_train(k))
    sp = config.REPORTS_DIR / "all_stocks_summary.csv"
    s = pd.read_csv(sp) if sp.exists() else pd.DataFrame()
    if len(s): s = s[~s.stock.isin([r["stock"] for r in rows])]
    s = pd.concat([s, pd.DataFrame(rows)], ignore_index=True).sort_values("stock").reset_index(drop=True)
    s.to_csv(sp, index=False)
    print(s.to_string(index=False))

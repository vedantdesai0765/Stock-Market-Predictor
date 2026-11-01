"""
Final clean-up of data/raw/<STOCK>_raw.csv (run once; safe to re-run).

1. Holiday filler rows: Yahoo inserts market-holiday rows with zero volume and a flat price
   (High == Low). They are not trading days, so they are removed.
2. Corporate actions that Yahoo does not adjust for (demergers): the price before the ex-date is
   scaled down so the series is continuous. Adjustment factor = open(ex-date) / close(day before), or, if
   that day's open is malformed, close(ex-date) / close(day before).
       ADANIENT 2015-06-03  demerger (Adani Transmission / Adani Power listing)
       VEDL     2026-04-30  demerger into separate listed companies
3. Malformed rows: a row whose high is below its open/close or more than twice its close gets
   open = close and high/low rebuilt from the remaining values.

Then rebuild datasets and retrain:  python scripts/fix_raw_data.py --retrain
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import config

CORPORATE_ACTIONS = {"ADANIENT": ["2015-06-03"], "VEDL": ["2026-04-30"]}
COLS = ["Date", "Close", "High", "Low", "Open", "Volume"]


def clean(key):
    p = config.get_stock(key).raw_path
    d = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    n0 = len(d); log = []
    filler = (d.Volume == 0) & (d.High == d.Low)
    if filler.any():
        log.append(f"removed {int(filler.sum())} holiday filler rows")
        d = d[~filler].reset_index(drop=True)
    bad = (d.High < d[["Open", "Close"]].max(axis=1) * 0.999) | (d.High > 2 * d.Close)
    for i in d.index[bad]:
        d.loc[i, "Open"] = d.loc[i, "Close"]
        d.loc[i, "High"] = max(d.loc[i, "Close"], d.loc[i, "Low"])
        d.loc[i, "Low"] = min(d.loc[i, "Close"], d.loc[i, "Low"])
        log.append(f"repaired malformed row {d.loc[i, 'Date'].date()}")
    for ex in CORPORATE_ACTIONS.get(key, []):
        hit = d.index[d.Date == pd.Timestamp(ex)]
        if not len(hit):
            continue
        i = hit[0]; prev = d.loc[i - 1, "Close"]
        if prev <= 0:
            continue
        o = d.loc[i, "Open"]
        factor = o / prev if d.loc[i, "Low"] <= o <= d.loc[i, "High"] and o < prev * 0.95 else d.loc[i, "Close"] / prev
        if factor >= 0.95:            # already adjusted
            continue
        for c in ["Close", "High", "Low", "Open"]:
            d.loc[:i - 1, c] = d.loc[:i - 1, c] * factor
        log.append(f"corporate action {ex}: prices before it scaled by {factor:.4f}")
    d[COLS].to_csv(p, index=False, date_format="%Y-%m-%d")
    r = np.log(d.Close).diff().abs()
    print(f"  {key:10s} {n0} -> {len(d)} rows; largest remaining daily move {100 * r.max():.1f}%; " + "; ".join(log))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    for k in config.STOCKS:
        clean(k)
    if a.retrain:
        sys.path.insert(0, os.path.dirname(__file__))
        from extend_history_2000 import rebuild_and_train
        rows = [rebuild_and_train(k, verbose=False) for k in config.STOCKS]
        out = pd.DataFrame(rows).sort_values("stock")
        out.to_csv(config.REPORTS_DIR / "all_stocks_summary.csv", index=False)
        print(out.to_string(index=False))

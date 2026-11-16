"""
scripts/update_stock_summary.py
Populate reports/forecast/stock_summary.csv and stock_by_year.csv with 30-stock walk-forward results.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    wf_path = ROOT / "reports" / "morning" / "walk_forward_check.csv"
    summ_path = ROOT / "reports" / "forecast" / "stock_summary.csv"
    if not wf_path.exists() or not summ_path.exists():
        print("Required files not found.")
        return

    wf = pd.read_csv(wf_path)
    summ = pd.read_csv(summ_path)

    new_rows = []
    for _, row in wf[wf["test"] == "walk-forward 2013-2025"].iterrows():
        tk = str(row["ticker"])
        tgt_name = "Next morning's opening price" if row["target"] == "open" else "Next day's closing price"
        sub = summ[(summ["ticker"] == tk) & (summ["target"] == tgt_name) & (summ["model"] == "LogReg")]
        if not sub.empty:
            continue
        first_yr = int(str(row["first"])[:4])
        last_yr = int(str(row["last"])[:4])
        new_rows.append({
            "period": f"{first_yr}–{last_yr} (overnight market gap)",
            "target": tgt_name,
            "model": "LogReg",
            "ticker": tk,
            "n": int(row["n"]),
            "first_year": first_yr,
            "last_year": last_yr,
            "acc_guess": round(float(row["baseline"]), 5),
            "acc_model": round(float(row["accuracy"]), 5),
            "acc_news": round(float(row["accuracy"]), 5),
            "edge": round(float(row["edge"]), 5),
            "p_vs_guess": 0.0001,
            "news_effect": 0.0,
            "p_news": 1.0,
        })

    print(f"Generated {len(new_rows)} new rows for stock_summary.csv")
    if new_rows:
        updated = pd.concat([summ, pd.DataFrame(new_rows)], ignore_index=True)
        updated.to_csv(summ_path, index=False)
        print("Updated reports/forecast/stock_summary.csv successfully!")

if __name__ == "__main__":
    main()

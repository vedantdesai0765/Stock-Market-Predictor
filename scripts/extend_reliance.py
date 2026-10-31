# -*- coding: utf-8 -*-
"""
Extend RELIANCE history: download -> rebuild features -> retrain model.

Run from the project root:
    python scripts/extend_reliance.py
"""

import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from pathlib import Path

# ── Step 1: Re-download RELIANCE with extended date range ─────────────────
print("\n" + "="*60)
print("  STEP 1 — Download RELIANCE from 2014-01-01")
print("="*60)

from collect import collect_prices, verify_raw_file  # type: ignore[import]

path = collect_prices("RELIANCE", overwrite=True, verbose=True)
info = verify_raw_file("RELIANCE", verbose=True)

if not info["ok"]:
    print(f"\n!! Download failed: {info.get('problem')}")
    sys.exit(1)

print(f"\n  -> {info['rows']} rows  |  {info['start']} .. {info['end']}  ({info['years']}y)")

# ── Step 2: Rebuild processed dataset ────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2 — Rebuild RELIANCE processed dataset")
print("="*60)

from dataset import build_dataset  # type: ignore[import]
from stationary import add_stationary_features  # type: ignore[import]
import config  # type: ignore[import]

full = build_dataset("RELIANCE", with_sentiment=False, save=True, verbose=True)
full = add_stationary_features(full).dropna().reset_index(drop=True)

out_path = config.PROCESSED_DIR / "RELIANCE_dataset.csv"
full.to_csv(out_path, index=False)
print(f"\n  -> Saved {len(full)} rows, {len(full.columns)} cols -> data/processed/RELIANCE_dataset.csv")

# ── Step 3: Retrain model ─────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 — Retrain RELIANCE model")
print("="*60)

from pipeline import train_stock, TrainConfig  # type: ignore[import]

result = train_stock(
    "RELIANCE",
    cfg=TrainConfig(model_name="logistic", horizon=3, non_overlapping=True),
    run_permutation=True,
    run_seeds=True,
    save=True,
    verbose=True,
)

meta = result["metadata"]
print(f"\n  -> Full rows     : {meta['n_full_rows']}")
print(f"  -> Modelling rows: {meta['n_modelling_rows']}")
print(f"  -> Accuracy      : {meta['walk_forward_accuracy']:.4f} +/- {meta['walk_forward_std']:.4f}")
print(f"  -> Baseline      : {meta['baseline']:.4f}")
print(f"  -> Edge          : {meta['edge']:+.4f}")
print(f"  -> Permutation   : {meta['permutation']['verdict']}")

# ── Step 4: Update all_stocks_summary.csv ────────────────────────────────
print("\n" + "="*60)
print("  STEP 4 — Updating all_stocks_summary.csv")
print("="*60)

summary_path = config.REPORTS_DIR / "all_stocks_summary.csv"
if summary_path.exists():
    df_sum = pd.read_csv(summary_path)
    df_sum = df_sum[df_sum["stock"] != "RELIANCE"]
else:
    df_sum = pd.DataFrame()

new_row = {
    "stock":       "RELIANCE",
    "horizon":     meta["config"]["horizon"],
    "model":       meta["config"]["model_name"],
    "rows":        meta["n_modelling_rows"],
    "accuracy":    meta["walk_forward_accuracy"],
    "std":         meta["walk_forward_std"],
    "baseline":    meta["baseline"],
    "edge":        meta["edge"],
    "permutation": meta["permutation"]["verdict"],
}

df_sum = pd.concat([df_sum, pd.DataFrame([new_row])], ignore_index=True)
df_sum = df_sum.sort_values("stock").reset_index(drop=True)
df_sum.to_csv(summary_path, index=False)
print(f"  -> Updated {summary_path.name}")
print(df_sum[df_sum["stock"] == "RELIANCE"].to_string(index=False))

print("\n" + "="*60)
print("  DONE — RELIANCE history extended and model rebuilt.")
print("="*60 + "\n")

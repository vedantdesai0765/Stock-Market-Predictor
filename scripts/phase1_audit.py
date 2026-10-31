# -*- coding: utf-8 -*-
"""
Phase 1 Audit Script
====================
Checks:
  1. Raw CSV stats per stock: date range, row count, duplicates, missing values
  2. Processed dataset stats: date range, row count, column presence, missing values
  3. Model artifact presence per stock
  4. Pipeline row reconciliation: raw -> processed -> modelling rows
  5. Config vs actual file agreement
"""

import sys
import io
import os

# Patch stdout for UTF-8 output (fixes encoding issues on Windows terminals)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add src/ to path so `config` and other src modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from config import STOCKS, RAW_DIR, PROCESSED_DIR, MODELS_DIR  # type: ignore[import]

# colour helpers
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}OK{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}WARN{RESET} {msg}")
def fail(msg): print(f"  {RED}FAIL{RESET} {msg}")
def info(msg): print(f"  --> {msg}")

issues = []

def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def detect_date_col(df):
    for c in df.columns:
        if "date" in c.lower():
            return c
    return None

def audit_raw(stock_key, raw_path):
    result = {"key": stock_key}
    if not raw_path.exists():
        fail(f"Raw file MISSING: {raw_path.name}")
        issues.append(f"{stock_key}: raw CSV missing")
        result["exists"] = False
        return result

    result["exists"] = True
    df = load_csv_safe(raw_path)
    if df is None:
        fail("Could not parse CSV")
        issues.append(f"{stock_key}: raw CSV unreadable")
        return result

    result["raw_rows"] = len(df)
    result["raw_cols"] = list(df.columns)

    date_col = detect_date_col(df)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        result["earliest"] = str(df[date_col].min().date())
        result["latest"]   = str(df[date_col].max().date())
        info(f"Date range : {result['earliest']} -> {result['latest']}")
        info(f"Rows       : {result['raw_rows']}")
    else:
        warn("No date column found")
        issues.append(f"{stock_key}: no date column in raw CSV")

    if date_col and date_col in df.columns:
        dups = df[date_col].duplicated().sum()
        result["duplicates"] = int(dups)
        if dups > 0:
            warn(f"Duplicate dates: {dups}")
            issues.append(f"{stock_key}: {dups} duplicate dates in raw CSV")
        else:
            ok("No duplicate dates")

    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    result["missing_total"] = total_missing
    if total_missing > 0:
        cols_with_missing = missing[missing > 0].to_dict()
        warn(f"Missing values: {cols_with_missing}")
        issues.append(f"{stock_key}: missing values in raw CSV: {cols_with_missing}")
    else:
        ok("No missing values")

    expected = {"Open", "High", "Low", "Close", "Volume"}
    missing_cols = expected - set(df.columns)
    if missing_cols:
        warn(f"Missing OHLCV columns: {missing_cols}")
        issues.append(f"{stock_key}: missing OHLCV cols {missing_cols}")
    else:
        ok("All OHLCV columns present")

    return result


def audit_processed(stock_key, processed_path):
    result = {}
    if not processed_path.exists():
        fail(f"Processed file MISSING: {processed_path.name}")
        issues.append(f"{stock_key}: processed dataset missing")
        result["exists"] = False
        return result

    result["exists"] = True
    df = load_csv_safe(processed_path)
    if df is None:
        fail("Could not parse processed CSV")
        return result

    result["processed_rows"] = len(df)
    result["processed_cols"] = len(df.columns)

    date_col = detect_date_col(df)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        result["proc_earliest"] = str(df[date_col].min().date())
        result["proc_latest"]   = str(df[date_col].max().date())
        info(f"Processed rows : {result['processed_rows']}  cols={result['processed_cols']}")
        info(f"Date range     : {result['proc_earliest']} -> {result['proc_latest']}")

    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    result["proc_missing"] = total_missing
    if total_missing > 0:
        cols_with_missing = missing[missing > 0].to_dict()
        warn(f"Missing in processed: {cols_with_missing}")
    else:
        ok("No missing values in processed dataset")

    if "Target" in df.columns:
        ok("Target column present")
        target_counts = df["Target"].value_counts().to_dict()
        info(f"Target distribution: {target_counts}")
        result["target_counts"] = target_counts
    else:
        warn("No Target column found")
        issues.append(f"{stock_key}: no Target column in processed dataset")

    return result


def audit_models(stock_key, model_dir):
    result = {}
    if not model_dir.exists():
        fail(f"Model directory MISSING: {model_dir.name}")
        issues.append(f"{stock_key}: model directory missing")
        result["model_exists"] = False
        return result

    files = list(model_dir.iterdir())
    result["model_files"] = [f.name for f in files]
    info(f"Model files: {result['model_files']}")

    model_path = model_dir / "model.joblib"
    if model_path.exists():
        ok("model.joblib found")
        result["model_exists"] = True
        try:
            model = joblib.load(model_path)
            result["model_type"] = type(model).__name__
            ok(f"Model type: {result['model_type']}")
        except Exception as e:
            warn(f"Could not load model: {e}")
    else:
        warn("model.joblib not found")
        issues.append(f"{stock_key}: model.joblib missing")
        result["model_exists"] = False

    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        ok("metadata.json found")
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            result["metadata"] = meta
            for k in ("accuracy", "baseline", "edge", "permutation"):
                if k in meta:
                    info(f"  {k}: {meta[k]}")
        except Exception as e:
            warn(f"Could not read metadata: {e}")
    else:
        warn("metadata.json not found")
        issues.append(f"{stock_key}: metadata.json missing")

    return result


# ── main audit ──────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  PHASE 1 AUDIT -- Stock Market Predictor")
print("="*70)

summary_rows = []

for key, cfg in STOCKS.items():
    print(f"\n" + "-"*60)
    print(f"  {key}  ({cfg.ticker})  |  config start={cfg.start} end={cfg.end}")
    print("-"*60)

    row = {"stock": key, "ticker": cfg.ticker,
           "config_start": cfg.start, "config_end": cfg.end}

    print("\n  [RAW]")
    raw_r = audit_raw(key, cfg.raw_path)
    row.update({
        "raw_exists":   raw_r.get("exists", False),
        "raw_rows":     raw_r.get("raw_rows", "N/A"),
        "raw_earliest": raw_r.get("earliest", "N/A"),
        "raw_latest":   raw_r.get("latest",   "N/A"),
        "raw_dups":     raw_r.get("duplicates", "N/A"),
        "raw_missing":  raw_r.get("missing_total", "N/A"),
    })

    print("\n  [PROCESSED]")
    proc_r = audit_processed(key, cfg.features_path)
    row.update({
        "proc_exists":   proc_r.get("exists", False),
        "proc_rows":     proc_r.get("processed_rows", "N/A"),
        "proc_earliest": proc_r.get("proc_earliest", "N/A"),
        "proc_latest":   proc_r.get("proc_latest",   "N/A"),
        "proc_missing":  proc_r.get("proc_missing", "N/A"),
    })

    print("\n  [MODEL ARTIFACTS]")
    model_r = audit_models(key, cfg.model_dir)
    row.update({
        "model_exists": model_r.get("model_exists", False),
        "model_type":   model_r.get("model_type", "N/A"),
    })
    if "metadata" in model_r:
        meta = model_r["metadata"]
        row.update({
            "reported_acc":  meta.get("accuracy", "N/A"),
            "reported_edge": meta.get("edge", "N/A"),
            "permutation":   meta.get("permutation", "N/A"),
        })

    if raw_r.get("raw_rows") and proc_r.get("processed_rows"):
        drop_pct = 100 * (1 - proc_r["processed_rows"] / raw_r["raw_rows"])
        row["rows_dropped_pct"] = f"{drop_pct:.1f}%"
        info(f"\n  Pipeline: raw={raw_r['raw_rows']} -> processed={proc_r['processed_rows']} (dropped {drop_pct:.1f}%)")

    summary_rows.append(row)


# ── summary table ───────────────────────────────────────────────────────────

print("\n\n" + "="*70)
print("  SUMMARY TABLE")
print("="*70)

summary_df = pd.DataFrame(summary_rows)
display_cols = [
    "stock", "raw_rows", "raw_earliest", "raw_latest",
    "raw_dups", "raw_missing", "proc_rows",
    "model_exists", "reported_acc", "reported_edge", "permutation"
]
display_cols = [c for c in display_cols if c in summary_df.columns]
print(summary_df[display_cols].to_string(index=False))


# ── config vs actual ────────────────────────────────────────────────────────

print("\n\n" + "="*70)
print("  CONFIG vs ACTUAL DATA AGREEMENT")
print("="*70)
for row in summary_rows:
    key = row["stock"]
    cfg_start = row.get("config_start", "?")
    actual_earliest = row.get("raw_earliest", "?")
    if actual_earliest not in ("N/A", "?"):
        diff_days = (pd.to_datetime(actual_earliest) - pd.to_datetime(cfg_start)).days
        if abs(diff_days) > 30:
            warn(f"{key}: config start={cfg_start} but actual earliest={actual_earliest} (diff={diff_days:+d} days)")
            if diff_days > 0:
                issues.append(f"{key}: actual data starts {diff_days} days LATER than config start={cfg_start}")
        else:
            ok(f"{key}: config start={cfg_start} ~= actual earliest={actual_earliest}")
    else:
        warn(f"{key}: cannot compare (raw file missing)")


# ── issues summary ──────────────────────────────────────────────────────────

print("\n\n" + "="*70)
print(f"  ISSUES FOUND ({len(issues)} total)")
print("="*70)
if issues:
    for i, issue in enumerate(issues, 1):
        print(f"  {i:2d}. {issue}")
else:
    print("  No issues found!")

print("\n" + "="*70 + "\n")

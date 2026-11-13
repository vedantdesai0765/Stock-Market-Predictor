"""Reading and writing the files in live/ (the Morning Call's small public record)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .live import HISTORY_COLUMNS

FORECAST_FILE = "latest_forecast.json"
HISTORY_FILE = "forecast_history.csv"
SCORECARD_FILE = "scorecard.json"
HOLIDAY_FILE = "nse_holidays.json"


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return default


def write_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_history(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    h = pd.read_csv(path, dtype={"target_date": str, "based_on": str, "ticker": str})
    for c in HISTORY_COLUMNS:
        if c not in h.columns:
            h[c] = pd.NA
    return h[HISTORY_COLUMNS]


def write_history(h: pd.DataFrame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h[HISTORY_COLUMNS].to_csv(path, index=False, lineterminator="\n")

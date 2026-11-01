"""All file locations in one place. Override the root with SENT_STUDY_ROOT if needed."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SENT_STUDY_ROOT", Path(__file__).resolve().parents[1]))
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
OOF = REPORTS / "oof"
FINBERT = RAW / "finbert"
for _d in (RAW, INTERIM, PROCESSED, REPORTS, OOF):
    _d.mkdir(parents=True, exist_ok=True)


def first(*cands):
    """first existing path among candidates (lets the shipped processed files stand in for interim ones)"""
    for c in cands:
        if Path(c).exists():
            return str(c)
    raise FileNotFoundError(" | ".join(map(str, cands)))


def price_file(name="yahoo_daily_prices_long"):
    return first(RAW / f"{name}.csv", RAW / f"{name}.parquet")

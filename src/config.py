# src/config.py
"""
Central configuration for the Stock Market Predictor.

This is the ONLY place where stock-specific information lives.
To add a new stock to the entire project, add one entry to STOCKS below.
Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# config.py lives in <root>/src/, so parents[1] is always the project root,
# no matter whether code is run from notebooks/, src/, or the root itself.

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

ALL_DIRS = [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]


def ensure_dirs() -> None:
    """Create every directory the project expects. Safe to call repeatedly."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    # Keep empty dirs alive in git
    for d in (MODELS_DIR, FIGURES_DIR):
        keep = d / ".gitkeep"
        if not keep.exists():
            keep.touch()


# ---------------------------------------------------------------------------
# Stock registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StockConfig:
    """Everything the pipeline needs to know about one stock."""

    key: str                      # short internal id, e.g. "TCS"
    ticker: str                   # Yahoo Finance symbol, e.g. "TCS.NS"
    display_name: str             # shown in the dashboard
    raw_filename: str             # file inside data/raw/
    start: str = "2014-01-01"     # download window (used later, Phase 5)
    end: str = "2024-01-01"
    currency: str = "\u20b9"      # rupee symbol
    sentiment_filename: Optional[str] = None   # file inside data/processed/
    notes: str = ""

    # -- derived paths -----------------------------------------------------

    @property
    def raw_path(self) -> Path:
        return RAW_DIR / self.raw_filename

    @property
    def features_path(self) -> Path:
        return PROCESSED_DIR / f"{self.key}_dataset.csv"

    @property
    def sentiment_path(self) -> Optional[Path]:
        if self.sentiment_filename is None:
            return None
        return PROCESSED_DIR / self.sentiment_filename

    @property
    def model_dir(self) -> Path:
        return MODELS_DIR / self.key

    @property
    def has_sentiment(self) -> bool:
        p = self.sentiment_path
        return p is not None and p.exists()


STOCKS: dict[str, StockConfig] = {
    "TCS": StockConfig(
        key="TCS",
        ticker="TCS.NS",
        display_name="Tata Consultancy Services",
        raw_filename="TCS_raw.csv",
        start="2014-01-01",
        end="2024-01-01",
        sentiment_filename="TCS_news_sentiment.csv",
        notes=(
            "Sentiment file is currently derived from the Combined_News_DJIA "
            "corpus (world news, 2008-2016), NOT from TCS-specific news. "
            "Treated as a proof-of-concept for the sentiment machinery only."
        ),
    ),
    "RELIANCE": StockConfig(
        key="RELIANCE",
        ticker="RELIANCE.NS",
        display_name="Reliance Industries",
        raw_filename="Reliance_raw.csv",
        start="2022-01-01",
        end="2024-01-01",
        sentiment_filename=None,
        notes="No news data collected yet.",
    ),
    "INFY": StockConfig(
        key="INFY",
        ticker="INFY.NS",
        display_name="Infosys",
        raw_filename="INFY_raw.csv",
        start="2014-01-01",
        end="2024-01-01",
        sentiment_filename=None,
        notes="Added 2026-08-30.",
    ),
}


# ---------------------------------------------------------------------------
# Modelling defaults (shared so every script/notebook agrees)
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42

# Indicator windows
SMA_SHORT: int = 20
SMA_LONG: int = 50
EMA_SPAN: int = 20
RSI_WINDOW: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

# Prediction horizon in trading days (Phase 3 will experiment with 3 and 5)
TARGET_HORIZON: int = 1

# The feature set used by the current baseline models.
# Phase 2 will replace these with stationary equivalents.
BASELINE_FEATURES: list[str] = [
    "SMA_20",
    "SMA_50",
    "EMA_20",
    "RSI",
    "MACD",
    "MACD_signal",
    "Daily_Return",
]

SENTIMENT_FEATURES: list[str] = [
    "vader_sentiment",
    "finbert_sentiment",
]

# Columns that must NEVER be used as model inputs.
FORBIDDEN_FEATURES: set[str] = {
    "Open", "High", "Low", "Close", "Volume",
    "Target", "Date", "Next_Close",
}


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def get_stock(key: str) -> StockConfig:
    """Look up a stock by key, case-insensitively, with a helpful error."""
    k = key.strip().upper()
    if k not in STOCKS:
        available = ", ".join(sorted(STOCKS))
        raise KeyError(f"Unknown stock '{key}'. Available: {available}")
    return STOCKS[k]


def list_stocks() -> list[str]:
    return sorted(STOCKS)


if __name__ == "__main__":
    ensure_dirs()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Registered   : {list_stocks()}")
    for s in STOCKS.values():
        print(f"  {s.key:10s} {s.ticker:14s} raw exists={s.raw_path.exists()}")

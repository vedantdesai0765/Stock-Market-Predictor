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

# Market-wide news sentiment shared by every stock (no free 20-year company-level
# Indian news archive exists). Produced by sentiment_study/src/export_to_main.py.
# The main pipeline trains with with_sentiment=False; this file is used only when a
# dataset is built with with_sentiment=True (and trims it to 2001-2022).
MARKET_SENTIMENT_FILE = "IN_market_news_sentiment.csv"   # old: one market-wide score shared by all stocks


def company_sentiment_file(key: str) -> str:
    """Per-stock, company-targeted sentiment (entity-level FinBERT; see sentiment_study/src/export_company_sentiment.py)."""
    return f"{key}_company_sentiment.csv"


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
        start="2004-08-25",
        end="2026-09-01",
        sentiment_filename=company_sentiment_file("TCS"),
        notes=(
            "Sentiment: company-targeted (entity-level FinBERT) scores of Times of India "
            "headlines naming TCS (2001-2022) and TCS's NSE filings (2008-2025), built by "
            "sentiment_study/src/export_company_sentiment.py. Earlier sources kept for reference: "
            "IN_market_news_sentiment.csv (market-wide) and TCS_news_sentiment.csv (Reddit proxy)."
        ),
    ),
    "RELIANCE": StockConfig(
        key="RELIANCE",
        ticker="RELIANCE.NS",
        display_name="Reliance Industries",
        raw_filename="Reliance_raw.csv",
        start="2000-01-01",
        end="2026-09-01",
        sentiment_filename=company_sentiment_file("RELIANCE"),
        notes="History extended from 2022-01-01 to 2014-01-01 on 2026-10-31. Uses its own company-targeted sentiment file.",
    ),
    "INFY": StockConfig(
        key="INFY",
        ticker="INFY.NS",
        display_name="Infosys",
        raw_filename="INFY_raw.csv",
        start="2000-01-01",
        end="2026-09-01",
        sentiment_filename=company_sentiment_file("INFY"),
        notes="Added 2026-08-30.",
    ),
    "HDFCBANK": StockConfig(
        key="HDFCBANK",
        ticker="HDFCBANK.NS",
        display_name="HDFC Bank",
        raw_filename="HDFCBANK_raw.csv",
        start="2000-01-01",
        end="2026-09-01",
        sentiment_filename=company_sentiment_file("HDFCBANK"),
        notes="Added 2026-08-30.",
    ),
    "SBI": StockConfig(
        key="SBI",
        ticker="SBIN.NS",
        display_name="State Bank of India",
        raw_filename="SBI_raw.csv",
        start="2000-01-01",
        end="2026-09-01",
        sentiment_filename=company_sentiment_file("SBI"),
        notes="Added 2026-08-30.",
    ),
    "ITC": StockConfig(
        key="ITC",
        ticker="ITC.NS",
        display_name="ITC",
        raw_filename="ITC_raw.csv",
        start="2000-01-01",
        end="2026-09-01",
        sentiment_filename=company_sentiment_file("ITC"),
        notes="Added 2026-08-30.",
    ),
    "ADANIENT": StockConfig(
        key="ADANIENT",
        ticker="ADANIENT.NS",
        display_name="Adani Enterprises",
        raw_filename="ADANIENT_raw.csv",
        start="2004-07-01",
        end="2026-09-01",
        sentiment_filename=company_sentiment_file("ADANIENT"),
        notes="Added 2026-08-30. High-volatility stock.",
    ),
    "VEDL": StockConfig(
        key="VEDL",
        ticker="VEDL.NS",
        display_name="Vedanta",
        raw_filename="VEDL_raw.csv",
        start="2000-01-01",
        end="2026-09-01",
        sentiment_filename=company_sentiment_file("VEDL"),
        notes="Added 2026-08-30. High-volatility stock.",
    ),
    "ETERNAL": StockConfig(
        key="ETERNAL",
        ticker="ETERNAL.NS",
        display_name="Eternal Limited",
        raw_filename="ETERNAL_raw.csv",
        start="2021-07-23",
        end="2026-09-01",
        sentiment_filename=company_sentiment_file("ETERNAL"),
        notes="Eternal Limited (formerly Zomato Limited). Uses its own company-targeted sentiment file.",
    ),
    "ADANIGREEN": StockConfig(
        key="ADANIGREEN",
        ticker="ADANIGREEN.NS",
        display_name="Adani Green Energy",
        raw_filename="ADANIGREEN_raw.csv",
        start="2018-06-18",
        end="2026-09-01",
        sentiment_filename=company_sentiment_file("ADANIGREEN"),
        notes="Added 2026-08-30. High-volatility stock.",
    ),

    "ICICIBANK": StockConfig(
        key="ICICIBANK", ticker="ICICIBANK.NS",
        display_name="ICICI Bank", raw_filename="ICICIBANK_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "BHARTIARTL": StockConfig(
        key="BHARTIARTL", ticker="BHARTIARTL.NS",
        display_name="Bharti Airtel", raw_filename="BHARTIARTL_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "LT": StockConfig(
        key="LT", ticker="LT.NS",
        display_name="Larsen & Toubro", raw_filename="LT_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "AXISBANK": StockConfig(
        key="AXISBANK", ticker="AXISBANK.NS",
        display_name="Axis Bank", raw_filename="AXISBANK_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "KOTAKBANK": StockConfig(
        key="KOTAKBANK", ticker="KOTAKBANK.NS",
        display_name="Kotak Mahindra Bank", raw_filename="KOTAKBANK_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "M_M": StockConfig(
        key="M_M", ticker="M&M.NS",
        display_name="Mahindra & Mahindra", raw_filename="M_M_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "BAJFINANCE": StockConfig(
        key="BAJFINANCE", ticker="BAJFINANCE.NS",
        display_name="Bajaj Finance", raw_filename="BAJFINANCE_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "HINDUNILVR": StockConfig(
        key="HINDUNILVR", ticker="HINDUNILVR.NS",
        display_name="Hindustan Unilever", raw_filename="HINDUNILVR_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "MARUTI": StockConfig(
        key="MARUTI", ticker="MARUTI.NS",
        display_name="Maruti Suzuki", raw_filename="MARUTI_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "SUNPHARMA": StockConfig(
        key="SUNPHARMA", ticker="SUNPHARMA.NS",
        display_name="Sun Pharmaceutical", raw_filename="SUNPHARMA_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "TITAN": StockConfig(
        key="TITAN", ticker="TITAN.NS",
        display_name="Titan Company", raw_filename="TITAN_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "ULTRACEMCO": StockConfig(
        key="ULTRACEMCO", ticker="ULTRACEMCO.NS",
        display_name="UltraTech Cement", raw_filename="ULTRACEMCO_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "NTPC": StockConfig(
        key="NTPC", ticker="NTPC.NS",
        display_name="NTPC", raw_filename="NTPC_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "POWERGRID": StockConfig(
        key="POWERGRID", ticker="POWERGRID.NS",
        display_name="Power Grid Corporation of India",
        raw_filename="POWERGRID_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "ONGC": StockConfig(
        key="ONGC", ticker="ONGC.NS",
        display_name="Oil & Natural Gas Corporation", raw_filename="ONGC_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "BAJAJFINSV": StockConfig(
        key="BAJAJFINSV", ticker="BAJAJFINSV.NS",
        display_name="Bajaj Finserv", raw_filename="BAJAJFINSV_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "BEL": StockConfig(
        key="BEL", ticker="BEL.NS",
        display_name="Bharat Electronics", raw_filename="BEL_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "ASIANPAINT": StockConfig(
        key="ASIANPAINT", ticker="ASIANPAINT.NS",
        display_name="Asian Paints", raw_filename="ASIANPAINT_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "APOLLOHOSP": StockConfig(
        key="APOLLOHOSP", ticker="APOLLOHOSP.NS",
        display_name="Apollo Hospitals Enterprise",
        raw_filename="APOLLOHOSP_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
    ),
    "CIPLA": StockConfig(
        key="CIPLA", ticker="CIPLA.NS",
        display_name="Cipla", raw_filename="CIPLA_raw.csv",
        start="2014-01-01", end="2026-11-16",
        sentiment_filename=None, notes="Expanded stock universe.",
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

# Prediction horizon in trading days (default Phase 3/4 standard is 3 days)
TARGET_HORIZON: int = 3

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

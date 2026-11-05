"""
backend/app/services/status_service.py
Service adapter for data lineage, timestamp separation, and dataset freshness.
Distinguishes raw market data dates from training timestamps and out-of-sample evaluation dates.
"""

import json
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

from backend.app.core.config import settings
from backend.app.core.errors import StockNotFoundError
from backend.app.schemas.status import DataStatusResponse

# Ensure src/ is importable
if str(settings.SRC_DIR) not in sys.path:
    sys.path.insert(0, str(settings.SRC_DIR))
if str(settings.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(settings.ROOT_DIR))

import config


class StatusService:
    @staticmethod
    def get_data_status(ticker: str) -> DataStatusResponse:
        key = ticker.strip().upper()
        if key not in config.STOCKS:
            raise StockNotFoundError(ticker=ticker)

        stock = config.get_stock(key)

        # 1. Raw Market Data
        raw_exists = stock.raw_path.exists()
        raw_last_date: Optional[str] = None
        raw_rows: Optional[int] = None

        if raw_exists:
            try:
                # Read only Date column for speed
                raw_df = pd.read_csv(stock.raw_path, usecols=["Date"])
                raw_rows = len(raw_df)
                if raw_rows > 0:
                    raw_last_date = str(raw_df["Date"].iloc[-1])[:10]
            except Exception:
                pass

        # 2. Model Artifacts & Out-of-Sample Date
        model_joblib = stock.model_dir / "model.joblib"
        metadata_json = stock.model_dir / "metadata.json"
        oof_csv = stock.model_dir / "oof_predictions.csv"

        model_exists = model_joblib.exists() and metadata_json.exists()
        model_trained_at: Optional[str] = None
        oof_last_date: Optional[str] = None

        if metadata_json.exists():
            try:
                with open(metadata_json) as fh:
                    meta = json.load(fh)
                model_trained_at = meta.get("trained_at")
            except Exception:
                pass

        if oof_csv.exists():
            try:
                oof_df = pd.read_csv(oof_csv, usecols=["Date"])
                if not oof_df.empty:
                    oof_last_date = str(oof_df["Date"].iloc[-1])[:10]
            except Exception:
                pass

        # 3. News & Sentiment Data
        news_file = settings.NEWS_DIR / f"{key}_news_items.csv"
        news_exists = news_file.exists()
        news_last_date: Optional[str] = None
        news_count: Optional[int] = None

        if news_exists:
            try:
                ndf = pd.read_csv(news_file, usecols=["trade_date"])
                news_count = len(ndf)
                if news_count > 0:
                    news_last_date = str(ndf["trade_date"].iloc[-1])[:10]
            except Exception:
                pass

        # Build comprehensive summary
        summary_parts = [
            f"Stock: {stock.display_name} ({stock.ticker}).",
            f"Raw OHLCV data: {'Available up to ' + raw_last_date + ' (' + str(raw_rows) + ' records)' if raw_exists else 'Not found'}.",
            f"Model artifacts: {'Trained on ' + str(model_trained_at) if model_exists else 'Not yet trained'}.",
            f"Out-of-sample evaluation boundary: {oof_last_date if oof_last_date else 'N/A'}.",
            f"News sentiment feed: {'Available up to ' + str(news_last_date) + ' (' + str(news_count) + ' articles)' if news_exists else 'Not found'}.",
            "Mode: Historical research snapshot (live broker feed disabled).",
        ]
        status_summary = " ".join(summary_parts)

        return DataStatusResponse(
            ticker=key,
            display_name=stock.display_name,
            raw_data_file=stock.raw_filename,
            raw_data_exists=raw_exists,
            raw_data_last_date=raw_last_date,
            raw_data_rows=raw_rows,
            model_exists=model_exists,
            model_trained_at=model_trained_at,
            oof_eval_last_date=oof_last_date,
            news_file_exists=news_exists,
            news_last_date=news_last_date,
            news_count=news_count,
            status_summary=status_summary,
            is_live_market=False,
        )

"""
backend/app/services/stock_service.py
Service adapter for stock registry and stock metadata.
"""

import sys
from pathlib import Path
from typing import List

from backend.app.core.config import settings
from backend.app.core.errors import StockNotFoundError
from backend.app.schemas.stock import StockItem, StockListResponse

# Ensure src/ is importable
if str(settings.SRC_DIR) not in sys.path:
    sys.path.insert(0, str(settings.SRC_DIR))
if str(settings.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(settings.ROOT_DIR))

import config


class StockService:
    @staticmethod
    def get_all_stocks() -> StockListResponse:
        items: List[StockItem] = []
        for key in config.list_stocks():
            stock = config.get_stock(key)
            has_raw = stock.raw_path.exists()
            has_model = (stock.model_dir / "model.joblib").exists() and (stock.model_dir / "metadata.json").exists()
            has_sentiment = stock.has_sentiment

            items.append(
                StockItem(
                    ticker=stock.key,
                    symbol=stock.ticker,
                    display_name=stock.display_name,
                    currency=stock.currency,
                    start_date=stock.start,
                    end_date=stock.end,
                    has_model=has_model,
                    has_raw_data=has_raw,
                    has_sentiment=has_sentiment,
                    notes=stock.notes,
                )
            )
        return StockListResponse(stocks=items, total=len(items))

    @staticmethod
    def get_stock(ticker: str) -> StockItem:
        key = ticker.strip().upper()
        if key not in config.STOCKS:
            raise StockNotFoundError(ticker=ticker)

        stock = config.get_stock(key)
        has_raw = stock.raw_path.exists()
        has_model = (stock.model_dir / "model.joblib").exists() and (stock.model_dir / "metadata.json").exists()
        has_sentiment = stock.has_sentiment

        return StockItem(
            ticker=stock.key,
            symbol=stock.ticker,
            display_name=stock.display_name,
            currency=stock.currency,
            start_date=stock.start,
            end_date=stock.end,
            has_model=has_model,
            has_raw_data=has_raw,
            has_sentiment=has_sentiment,
            notes=stock.notes,
        )

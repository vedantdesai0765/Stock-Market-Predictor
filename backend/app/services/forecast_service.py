"""
backend/app/services/forecast_service.py
Service adapter for Next-Morning Forecast and overnight gap analytics.
"""

import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd

from backend.app.core.config import settings
from backend.app.core.errors import DataFileNotFoundError, StockNotFoundError
from backend.app.schemas.forecast import (
    AnnualForecastItem,
    ForecastResponse,
    ForecastSummaryItem,
)

# Ensure src/ is importable
if str(settings.SRC_DIR) not in sys.path:
    sys.path.insert(0, str(settings.SRC_DIR))
if str(settings.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(settings.ROOT_DIR))

import config


class ForecastService:
    @staticmethod
    def get_forecast(ticker: str) -> ForecastResponse:
        key = ticker.strip().upper()
        if key not in config.STOCKS:
            raise StockNotFoundError(ticker=ticker)

        summary_csv = settings.FORECAST_DIR / "stock_summary.csv"
        if not summary_csv.exists():
            raise DataFileNotFoundError(ticker=ticker, file_type="reports/forecast/stock_summary.csv")

        df = pd.read_csv(summary_csv)
        stock_df = df[df["ticker"].astype(str).str.upper() == key]

        if stock_df.empty:
            raise DataFileNotFoundError(ticker=ticker, file_type=f"forecast summary rows for {key}")

        summaries: List[ForecastSummaryItem] = []
        best_model: Optional[str] = None
        best_acc: float = -1.0
        best_edge: float = -999.0

        for _, row in stock_df.iterrows():
            item = ForecastSummaryItem(
                period=str(row["period"]),
                target=str(row["target"]),
                model=str(row["model"]),
                ticker=str(row["ticker"]),
                n=int(row["n"]),
                first_year=int(row["first_year"]),
                last_year=int(row["last_year"]),
                acc_guess=round(float(row["acc_guess"]), 4),
                acc_model=round(float(row["acc_model"]), 4),
                acc_news=round(float(row["acc_news"]), 4),
                edge=round(float(row["edge"]), 4),
                p_vs_guess=round(float(row["p_vs_guess"]), 5),
                news_effect=round(float(row["news_effect"]), 4),
                p_news=round(float(row["p_news"]), 5),
            )
            summaries.append(item)
            if item.acc_model > best_acc:
                best_acc = item.acc_model
                best_model = item.model
                best_edge = item.edge

        # Load annual breakdown if available
        annual_items: List[AnnualForecastItem] = []
        year_csv = settings.FORECAST_DIR / "stock_by_year.csv"
        if year_csv.exists():
            ydf = pd.read_csv(year_csv)
            ystock = ydf[ydf["ticker"].astype(str).str.upper() == key]
            for _, yrow in ystock.iterrows():
                acc_m = float(yrow["acc_model"])
                acc_g = float(yrow["acc_guess"])
                annual_items.append(
                    AnnualForecastItem(
                        year=int(yrow["year"]),
                        model=str(yrow["model"]),
                        acc_guess=round(acc_g, 4),
                        acc_model=round(acc_m, 4),
                        acc_news=round(float(yrow["acc_news"]), 4),
                        edge=round(acc_m - acc_g, 4),
                        sample_count=int(yrow["n"]),
                    )
                )

        stock_cfg = config.get_stock(key)

        return ForecastResponse(
            ticker=key,
            display_name=stock_cfg.display_name,
            summaries=summaries,
            best_model=best_model,
            best_accuracy=best_acc if best_acc >= 0 else None,
            best_edge=best_edge if best_edge > -900 else None,
            annual_breakdown=annual_items if annual_items else None,
        )

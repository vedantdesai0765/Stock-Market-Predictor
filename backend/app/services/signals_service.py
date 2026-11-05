"""
backend/app/services/signals_service.py
Service adapter for Actionable Trading Signal Engine.
"""

import sys
from typing import Optional

from backend.app.core.config import settings
from backend.app.core.errors import StockNotFoundError, StockMarketPredictorException
from backend.app.schemas.signals import LatestSignalResponse

# Ensure src/ is importable
if str(settings.SRC_DIR) not in sys.path:
    sys.path.insert(0, str(settings.SRC_DIR))
if str(settings.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(settings.ROOT_DIR))

import config
import signals


class SignalsService:
    @staticmethod
    def get_latest_signal(
        ticker: str,
        variant: str = "A",
        friction_bps: float = 15.0,
    ) -> LatestSignalResponse:
        key = ticker.strip().upper()
        if key not in config.STOCKS:
            raise StockNotFoundError(ticker=ticker)

        var = variant.strip().upper()
        if var not in {"A", "B", "C", "D", "E"}:
            raise StockMarketPredictorException(
                message=f"Invalid strategy variant '{variant}'. Allowed: A, B, C, D, E.",
                status_code=400,
                error_code="INVALID_VARIANT",
            )

        friction_rate = friction_bps / 10000.0

        try:
            status = signals.get_latest_signal_status(
                stock_key=key,
                variant=var,
                friction_rate=friction_rate,
                base_dir=settings.ROOT_DIR,
            )
        except Exception as exc:
            raise StockMarketPredictorException(
                message=f"Signal calculation failed for {key}: {str(exc)}",
                status_code=500,
                error_code="SIGNAL_GENERATION_FAILED",
            )

        return LatestSignalResponse(
            ticker=key,
            date=str(status["date"]),
            signal=int(status["signal"]),
            signal_label=str(status["signal_label"]),
            calibrated_proba_up=round(float(status["calibrated_proba_up"]), 4),
            confidence=round(float(status["confidence"]), 4),
            expected_gross_return=round(float(status["expected_gross_return"]), 4),
            friction_rate=round(float(status["friction_rate"]), 5),
            net_expectancy=round(float(status["net_expectancy"]), 4),
            rationale=str(status["rationale"]),
            close=round(float(status["close"]), 2),
            rsi=round(float(status["rsi"]), 2),
            macd_hist=round(float(status["macd_hist"]), 4),
            sma20=round(float(status["sma20"]), 2),
            sma50=round(float(status["sma50"]), 2),
            sentiment_score=round(float(status["sentiment_score"]), 4),
            market_regime=str(status["market_regime"]),
            flag_model=bool(status["flag_model"]),
            flag_tech=bool(status["flag_tech"]),
            flag_sentiment=bool(status["flag_sentiment"]),
        )

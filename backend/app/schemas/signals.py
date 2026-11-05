"""
backend/app/schemas/signals.py
Pydantic schemas for Actionable Trading Signals and multi-factor breakdown.
"""

from typing import Optional
from pydantic import BaseModel, Field


class LatestSignalResponse(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol key, e.g. TCS")
    date: str = Field(..., description="Signal computation date (YYYY-MM-DD)")
    signal: int = Field(..., description="Numeric signal: 1 (BUY), -1 (SELL), 0 (HOLD)")
    signal_label: str = Field(..., description="Actionable signal label: 'BUY', 'SELL', or 'HOLD'")
    calibrated_proba_up: float = Field(..., description="Isotonic calibrated upward probability")
    confidence: float = Field(..., description="Signal conviction distance abs(p_up - 0.5)")
    expected_gross_return: float = Field(..., description="Expected 3-day gross return based on calibrated probability")
    friction_rate: float = Field(..., description="Estimated round-trip statutory friction rate (e.g. 0.0015 for 15 bps)")
    net_expectancy: float = Field(..., description="Net expected return after subtracting friction hurdle")
    rationale: str = Field(..., description="Human-readable decision explanation with factor inputs")
    close: float = Field(..., description="Closing price at signal generation")
    rsi: float = Field(..., description="14-day Relative Strength Index")
    macd_hist: float = Field(..., description="Normalized MACD Histogram value")
    sma20: float = Field(..., description="20-day Simple Moving Average")
    sma50: float = Field(..., description="50-day Simple Moving Average")
    sentiment_score: float = Field(..., description="Rolling 5-day entity-targeted FinBERT sentiment score")
    market_regime: str = Field(..., description="Current classified market regime (e.g. TRENDING_BULL, HIGH_VOLATILITY)")
    flag_model: bool = Field(..., description="Whether model confidence hurdle was satisfied")
    flag_tech: bool = Field(..., description="Whether technical trend confirmation was satisfied")
    flag_sentiment: bool = Field(..., description="Whether news sentiment direction agreed with signal")

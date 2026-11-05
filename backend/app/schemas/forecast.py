"""
backend/app/schemas/forecast.py
Pydantic schemas for Next-Morning Forecast and overnight gap analysis.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ForecastSummaryItem(BaseModel):
    period: str = Field(..., description="Evaluation period, e.g. 2013-2025")
    target: str = Field(..., description="Target definition, e.g. Next morning's opening price")
    model: str = Field(..., description="Model name (e.g. LightGBM, LogReg, XGBoost)")
    ticker: str = Field(..., description="Stock ticker or ALL")
    n: int = Field(..., description="Number of historical overnight samples")
    first_year: int
    last_year: int
    acc_guess: float = Field(..., description="Zero-intelligence prior / majority guess accuracy")
    acc_model: float = Field(..., description="Overnight prediction accuracy using technicals/price indicators")
    acc_news: float = Field(..., description="Overnight prediction accuracy with news features added")
    edge: float = Field(..., description="Edge over zero-intelligence guess (acc_model - acc_guess)")
    p_vs_guess: float = Field(..., description="Statistical p-value against random guessing")
    news_effect: float = Field(..., description="Incremental edge contributed by news features")
    p_news: float = Field(..., description="Statistical significance of news contribution")


class AnnualForecastItem(BaseModel):
    year: int
    model: str
    acc_guess: float
    acc_model: float
    acc_news: float
    edge: float
    sample_count: int


class ForecastResponse(BaseModel):
    ticker: str
    display_name: str
    summaries: List[ForecastSummaryItem]
    best_model: Optional[str] = None
    best_accuracy: Optional[float] = None
    best_edge: Optional[float] = None
    annual_breakdown: Optional[List[AnnualForecastItem]] = None

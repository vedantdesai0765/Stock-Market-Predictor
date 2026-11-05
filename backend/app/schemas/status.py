"""
backend/app/schemas/status.py
Pydantic schemas for data lineage, timestamp separation, and dataset freshness.
"""

from typing import Optional
from pydantic import BaseModel, Field


class DataStatusResponse(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol key, e.g. TCS")
    display_name: str = Field(..., description="Stock display name")

    # Raw Market Data
    raw_data_file: str = Field(..., description="Raw OHLCV data filename")
    raw_data_exists: bool = Field(..., description="True if raw CSV exists")
    raw_data_last_date: Optional[str] = Field(None, description="Latest market date in raw OHLCV dataset")
    raw_data_rows: Optional[int] = Field(None, description="Total rows in raw OHLCV dataset")

    # Model Artifacts
    model_exists: bool = Field(..., description="True if trained model.joblib and metadata.json exist")
    model_trained_at: Optional[str] = Field(None, description="Timestamp when model was trained (ISO 8601)")
    oof_eval_last_date: Optional[str] = Field(None, description="Latest date in out-of-sample walk-forward evaluation")

    # News & Sentiment Data
    news_file_exists: bool = Field(..., description="True if company news items CSV exists")
    news_last_date: Optional[str] = Field(None, description="Latest date in news items dataset")
    news_count: Optional[int] = Field(None, description="Total news articles recorded")

    # Context & Freshness
    status_summary: str = Field(..., description="Human-readable explanation of timestamps and data boundaries")
    is_live_market: bool = Field(default=False, description="Whether data stream is connected to live real-time feed")

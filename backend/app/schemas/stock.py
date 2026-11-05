"""
backend/app/schemas/stock.py
Pydantic schemas for stock registry and stock metadata.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class StockItem(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol key, e.g. TCS")
    symbol: str = Field(..., description="Exchange symbol, e.g. TCS.NS")
    display_name: str = Field(..., description="Company name, e.g. Tata Consultancy Services")
    currency: str = Field(default="₹", description="Currency symbol")
    start_date: Optional[str] = Field(None, description="Configured start date")
    end_date: Optional[str] = Field(None, description="Configured end date")
    has_model: bool = Field(..., description="Whether trained model artifacts exist on disk")
    has_raw_data: bool = Field(..., description="Whether raw OHLCV CSV file exists on disk")
    has_sentiment: bool = Field(..., description="Whether company-targeted sentiment file exists")
    notes: Optional[str] = Field(None, description="Stock-specific notes")


class StockListResponse(BaseModel):
    stocks: List[StockItem]
    total: int

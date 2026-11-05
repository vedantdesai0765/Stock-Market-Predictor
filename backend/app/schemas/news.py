"""
backend/app/schemas/news.py
Pydantic schemas for financial news sentiment and explainability.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class NewsArticleItem(BaseModel):
    trade_date: str = Field(..., description="Associated trade date for market reflection (YYYY-MM-DD)")
    published: Optional[str] = Field(None, description="Actual publication date/timestamp")
    source: Optional[str] = Field(None, description="News publisher or filing source")
    entity: Optional[str] = Field(None, description="Targeted company entity")
    text: str = Field(..., description="Article headline or announcement summary")
    targeted_score: Optional[float] = Field(None, description="FinBERT targeted sentiment score (-1.0 to +1.0)")
    plain_score: Optional[float] = Field(None, description="General market sentiment score")
    label: Optional[str] = Field(None, description="Sentiment label (positive, negative, neutral)")
    price_report: Optional[bool] = Field(False, description="True if article is a retrospective price summary")


class LeadLagPoint(BaseModel):
    scope: str
    variant: str
    lag: int
    rho: float
    p: float
    n: int


class SentimentTimelinePoint(BaseModel):
    date: str
    close: Optional[float] = None
    sentiment_score: Optional[float] = None
    article_count: int = 0
    sentiment_label: Optional[str] = None
    top_headline: Optional[str] = None


class NewsResponse(BaseModel):
    ticker: str
    total_articles: int
    average_sentiment: float
    positive_count: int
    negative_count: int
    neutral_count: int
    recent_articles: List[NewsArticleItem]
    lead_lag: Optional[List[LeadLagPoint]] = None
    timeline: Optional[List[SentimentTimelinePoint]] = None

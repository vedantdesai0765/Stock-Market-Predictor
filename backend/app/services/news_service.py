"""
backend/app/services/news_service.py
Service adapter for financial news, entity-targeted sentiment, and lead-lag analysis.
"""

import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd

from backend.app.core.config import settings
from backend.app.core.errors import StockNotFoundError
from backend.app.schemas.news import (
    LeadLagPoint,
    NewsArticleItem,
    NewsResponse,
    SentimentTimelinePoint,
)

# Ensure src/ is importable
if str(settings.SRC_DIR) not in sys.path:
    sys.path.insert(0, str(settings.SRC_DIR))
if str(settings.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(settings.ROOT_DIR))

import config


class NewsService:
    @staticmethod
    def get_news_and_sentiment(ticker: str, limit: int = 50) -> NewsResponse:
        key = ticker.strip().upper()
        if key not in config.STOCKS:
            raise StockNotFoundError(ticker=ticker)

        news_file = settings.NEWS_DIR / f"{key}_news_items.csv"
        if not news_file.exists():
            return NewsResponse(
                ticker=key,
                total_articles=0,
                average_sentiment=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                recent_articles=[],
                lead_lag=None,
                timeline=None,
            )

        df = pd.read_csv(news_file)
        total = len(df)
        if total == 0:
            return NewsResponse(
                ticker=key,
                total_articles=0,
                average_sentiment=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                recent_articles=[],
                lead_lag=None,
                timeline=None,
            )

        score_col = "targeted" if "targeted" in df.columns else "plain"
        avg_sent = float(df[score_col].dropna().mean()) if score_col in df.columns else 0.0

        label_counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
        pos = int(label_counts.get("positive", 0))
        neg = int(label_counts.get("negative", 0))
        neu = int(label_counts.get("neutral", 0))

        # Recent articles (tail sorted by date descending)
        recent_df = df.tail(limit).iloc[::-1]
        articles: List[NewsArticleItem] = []
        for _, row in recent_df.iterrows():
            articles.append(
                NewsArticleItem(
                    trade_date=str(row.get("trade_date", "")),
                    published=str(row.get("published", "")) if pd.notnull(row.get("published")) else None,
                    source=str(row.get("source", "")) if pd.notnull(row.get("source")) else None,
                    entity=str(row.get("entity", "")) if pd.notnull(row.get("entity")) else None,
                    text=str(row.get("text", "")),
                    targeted_score=round(float(row["targeted"]), 4) if pd.notnull(row.get("targeted")) else None,
                    plain_score=round(float(row["plain"]), 4) if pd.notnull(row.get("plain")) else None,
                    label=str(row.get("label", "neutral")) if pd.notnull(row.get("label")) else None,
                    price_report=bool(row.get("price_report", False)),
                )
            )

        # Lead-lag study points
        lead_lag_items: Optional[List[LeadLagPoint]] = None
        lead_lag_file = settings.EXPLAIN_DIR / "lead_lag.csv"
        if lead_lag_file.exists():
            try:
                lldf = pd.read_csv(lead_lag_file)
                lead_lag_items = []
                for _, llrow in lldf.head(30).iterrows():
                    lead_lag_items.append(
                        LeadLagPoint(
                            scope=str(llrow.get("scope", "")),
                            variant=str(llrow.get("variant", "")),
                            lag=int(llrow.get("lag", 0)),
                            rho=round(float(llrow.get("rho", 0.0)), 4),
                            p=round(float(llrow.get("p", 0.0)), 5),
                            n=int(llrow.get("n", 0)),
                        )
                    )
            except Exception:
                lead_lag_items = None

        # Price & sentiment timeline
        timeline_items: Optional[List[SentimentTimelinePoint]] = None
        stock = config.get_stock(key)
        if stock.raw_path.exists() and "trade_date" in df.columns:
            try:
                raw = pd.read_csv(stock.raw_path, usecols=["Date", "Close"])
                last_news_date = str(df["trade_date"].max())
                sub_raw = raw[raw["Date"] <= last_news_date].tail(100)

                daily_news = df.groupby("trade_date").agg(
                    sent=(score_col, "mean"),
                    cnt=(score_col, "count"),
                    top_headline=("text", "first"),
                    label=("label", "first")
                ).reset_index().rename(columns={"trade_date": "Date"})

                merged = pd.merge(sub_raw, daily_news, on="Date", how="left")

                timeline_items = []
                for _, row in merged.iterrows():
                    sent_val = float(row["sent"]) if pd.notnull(row.get("sent")) else None
                    cnt_val = int(row["cnt"]) if pd.notnull(row.get("cnt")) else 0
                    lbl_val = str(row["label"]) if pd.notnull(row.get("label")) else None
                    if sent_val is not None and lbl_val is None:
                        lbl_val = "positive" if sent_val > 0.05 else ("negative" if sent_val < -0.05 else "neutral")

                    timeline_items.append(
                        SentimentTimelinePoint(
                            date=str(row["Date"])[:10],
                            close=round(float(row["Close"]), 2) if pd.notnull(row.get("Close")) else None,
                            sentiment_score=round(sent_val, 4) if sent_val is not None else None,
                            article_count=cnt_val,
                            sentiment_label=lbl_val,
                            top_headline=str(row["top_headline"]) if pd.notnull(row.get("top_headline")) else None
                        )
                    )
            except Exception:
                timeline_items = None

        return NewsResponse(
            ticker=key,
            total_articles=total,
            average_sentiment=round(avg_sent, 4),
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            recent_articles=articles,
            lead_lag=lead_lag_items,
            timeline=timeline_items,
        )

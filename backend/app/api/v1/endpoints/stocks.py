"""
backend/app/api/v1/endpoints/stocks.py
REST endpoints for stock discovery, predictions, forecasting, trading signals, backtesting, and news sentiment.
"""

from typing import Optional
from fastapi import APIRouter, Query

from backend.app.schemas.stock import StockItem, StockListResponse
from backend.app.schemas.prediction import PredictionResponse
from backend.app.schemas.forecast import ForecastResponse
from backend.app.schemas.signals import LatestSignalResponse
from backend.app.schemas.backtest import BacktestResponse
from backend.app.schemas.news import NewsResponse
from backend.app.schemas.status import DataStatusResponse

from backend.app.services.stock_service import StockService
from backend.app.services.prediction_service import PredictionService
from backend.app.services.forecast_service import ForecastService
from backend.app.services.signals_service import SignalsService
from backend.app.services.backtest_service import BacktestService
from backend.app.services.news_service import NewsService
from backend.app.services.status_service import StatusService

router = APIRouter(prefix="/stocks", tags=["Stocks & Analytics"])


@router.get("", response_model=StockListResponse, summary="List All Supported Stocks")
def list_stocks():
    """Returns the list of all registered stocks in the system with their availability statuses."""
    return StockService.get_all_stocks()


@router.get("/{ticker}", response_model=StockItem, summary="Get Stock Details")
def get_stock(ticker: str):
    """Returns metadata and artifact availability flags for a specific stock ticker."""
    return StockService.get_stock(ticker)


@router.get("/{ticker}/prediction", response_model=PredictionResponse, summary="Get Latest Directional Prediction")
def get_prediction(ticker: str):
    """
    Returns latest walk-forward model prediction, probability distribution,
    operating confidence threshold, and abstention signal ('UP', 'DOWN', or 'NO SIGNAL').
    """
    return PredictionService.get_prediction(ticker)


@router.get("/{ticker}/forecast", response_model=ForecastResponse, summary="Get Next-Morning Forecast")
def get_forecast(ticker: str):
    """
    Returns next-morning opening forecast evaluation metrics, model comparisons
    (LightGBM, LogReg, XGBoost), news contribution effect, and annual breakdown.
    """
    return ForecastService.get_forecast(ticker)


@router.get("/{ticker}/signals/latest", response_model=LatestSignalResponse, summary="Get Latest Actionable Trading Signal")
def get_latest_signal(
    ticker: str,
    variant: str = Query("A", description="Strategy variant (A: Baseline, B: Cost Hurdle, C: Tech Confirm, D: Consensus, E: Regime Filtered)"),
    friction_bps: float = Query(15.0, description="Round-trip friction hurdle in basis points (e.g. 15.0)"),
):
    """
    Returns the real-time actionable BUY, SELL, or HOLD signal for the latest market date,
    including calibrated probability, expected return, cost hurdle, and multi-factor flags.
    """
    return SignalsService.get_latest_signal(ticker=ticker, variant=variant, friction_bps=friction_bps)


@router.get("/{ticker}/backtest", response_model=BacktestResponse, summary="Run Walk-Forward Backtest")
def run_backtest(
    ticker: str,
    variant: str = Query("A", description="Strategy variant (A, B, C, D, or E)"),
    long_only: bool = Query(True, description="Enforce long-only execution (Cash on SELL/HOLD)"),
    friction_bps: float = Query(15.0, description="Round-trip Indian transaction friction in basis points"),
    starting_capital: float = Query(1000000.0, description="Starting portfolio capital in INR"),
):
    """
    Simulates chronological walk-forward execution over out-of-sample data points.
    Returns equity curves, individual trades, and quantitative metrics (CAGR, Sharpe, Sortino, Drawdown)
    benchmarked against Buy & Hold and Cash.
    """
    return BacktestService.run_backtest(
        ticker=ticker,
        variant=variant,
        long_only=long_only,
        friction_bps=friction_bps,
        starting_capital=starting_capital,
    )


@router.get("/{ticker}/data-status", response_model=DataStatusResponse, summary="Get Data Freshness & Lineage")
def get_data_status(ticker: str):
    """
    Inspects dataset boundaries, raw OHLCV date limits, training timestamps,
    and out-of-sample evaluation cutoffs to ensure transparent temporal separation.
    """
    return StatusService.get_data_status(ticker)


@router.get("/{ticker}/news", response_model=NewsResponse, summary="Get News Sentiment & Lead-Lag Analysis")
def get_news(
    ticker: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of recent news articles to return"),
):
    """
    Returns entity-targeted FinBERT news sentiment records, positive/negative distributions,
    and cross-correlation lead-lag study metrics.
    """
    return NewsService.get_news_and_sentiment(ticker=ticker, limit=limit)

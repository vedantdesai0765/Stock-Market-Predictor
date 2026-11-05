"""
backend/app/schemas/backtest.py
Pydantic schemas for walk-forward backtest execution and quantitative analytics.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class BacktestEquityPoint(BaseModel):
    date: str
    close: float
    strategy_equity: float
    buy_hold_equity: float
    cash_equity: float
    strategy_return: float
    drawdown: float
    position: float


class TradeRecordItem(BaseModel):
    entry_date: str
    entry_price: float
    position: str
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    net_return: Optional[float] = None
    status: str
    holding_steps: Optional[int] = None


class SeriesMetrics(BaseModel):
    cumulative_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float


class StrategyMetrics(SeriesMetrics):
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_holding_days: float
    total_friction_bps: float
    time_in_market: float
    start_date: str
    end_date: str
    years: float


class BacktestMetricsResponse(BaseModel):
    strategy: StrategyMetrics
    buy_and_hold: SeriesMetrics
    cash: SeriesMetrics


class BacktestResponse(BaseModel):
    ticker: str
    variant: str = Field(..., description="Strategy variant (A: Baseline, B: Cost Hurdle, C: Tech Confirm, D: Consensus, E: Regime Filtered)")
    long_only: bool
    friction_bps: float
    starting_capital: float
    metrics: BacktestMetricsResponse
    equity_curve: List[BacktestEquityPoint]
    trades: List[TradeRecordItem]

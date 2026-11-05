"""
backend/app/services/backtest_service.py
Service adapter for walk-forward backtesting execution and quantitative analytics.
"""

import sys
from typing import List, Optional
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.core.errors import StockNotFoundError, StockMarketPredictorException
from backend.app.schemas.backtest import (
    BacktestEquityPoint,
    BacktestMetricsResponse,
    BacktestResponse,
    SeriesMetrics,
    StrategyMetrics,
    TradeRecordItem,
)

# Ensure src/ is importable
if str(settings.SRC_DIR) not in sys.path:
    sys.path.insert(0, str(settings.SRC_DIR))
if str(settings.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(settings.ROOT_DIR))

import config
import backtest


class BacktestService:
    @staticmethod
    def run_backtest(
        ticker: str,
        variant: str = "A",
        long_only: bool = True,
        friction_bps: float = 15.0,
        starting_capital: float = 1000000.0,
    ) -> BacktestResponse:
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
            res = backtest.run_walk_forward_backtest(
                stock_key=key,
                variant=var,
                long_only=long_only,
                friction_rate=friction_rate,
                risk_free_rate=0.05,
                starting_capital=starting_capital,
                base_dir=settings.ROOT_DIR,
            )
        except Exception as exc:
            raise StockMarketPredictorException(
                message=f"Backtest execution failed for {key}: {str(exc)}",
                status_code=500,
                error_code="BACKTEST_EXECUTION_FAILED",
            )

        eq_df = res["equity_curve"]
        trade_df = res["trade_log"]
        metrics_dict = res["metrics"]

        # Parse metrics
        sm = metrics_dict["strategy"]
        bm = metrics_dict["buy_and_hold"]
        cm = metrics_dict["cash"]

        strategy_metrics = StrategyMetrics(
            cumulative_return=round(float(sm["cumulative_return"]), 4),
            cagr=round(float(sm["cagr"]), 4),
            annualized_volatility=round(float(sm["annualized_volatility"]), 4),
            sharpe_ratio=round(float(sm["sharpe_ratio"]), 4),
            sortino_ratio=round(float(sm["sortino_ratio"]), 4),
            max_drawdown=round(float(sm["max_drawdown"]), 4),
            calmar_ratio=round(float(sm["calmar_ratio"]), 4),
            total_trades=int(sm["total_trades"]),
            win_rate=round(float(sm["win_rate"]), 4),
            profit_factor=round(float(sm["profit_factor"]), 4),
            avg_trade_return=round(float(sm["avg_trade_return"]), 4),
            avg_holding_days=round(float(sm["avg_holding_days"]), 2),
            total_friction_bps=round(float(sm["total_friction_bps"]), 2),
            time_in_market=round(float(sm["time_in_market"]), 4),
            start_date=str(sm["start_date"]),
            end_date=str(sm["end_date"]),
            years=float(sm["years"]),
        )

        bh_metrics = SeriesMetrics(
            cumulative_return=round(float(bm["cumulative_return"]), 4),
            cagr=round(float(bm["cagr"]), 4),
            annualized_volatility=round(float(bm["annualized_volatility"]), 4),
            sharpe_ratio=round(float(bm["sharpe_ratio"]), 4),
            sortino_ratio=round(float(bm["sortino_ratio"]), 4),
            max_drawdown=round(float(bm["max_drawdown"]), 4),
            calmar_ratio=round(float(bm["calmar_ratio"]), 4),
        )

        cash_metrics = SeriesMetrics(
            cumulative_return=round(float(cm["cumulative_return"]), 4),
            cagr=round(float(cm["cagr"]), 4),
            annualized_volatility=round(float(cm["annualized_volatility"]), 4),
            sharpe_ratio=round(float(cm["sharpe_ratio"]), 4),
            sortino_ratio=round(float(cm["sortino_ratio"]), 4),
            max_drawdown=round(float(cm["max_drawdown"]), 4),
            calmar_ratio=round(float(cm["calmar_ratio"]), 4),
        )

        metrics_resp = BacktestMetricsResponse(
            strategy=strategy_metrics,
            buy_and_hold=bh_metrics,
            cash=cash_metrics,
        )

        # Parse equity curve points
        equity_points: List[BacktestEquityPoint] = []
        for _, row in eq_df.iterrows():
            d_val = row["Date"]
            date_str = str(d_val.date()) if hasattr(d_val, "date") else str(d_val)[:10]
            equity_points.append(
                BacktestEquityPoint(
                    date=date_str,
                    close=round(float(row["Close"]), 2),
                    strategy_equity=round(float(row["Strategy_Equity"]), 2),
                    buy_hold_equity=round(float(row["BuyHold_Equity"]), 2),
                    cash_equity=round(float(row["Cash_Equity"]), 2),
                    strategy_return=round(float(row["Strategy_Return"]), 5),
                    drawdown=round(float(row["Strategy_Drawdown"]), 4),
                    position=float(row["Position"]),
                )
            )

        # Parse trade records
        trades: List[TradeRecordItem] = []
        if trade_df is not None and not trade_df.empty:
            for _, tr in trade_df.iterrows():
                trades.append(
                    TradeRecordItem(
                        entry_date=str(tr["entry_date"]),
                        entry_price=round(float(tr["entry_price"]), 2),
                        position=str(tr["position"]),
                        exit_date=str(tr["exit_date"]) if pd.notnull(tr.get("exit_date")) else None,
                        exit_price=round(float(tr["exit_price"]), 2) if pd.notnull(tr.get("exit_price")) else None,
                        net_return=round(float(tr["net_return"]), 4) if pd.notnull(tr.get("net_return")) else None,
                        status=str(tr.get("status", "CLOSED")),
                        holding_steps=int(tr["holding_steps"]) if pd.notnull(tr.get("holding_steps")) else None,
                    )
                )

        return BacktestResponse(
            ticker=key,
            variant=var,
            long_only=long_only,
            friction_bps=friction_bps,
            starting_capital=starting_capital,
            metrics=metrics_resp,
            equity_curve=equity_points,
            trades=trades,
        )

"""
src/backtest.py
Realistic Walk-Forward Backtesting Engine for Indian Equities.

Simulates trading execution over out-of-sample walk-forward periods with:
- Realistic Indian transaction friction (STT, brokerage, exchange fees, stamp duty, slippage)
- Long-only execution (Cash on SELL/HOLD) and optional Long-Short execution
- Accurate turnover-based fee accounting (fees paid only on rebalancing/trade changes)
- Full quantitative performance analytics (CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor)
- Benchmarking against Buy & Hold and Cash
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from signals import (
    DEFAULT_FRICTION_RATE,
    generate_signals,
)


# ---------------------------------------------------------------------------
# Backtest Simulator
# ---------------------------------------------------------------------------

def run_walk_forward_backtest(
    stock_key: str,
    variant: str = "A",
    long_only: bool = True,
    friction_rate: float = DEFAULT_FRICTION_RATE,
    risk_free_rate: float = 0.05,  # 5% annual risk-free rate for Indian G-Secs/FDs
    starting_capital: float = 100000.0,
    base_dir: Optional[Path] = None,
) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
    """
    Executes a walk-forward chronological backtest on the out-of-sample periods.

    Returns:
      A dictionary containing:
        - 'equity_curve': DataFrame with Date, Close, Strategy equity, Buy&Hold equity, Cash equity, Drawdown
        - 'trade_log': DataFrame listing individual entry/exit trades, holding periods, returns, and friction
        - 'metrics': Quantitative performance summary comparing Strategy vs Buy & Hold vs Cash
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    # 1. Generate signals on OOF dataset
    df = generate_signals(
        stock_key=stock_key,
        variant=variant,
        long_only=long_only,
        friction_rate=friction_rate,
        base_dir=base_dir,
    )

    df = df.sort_values("Date").reset_index(drop=True)
    n_rows = len(df)
    if n_rows < 2:
        raise ValueError(f"Insufficient out-of-sample data points for {stock_key} (found {n_rows})")

    # One-way friction rate
    one_way_friction = friction_rate / 2.0

    # 2. Iterate chronologically over non-overlapping holding steps
    # Each row i represents decision at Date[i], position held until Date[i+1]
    # Gross price return over period [i, i+1] is Close[i+1] / Close[i] - 1
    dates = []
    prices = []
    strategy_returns = []
    bh_returns = []
    cash_returns = []
    positions = []
    turnovers = []
    frictions_paid = []

    trade_records = []
    curr_trade = None

    # Track position
    current_pos = 0.0
    
    # Buy & Hold starts with full position of 1.0 (pays entry fee at step 0)
    # Annual risk-free rate per 3-day period (~252 trading days per year, 3 days = 3/252 of year)
    period_rf = (1.0 + risk_free_rate) ** (3.0 / 252.0) - 1.0

    for i in range(n_rows - 1):
        d_entry = df["Date"].iloc[i]
        d_exit = df["Date"].iloc[i + 1]
        p_entry = df["Close"].iloc[i]
        p_exit = df["Close"].iloc[i + 1]
        sig = float(df["signal"].iloc[i])

        # Gross asset return over this step
        gross_asset_ret = (p_exit / p_entry) - 1.0

        # Desired position for period [i, i+1]
        target_pos = sig if not long_only else max(0.0, sig)

        # Turnover calculation
        turnover = abs(target_pos - current_pos)
        step_friction = turnover * one_way_friction

        # Net period strategy return
        # If position is 0 (Cash), earns risk-free rate or 0
        if target_pos == 0.0:
            strat_ret = period_rf - step_friction
        else:
            strat_ret = (target_pos * gross_asset_ret) - step_friction

        # Buy & hold return
        # Step 0 pays entry friction
        bh_fee = one_way_friction if i == 0 else 0.0
        # Last step will pay exit friction
        if i == n_rows - 2:
            bh_fee += one_way_friction
        bh_ret = gross_asset_ret - bh_fee

        # Cash return
        cash_ret = period_rf

        # Record trade log when position changes or trade closes
        if turnover > 0:
            if current_pos != 0 and curr_trade is not None:
                # Close ongoing trade record
                curr_trade["exit_date"] = str(d_entry.date())
                curr_trade["exit_price"] = p_entry
                curr_trade["net_return"] = curr_trade["cum_ret"] - (one_way_friction)
                curr_trade["status"] = "CLOSED"
                trade_records.append(curr_trade)
                curr_trade = None

            if target_pos != 0:
                # Open new trade record
                curr_trade = {
                    "trade_id": len(trade_records) + 1,
                    "entry_date": str(d_entry.date()),
                    "entry_price": p_entry,
                    "position": "LONG" if target_pos > 0 else "SHORT",
                    "exit_date": None,
                    "exit_price": None,
                    "cum_ret": target_pos * gross_asset_ret - (one_way_friction),
                    "friction_paid": step_friction,
                    "holding_steps": 1,
                    "status": "OPEN",
                }
        elif current_pos != 0 and curr_trade is not None:
            # Continue holding
            curr_trade["cum_ret"] = (1.0 + curr_trade["cum_ret"]) * (1.0 + target_pos * gross_asset_ret) - 1.0
            curr_trade["holding_steps"] += 1

        dates.append(d_exit)
        prices.append(p_exit)
        strategy_returns.append(strat_ret)
        bh_returns.append(bh_ret)
        cash_returns.append(cash_ret)
        positions.append(target_pos)
        turnovers.append(turnover)
        frictions_paid.append(step_friction)

        current_pos = target_pos

    # Close any open trade at the end of simulation
    if curr_trade is not None:
        curr_trade["exit_date"] = str(df["Date"].iloc[-1].date())
        curr_trade["exit_price"] = df["Close"].iloc[-1]
        curr_trade["net_return"] = curr_trade["cum_ret"] - one_way_friction
        curr_trade["status"] = "CLOSED"
        trade_records.append(curr_trade)

    # 3. Construct Equity Curves
    eq_df = pd.DataFrame({
        "Date": dates,
        "Close": prices,
        "Position": positions,
        "Strategy_Return": strategy_returns,
        "BuyHold_Return": bh_returns,
        "Cash_Return": cash_returns,
        "Friction_Paid": frictions_paid,
    })

    eq_df["Strategy_Equity"] = starting_capital * np.cumprod(1.0 + eq_df["Strategy_Return"])
    eq_df["BuyHold_Equity"] = starting_capital * np.cumprod(1.0 + eq_df["BuyHold_Return"])
    eq_df["Cash_Equity"] = starting_capital * np.cumprod(1.0 + eq_df["Cash_Return"])

    # Underwater Drawdown calculation
    strat_peak = eq_df["Strategy_Equity"].cummax()
    eq_df["Strategy_Drawdown"] = (eq_df["Strategy_Equity"] - strat_peak) / strat_peak

    bh_peak = eq_df["BuyHold_Equity"].cummax()
    eq_df["BuyHold_Drawdown"] = (eq_df["BuyHold_Equity"] - bh_peak) / bh_peak

    trade_df = pd.DataFrame(trade_records)

    # 4. Compute Statistical & Performance Metrics
    metrics = _compute_metrics(
        eq_df=eq_df,
        trade_df=trade_df,
        risk_free_rate=risk_free_rate,
        friction_rate=friction_rate,
    )

    return {
        "stock": stock_key,
        "variant": variant,
        "long_only": long_only,
        "equity_curve": eq_df,
        "trade_log": trade_df,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Quantitative Metrics Engine
# ---------------------------------------------------------------------------

def _compute_metrics(
    eq_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    risk_free_rate: float,
    friction_rate: float,
) -> Dict[str, Dict[str, float]]:
    """
    Computes rigorous risk-adjusted performance metrics for Strategy vs Buy&Hold vs Cash.
    """
    start_date = eq_df["Date"].iloc[0]
    end_date = eq_df["Date"].iloc[-1]
    n_days = max(1, (end_date - start_date).days)
    years = n_days / 365.25
    n_periods = len(eq_df)
    periods_per_year = 252.0 / 3.0  # ~84 non-overlapping 3-day steps per year

    def calc_series_metrics(ret_series: pd.Series, eq_series: pd.Series) -> Dict[str, float]:
        cum_ret = float((eq_series.iloc[-1] / eq_series.iloc[0]) - 1.0)
        cagr = float((1.0 + cum_ret) ** (1.0 / max(0.1, years)) - 1.0) if cum_ret > -1.0 else -1.0
        
        # Volatility
        mean_ret = float(ret_series.mean())
        std_ret = float(ret_series.std())
        ann_vol = float(std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

        # Sharpe Ratio (annualized excess return over risk-free rate)
        sharpe = (cagr - risk_free_rate) / ann_vol if ann_vol > 1e-6 else 0.0

        # Downside Deviation & Sortino Ratio
        downside_ret = ret_series[ret_series < 0]
        downside_std = float(downside_ret.std()) if len(downside_ret) > 1 else 0.0
        ann_downside_vol = float(downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0
        sortino = (cagr - risk_free_rate) / ann_downside_vol if ann_downside_vol > 1e-6 else 0.0

        # Max Drawdown
        peak = eq_series.cummax()
        dd = (eq_series - peak) / peak
        max_dd = float(abs(dd.min()))

        # Calmar Ratio
        calmar = cagr / max_dd if max_dd > 1e-4 else 0.0

        return {
            "cumulative_return": cum_ret,
            "cagr": cagr,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
        }

    strat_m = calc_series_metrics(eq_df["Strategy_Return"], eq_df["Strategy_Equity"])
    bh_m = calc_series_metrics(eq_df["BuyHold_Return"], eq_df["BuyHold_Equity"])
    cash_m = calc_series_metrics(eq_df["Cash_Return"], eq_df["Cash_Equity"])

    # Trading specific metrics
    n_trades = len(trade_df) if not trade_df.empty else 0
    if n_trades > 0 and "net_return" in trade_df.columns:
        wins = trade_df[trade_df["net_return"] > 0]
        losses = trade_df[trade_df["net_return"] <= 0]
        win_rate = len(wins) / n_trades
        gross_gains = wins["net_return"].sum()
        gross_losses = abs(losses["net_return"].sum())
        profit_factor = (gross_gains / gross_losses) if gross_losses > 1e-6 else (10.0 if gross_gains > 0 else 0.0)
        avg_trade_ret = float(trade_df["net_return"].mean())
        avg_holding = float(trade_df["holding_steps"].mean() * 3.0)  # in trading days
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_ret = 0.0
        avg_holding = 0.0

    total_friction_bps = float(eq_df["Friction_Paid"].sum() * 10000.0)
    time_in_market = float((eq_df["Position"] != 0).mean())

    strat_m.update({
        "total_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_return": avg_trade_ret,
        "avg_holding_days": avg_holding,
        "total_friction_bps": total_friction_bps,
        "time_in_market": time_in_market,
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "years": round(years, 2),
    })

    return {
        "strategy": strat_m,
        "buy_and_hold": bh_m,
        "cash": cash_m,
    }


# ---------------------------------------------------------------------------
# Multi-Stock Batch Evaluation (For Audit & Documentation)
# ---------------------------------------------------------------------------

def run_all_variants_audit(
    stock_keys: Optional[List[str]] = None,
    variants: Tuple[str, ...] = ("A", "B", "C", "D", "E"),
    long_only: bool = True,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Executes backtest for all stocks across all 5 strategy variants,
    compiling results into a comparison dataframe.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    if stock_keys is None:
        import config
        stock_keys = config.list_stocks()

    records = []
    for s in stock_keys:
        for var in variants:
            try:
                res = run_walk_forward_backtest(
                    stock_key=s,
                    variant=var,
                    long_only=long_only,
                    base_dir=base_dir,
                )
                st_m = res["metrics"]["strategy"]
                bh_m = res["metrics"]["buy_and_hold"]

                records.append({
                    "stock": s,
                    "variant": var,
                    "cagr_strat": st_m["cagr"],
                    "cagr_bh": bh_m["cagr"],
                    "sharpe_strat": st_m["sharpe_ratio"],
                    "sharpe_bh": bh_m["sharpe_ratio"],
                    "max_dd_strat": st_m["max_drawdown"],
                    "max_dd_bh": bh_m["max_drawdown"],
                    "win_rate": st_m["win_rate"],
                    "profit_factor": st_m["profit_factor"],
                    "total_trades": st_m["total_trades"],
                    "friction_bps": st_m["total_friction_bps"],
                    "time_in_market": st_m["time_in_market"],
                })
            except Exception as e:
                print(f"Error evaluating {s} variant {var}: {e}")

    return pd.DataFrame(records)

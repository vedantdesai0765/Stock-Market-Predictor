"""
src/signals_page.py
Actionable Trading Signal Terminal & Walk-Forward Performance View.

Renders:
- Current live actionable signal badge (BUY / SELL / HOLD)
- Calibrated probability, expected return vs. transaction friction hurdle
- Multi-factor confirmation checklist (Model, RSI, MACD, Trend, FinBERT, Regime)
- Chronological walk-forward backtest equity curves vs. Buy & Hold and Cash
- Underwater drawdown curves
- Candlestick chart with historical trade entry/exit markers
- Detailed trade execution log
- Academic & regulatory research disclaimers
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from backtest import run_walk_forward_backtest
from signals import (
    DEFAULT_FRICTION_BPS,
    get_latest_signal_status,
)

GREEN = "#10b981"
RED = "#ef4444"
AMBER = "#f59e0b"
BLUE = "#3b82f6"
PURPLE = "#8b5cf6"
SLATE = "#94a3b8"
CARD_BG = "#121824"
BORDER_COLOR = "#1f293d"

VARIANT_NAMES = {
    "A": "Variant A: Model Alone (Confidence & Cost Gated)",
    "B": "Variant B: Model + Technical Confirmation (RSI, MACD, SMA)",
    "C": "Variant C: Model + Sentiment Confirmation (Entity FinBERT)",
    "D": "Variant D: Multi-Factor Consensus (Model + Tech + Sentiment)",
    "E": "Variant E: Regime-Filtered Strategy (Volatility & Trend Filtered)",
}


def render_signals_page(
    stock_key: str,
    stock_display_names: Dict[str, str],
    chart_layout_fn: Callable,
    theme: Dict[str, str],
) -> None:
    """
    Renders the Actionable Trading Signal View in the Streamlit terminal.
    """
    display_name = stock_display_names.get(stock_key, stock_key)

    # -------------------------------------------------------------------------
    # 1. Top Badges & Header
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div style='display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px'>
      <span style='background:rgba(37,99,235,0.15);color:#60a5fa;border:1px solid rgba(37,99,235,0.3);border-radius:4px;padding:3px 9px;font-size:11px;font-weight:700;letter-spacing:0.5px'>ACTIONABLE SIGNAL ENGINE</span>
      <span style='background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.3);border-radius:4px;padding:3px 9px;font-size:11px;font-weight:700;letter-spacing:0.5px'>EVIDENCE-BASED DISCIPLINE</span>
      <span style='background:rgba(245,158,11,0.15);color:#fbbf24;border:1px solid rgba(245,158,11,0.3);border-radius:4px;padding:3px 9px;font-size:11px;font-weight:700;letter-spacing:0.5px'>NSE TRANSACTION FRICTION (15 BPS)</span>
      <span style='background:rgba(168,85,247,0.15);color:#c084fc;border:1px solid rgba(168,85,247,0.3);border-radius:4px;padding:3px 9px;font-size:11px;font-weight:700;letter-spacing:0.5px'>NON-OVERLAPPING 3-DAY HORIZON</span>
      <span style='background:rgba(148,163,184,0.12);color:#94a3b8;border:1px solid rgba(148,163,184,0.25);border-radius:4px;padding:3px 9px;font-size:11px;font-weight:600;margin-left:auto'>RESEARCH PROTOCOL · NOT FINANCIAL ADVICE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='padding:2px 0 16px 0;display:flex;justify-content:space-between;align-items:flex-end'>
      <div>
        <h1 style='margin:0;font-size:26px'>Actionable Trading Signals & Backtesting</h1>
        <div style='font-size:13px;color:{theme.get("text_muted", SLATE)};margin-top:4px'>
          <strong>{display_name} ({stock_key})</strong> &nbsp;•&nbsp; Systematic Decision Engine with Cost-Hurdle Gating
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Prominent Academic / Disclaimer Banner
    st.markdown(f"""
    <div style='background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.25);border-left:4px solid #f59e0b;border-radius:6px;padding:12px 16px;margin-bottom:20px;font-size:12.5px;color:#f8fafc;line-height:1.5'>
      <strong>RESEARCH REPRODUCIBILITY NOTICE:</strong> This trading engine is an empirical simulation designed to test whether machine learning predictions provide exploitable statistical edge after realistic transaction friction (brokerage, STT, exchange charges, slippage). It enforces <em>abstention discipline</em>: default state is <strong>HOLD (Cash)</strong> whenever confidence is low, technicals conflict, or expected returns fail to clear execution costs.
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Strategy Controls
    # -------------------------------------------------------------------------
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2.5, 1.2, 1.3])
    with col_ctrl1:
        variant_key = st.selectbox(
            "Strategy Variant",
            options=["A", "B", "C", "D", "E"],
            format_func=lambda k: VARIANT_NAMES[k],
            index=2,  # Default to Variant C (Model + Sentiment)
            help="Select which multi-factor confirmation strategy to evaluate.",
        )
    with col_ctrl2:
        exec_mode = st.selectbox(
            "Execution Mode",
            options=["Long-Only (Cash on Neutral/Sell)", "Long-Short (Allow Shorting)"],
            index=0,
            help="Long-only is standard for Indian delivery equity trading.",
        )
        is_long_only = bool(exec_mode.startswith("Long-Only"))
    with col_ctrl3:
        friction_bps = st.number_input(
            "Round-Trip Friction (bps)",
            min_value=5.0,
            max_value=50.0,
            value=float(DEFAULT_FRICTION_BPS),
            step=2.5,
            help="Round-trip transaction friction in basis points (15 bps = 0.15%). Includes STT, brokerage, turnover fees, stamp duty, GST, and slippage.",
        )
        friction_rate = friction_bps / 10000.0

    # -------------------------------------------------------------------------
    # 3. Current Live Signal Status Card
    # -------------------------------------------------------------------------
    latest = get_latest_signal_status(stock_key, variant=variant_key, friction_rate=friction_rate)

    sig_label = latest["signal_label"]
    p_up = latest["calibrated_proba_up"]
    conf = latest["confidence"]
    exp_gross = latest["expected_gross_return"]
    net_exp = latest["net_expectancy"]
    rationale = latest["rationale"]
    latest_date = latest["date"]
    latest_price = latest["close"]

    # Pill styles
    if sig_label == "BUY":
        pill_bg = "rgba(16, 185, 129, 0.15)"
        pill_border = "rgba(16, 185, 129, 0.5)"
        pill_color = GREEN
    elif sig_label == "SELL":
        pill_bg = "rgba(239, 68, 68, 0.15)"
        pill_border = "rgba(239, 68, 68, 0.5)"
        pill_color = RED
    else:
        pill_bg = "rgba(245, 158, 11, 0.15)"
        pill_border = "rgba(245, 158, 11, 0.5)"
        pill_color = AMBER

    st.markdown(f"""
    <div style='background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:8px;padding:20px;margin-bottom:24px'>
      <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:16px'>
        <div>
          <span style='font-size:11px;font-weight:700;color:{SLATE};letter-spacing:1px;text-transform:uppercase'>CURRENT ACTIONABLE SIGNAL</span>
          <div style='margin-top:6px;display:flex;align-items:center;gap:12px'>
            <span style='background:{pill_bg};border:1.5px solid {pill_border};color:{pill_color};font-family:monospace;font-size:22px;font-weight:800;padding:6px 22px;border-radius:24px;letter-spacing:1px'>
              {sig_label}
            </span>
            <span style='font-size:14px;color:#f8fafc'>
              As of <strong>{latest_date}</strong> (Close: ₹{latest_price:,.2f})
            </span>
          </div>
        </div>
        <div style='display:flex;gap:20px;text-align:right'>
          <div>
            <div style='font-size:11px;color:{SLATE};font-weight:600'>CALIBRATED P(UP)</div>
            <div style='font-size:18px;font-weight:700;font-family:monospace;color:#f8fafc'>{p_up:.1%}</div>
            <div style='font-size:10px;color:{SLATE}'>Conf: {conf:+.1%}</div>
          </div>
          <div>
            <div style='font-size:11px;color:{SLATE};font-weight:600'>EXP. GROSS RETURN</div>
            <div style='font-size:18px;font-weight:700;font-family:monospace;color:{"#10b981" if exp_gross > 0 else ("#ef4444" if exp_gross < 0 else "#f8fafc")}'>{exp_gross:+.2%}</div>
            <div style='font-size:10px;color:{SLATE}'>3-day horizon</div>
          </div>
          <div>
            <div style='font-size:11px;color:{SLATE};font-weight:600'>COST HURDLE</div>
            <div style='font-size:18px;font-weight:700;font-family:monospace;color:#f59e0b'>{friction_rate:.2%}</div>
            <div style='font-size:10px;color:{SLATE}'>{friction_bps:.0f} bps roundtrip</div>
          </div>
          <div>
            <div style='font-size:11px;color:{SLATE};font-weight:600'>NET EXPECTANCY</div>
            <div style='font-size:18px;font-weight:700;font-family:monospace;color:{"#10b981" if net_exp > 0 else ("#ef4444" if net_exp < 0 else SLATE)}'>{net_exp:+.2%}</div>
            <div style='font-size:10px;color:{SLATE}'>Gross − Friction</div>
          </div>
        </div>
      </div>
      <div style='background:rgba(15,23,42,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:6px;padding:10px 14px;font-size:13px;color:#cbd5e1'>
        <strong>Decision Rationale:</strong> {rationale}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Multi-Factor Confirmation Matrix
    # -------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94a3b8;margin-bottom:12px'>MULTI-FACTOR CONFIRMATION MATRIX</div>", unsafe_allow_html=True)

    mf_col1, mf_col2, mf_col3, mf_col4, mf_col5 = st.columns(5)
    with mf_col1:
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:6px;padding:14px;min-height:125px'>
          <div style='font-size:11px;font-weight:700;color:{SLATE}'>1. MODEL PROBABILITY</div>
          <div style='font-size:16px;font-weight:700;margin-top:6px;color:#f8fafc'>{p_up:.1%}</div>
          <div style='font-size:11px;color:{GREEN if p_up > 0.52 else (RED if p_up < 0.48 else SLATE)};margin-top:4px'>
            {"✓ Bullish Edge" if p_up > 0.52 else ("✗ Bearish Edge" if p_up < 0.48 else "— Neutral Zone")}
          </div>
          <div style='font-size:10px;color:{SLATE};margin-top:2px'>Calibrated In-Fold</div>
        </div>
        """, unsafe_allow_html=True)

    with mf_col2:
        rsi_val = latest["rsi"]
        rsi_state = "Overbought (>70)" if rsi_val >= 70 else ("Oversold (<30)" if rsi_val <= 30 else "Normal (30–70)")
        rsi_color = RED if rsi_val >= 70 else (GREEN if rsi_val <= 30 else "#f8fafc")
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:6px;padding:14px;min-height:125px'>
          <div style='font-size:11px;font-weight:700;color:{SLATE}'>2. RSI (14-PERIOD)</div>
          <div style='font-size:16px;font-weight:700;margin-top:6px;color:{rsi_color}'>{rsi_val:.1f}</div>
          <div style='font-size:11px;color:{SLATE};margin-top:4px'>{rsi_state}</div>
          <div style='font-size:10px;color:{GREEN if 30 <= rsi_val <= 70 else AMBER};margin-top:2px'>
            {"✓ Safe Entry Range" if 30 <= rsi_val <= 70 else "⚠ Overextended"}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with mf_col3:
        macd_val = latest["macd_hist"]
        macd_color = GREEN if macd_val > 0 else (RED if macd_val < 0 else SLATE)
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:6px;padding:14px;min-height:125px'>
          <div style='font-size:11px;font-weight:700;color:{SLATE}'>3. MACD MOMENTUM</div>
          <div style='font-size:16px;font-weight:700;margin-top:6px;color:{macd_color}'>{macd_val:+.3f}</div>
          <div style='font-size:11px;color:{macd_color};margin-top:4px'>
            {"Bullish Momentum" if macd_val > 0 else ("Bearish Momentum" if macd_val < 0 else "Neutral")}
          </div>
          <div style='font-size:10px;color:{SLATE};margin-top:2px'>Histogram Signal</div>
        </div>
        """, unsafe_allow_html=True)

    with mf_col4:
        sent_val = latest["sentiment_score"]
        sent_color = GREEN if sent_val >= 0.05 else (RED if sent_val <= -0.05 else SLATE)
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:6px;padding:14px;min-height:125px'>
          <div style='font-size:11px;font-weight:700;color:{SLATE}'>4. FINBERT SENTIMENT</div>
          <div style='font-size:16px;font-weight:700;margin-top:6px;color:{sent_color}'>{sent_val:+.2f}</div>
          <div style='font-size:11px;color:{sent_color};margin-top:4px'>
            {"Bullish Tone" if sent_val >= 0.05 else ("Bearish Tone" if sent_val <= -0.05 else "Neutral / No News")}
          </div>
          <div style='font-size:10px;color:{SLATE};margin-top:2px'>Entity-Targeted Model</div>
        </div>
        """, unsafe_allow_html=True)

    with mf_col5:
        regime_val = latest["market_regime"]
        regime_color = GREEN if regime_val == "TRENDING_BULL" else (RED if regime_val == "TRENDING_BEAR" else (AMBER if regime_val == "HIGH_VOLATILITY" else SLATE))
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:6px;padding:14px;min-height:125px'>
          <div style='font-size:11px;font-weight:700;color:{SLATE}'>5. MARKET REGIME</div>
          <div style='font-size:14px;font-weight:700;margin-top:6px;color:{regime_color}'>{regime_val.replace("_", " ")}</div>
          <div style='font-size:11px;color:{SLATE};margin-top:4px'>
            {"Trend Following" if "TRENDING" in regime_val else ("Risk Off (High Vol)" if regime_val == "HIGH_VOLATILITY" else "Choppy")}
          </div>
          <div style='font-size:10px;color:{SLATE};margin-top:2px'>SMA & Volatility Gating</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5. Walk-Forward Backtest Execution & Metrics Scorecard
    # -------------------------------------------------------------------------
    st.markdown("<div style='font-size:13px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:#94a3b8;margin-top:28px;margin-bottom:14px'>WALK-FORWARD OUT-OF-SAMPLE BACKTEST RESULTS</div>", unsafe_allow_html=True)

    with st.spinner("Running chronological walk-forward backtest simulation..."):
        backtest_res = run_walk_forward_backtest(
            stock_key=stock_key,
            variant=variant_key,
            long_only=is_long_only,
            friction_rate=friction_rate,
        )

    strat_m = backtest_res["metrics"]["strategy"]
    bh_m = backtest_res["metrics"]["buy_and_hold"]
    cash_m = backtest_res["metrics"]["cash"]
    eq_df = backtest_res["equity_curve"]
    trade_df = backtest_res["trade_log"]

    # 6 Metrics Tiles
    mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
    with mcol1:
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:6px;padding:12px;text-align:center'>
          <div style='font-size:11px;color:{SLATE};font-weight:600'>CAGR (ANNUALIZED)</div>
          <div style='font-size:20px;font-weight:700;font-family:monospace;color:{GREEN if strat_m["cagr"] >= bh_m["cagr"] else "#f8fafc"}'>{strat_m["cagr"]:.2%}</div>
          <div style='font-size:10px;color:{SLATE}'>B&H: {bh_m["cagr"]:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol2:
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:6px;padding:12px;text-align:center'>
          <div style='font-size:11px;color:{SLATE};font-weight:600'>SHARPE RATIO</div>
          <div style='font-size:20px;font-weight:700;font-family:monospace;color:{GREEN if strat_m["sharpe_ratio"] >= bh_m["sharpe_ratio"] else "#f8fafc"}'>{strat_m["sharpe_ratio"]:.2f}</div>
          <div style='font-size:10px;color:{SLATE}'>B&H: {bh_m["sharpe_ratio"]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol3:
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:6px;padding:12px;text-align:center'>
          <div style='font-size:11px;color:{SLATE};font-weight:600'>MAX DRAWDOWN</div>
          <div style='font-size:20px;font-weight:700;font-family:monospace;color:{GREEN if strat_m["max_drawdown"] < bh_m["max_drawdown"] else RED}'>{strat_m["max_drawdown"]:.2%}</div>
          <div style='font-size:10px;color:{SLATE}'>B&H: {bh_m["max_drawdown"]:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol4:
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:6px;padding:12px;text-align:center'>
          <div style='font-size:11px;color:{SLATE};font-weight:600'>WIN RATE</div>
          <div style='font-size:20px;font-weight:700;font-family:monospace;color:#f8fafc'>{strat_m["win_rate"]:.1%}</div>
          <div style='font-size:10px;color:{SLATE}'>Profit Factor: {strat_m["profit_factor"]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol5:
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:6px;padding:12px;text-align:center'>
          <div style='font-size:11px;color:{SLATE};font-weight:600'>TOTAL TRADES</div>
          <div style='font-size:20px;font-weight:700;font-family:monospace;color:#f8fafc'>{strat_m["total_trades"]}</div>
          <div style='font-size:10px;color:{SLATE}'>Avg Hold: {strat_m["avg_holding_days"]:.0f}d</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol6:
        st.markdown(f"""
        <div style='background:{CARD_BG};border:1px solid {BORDER_COLOR};border-radius:6px;padding:12px;text-align:center'>
          <div style='font-size:11px;color:{SLATE};font-weight:600'>TIME IN MARKET</div>
          <div style='font-size:20px;font-weight:700;font-family:monospace;color:#f8fafc'>{strat_m["time_in_market"]:.1%}</div>
          <div style='font-size:10px;color:{AMBER}'>Friction: {strat_m["total_friction_bps"]:.0f} bps</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 6. Interactive Visualizations
    # -------------------------------------------------------------------------
    # Tab 1: Equity Curves, Tab 2: Underwater Drawdown, Tab 3: Historical Signal Markers
    tab_eq, tab_dd, tab_chart = st.tabs(["📈 Equity Growth & Benchmark Comparison", "🌊 Underwater Drawdown", "🎯 Historical Signal Markers"])

    with tab_eq:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq_df["Date"], y=eq_df["Strategy_Equity"],
            mode="lines", name=f"Strategy ({variant_key})",
            line=dict(color=GREEN, width=2.2),
            hovertemplate="Strategy: ₹%{y:,.0f}<extra></extra>"
        ))
        fig_eq.add_trace(go.Scatter(
            x=eq_df["Date"], y=eq_df["BuyHold_Equity"],
            mode="lines", name="Buy & Hold Benchmark",
            line=dict(color=BLUE, width=1.5, dash="dash"),
            hovertemplate="Buy & Hold: ₹%{y:,.0f}<extra></extra>"
        ))
        fig_eq.add_trace(go.Scatter(
            x=eq_df["Date"], y=eq_df["Cash_Equity"],
            mode="lines", name="Cash (Risk-Free 5% p.a.)",
            line=dict(color=SLATE, width=1.2, dash="dot"),
            hovertemplate="Cash: ₹%{y:,.0f}<extra></extra>"
        ))

        eq_layout = chart_layout_fn(400, f"{stock_key} — PORTFOLIO VALUE (START: ₹100,000)")
        eq_layout["yaxis"]["title"] = dict(text="Portfolio Value (INR ₹)", font=dict(size=10))
        eq_layout["xaxis"]["title"] = dict(text="Out-of-Sample Date", font=dict(size=10))
        fig_eq.update_layout(**eq_layout)
        st.plotly_chart(fig_eq, use_container_width=True)

    with tab_dd:
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=eq_df["Date"], y=eq_df["Strategy_Drawdown"] * 100,
            mode="lines", name=f"Strategy ({variant_key})",
            fill="tozeroy",
            line=dict(color=GREEN, width=1.6),
            fillcolor="rgba(16, 185, 129, 0.12)",
            hovertemplate="Strategy Drawdown: %{y:.1f}%<extra></extra>"
        ))
        fig_dd.add_trace(go.Scatter(
            x=eq_df["Date"], y=eq_df["BuyHold_Drawdown"] * 100,
            mode="lines", name="Buy & Hold Drawdown",
            line=dict(color=RED, width=1.2, dash="dash"),
            hovertemplate="Buy&Hold Drawdown: %{y:.1f}%<extra></extra>"
        ))

        dd_layout = chart_layout_fn(360, f"{stock_key} — UNDERWATER DRAWDOWN PROFILE (%)")
        dd_layout["yaxis"]["title"] = dict(text="Drawdown from Peak (%)", font=dict(size=10))
        dd_layout["xaxis"]["title"] = dict(text="Date", font=dict(size=10))
        fig_dd.update_layout(**dd_layout)
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab_chart:
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(
            x=eq_df["Date"], y=eq_df["Close"],
            mode="lines", name="Close Price",
            line=dict(color="#cbd5e1", width=1.4),
            hovertemplate="Close: ₹%{y:,.2f}<extra></extra>"
        ))

        # Add BUY markers
        buy_points = eq_df[eq_df["Position"] > 0]
        if not buy_points.empty:
            fig_sig.add_trace(go.Scatter(
                x=buy_points["Date"], y=buy_points["Close"],
                mode="markers", name="Active Long Position",
                marker=dict(symbol="triangle-up", size=7, color=GREEN),
                hovertemplate="Invested: ₹%{y:,.2f}<extra></extra>"
            ))

        sig_layout = chart_layout_fn(400, f"{stock_key} — HISTORICAL POSITION OVERLAY ON PRICE")
        sig_layout["yaxis"]["title"] = dict(text="Price (INR ₹)", font=dict(size=10))
        fig_sig.update_layout(**sig_layout)
        st.plotly_chart(fig_sig, use_container_width=True)

    # -------------------------------------------------------------------------
    # 7. Trade Log & Export
    # -------------------------------------------------------------------------
    with st.expander(f"📋 View Trade Execution Log ({len(trade_df)} Completed Trades)", expanded=False):
        if not trade_df.empty:
            trade_display = trade_df.copy()
            trade_display["net_return_pct"] = trade_display["net_return"].apply(lambda r: f"{r:+.2%}")
            trade_display["cum_ret_pct"] = trade_display["cum_ret"].apply(lambda r: f"{r:+.2%}")
            trade_display["holding_days"] = trade_display["holding_steps"] * 3
            trade_display["entry_price"] = trade_display["entry_price"].apply(lambda p: f"₹{p:,.2f}")
            trade_display["exit_price"] = trade_display["exit_price"].apply(lambda p: f"₹{p:,.2f}")

            st.dataframe(
                trade_display[[
                    "trade_id", "position", "entry_date", "entry_price",
                    "exit_date", "exit_price", "holding_days", "net_return_pct"
                ]].rename(columns={
                    "trade_id": "ID",
                    "position": "Type",
                    "entry_date": "Entry Date",
                    "entry_price": "Entry Price",
                    "exit_date": "Exit Date",
                    "exit_price": "Exit Price",
                    "holding_days": "Days Held",
                    "net_return_pct": "Net Return",
                }),
                use_container_width=True,
                height=300,
            )
        else:
            st.info("No trades executed. Strategy abstained across all evaluation dates.")

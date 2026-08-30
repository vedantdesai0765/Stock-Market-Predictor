"""
Stock Market Predictor — Quantitative Financial Analytics Terminal

Dark Financial Terminal (Default Theme Only):
- Dark mode locked: charcoal navy background (#0b0f19), dark slate cards (#121824), off-white typography (#f8fafc)
- Clean sidebar matching design mock (STOCK PREDICTOR / QUANTITATIVE ANALYTICS TERMINAL)
- Stock selector with human-readable company names & ticker symbols
- Primary blue action button (▶ RUN ANALYSIS)
- Validation methodology checklist with green circular check icons
- Aligned model specification grid
- Preserved 100% of underlying ML models, datasets, validation methodology, and predictions
"""

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup & Module imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config
from dataset import load_raw_ohlcv
from features import build_features
from stationary import add_stationary_features
from pipeline import (artifacts_exist, load_artifacts, predict_latest)

STOCK_DISPLAY_NAMES = {
    "ADANIENT": "Adani Enterprises (ADANIENT)",
    "ADANIGREEN": "Adani Green Energy (ADANIGREEN)",
    "ETERNAL": "Eternal Limited (ETERNAL)",
    "HDFCBANK": "HDFC Bank (HDFCBANK)",
    "INFY": "Infosys (INFY)",
    "ITC": "ITC (ITC)",
    "RELIANCE": "Reliance Industries (RELIANCE)",
    "SBI": "State Bank of India (SBI)",
    "TCS": "Tata Consultancy Services (TCS)",
    "VEDL": "Vedanta (VEDL)",
}

# ---------------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Market Predictor | Quantitative Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar Layout
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='font-size:18px;font-weight:800;letter-spacing:0.5px;margin-bottom:2px;color:#f8fafc'>STOCK PREDICTOR</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;font-weight:600;letter-spacing:0.5px;color:#94a3b8;margin-bottom:22px;'>QUANTITATIVE ANALYTICS TERMINAL</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-group-title'>STOCK SELECTION</div>", unsafe_allow_html=True)
    available_stocks = config.list_stocks()
    stock_choice = st.selectbox(
        "Select Ticker",
        options=available_stocks,
        format_func=lambda key: STOCK_DISPLAY_NAMES.get(key, key),
        index=0,
    )

    st.button("▶ RUN ANALYSIS", width="stretch")

    st.markdown("<div style='margin-top:24px;' class='sidebar-group-title'>VALIDATION METHODOLOGY</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top:10px;'>
      <div class='checklist-row'>
        <div class='check-circle'>✓</div>
        <span>Stationary features (scale-free)</span>
      </div>
      <div class='checklist-row'>
        <div class='check-circle'>✓</div>
        <span>Non-overlapping labels</span>
      </div>
      <div class='checklist-row'>
        <div class='check-circle'>✓</div>
        <span>5-Fold Walk-forward CV</span>
      </div>
      <div class='checklist-row'>
        <div class='check-circle'>✓</div>
        <span>52-Day Embargo gap</span>
      </div>
      <div class='checklist-row'>
        <div class='check-circle'>✓</div>
        <span>100-Run Permutation test</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:24px;' class='sidebar-group-title'>MODEL SPECIFICATION</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top:10px;'>
      <div class='spec-row'>
        <span class='spec-key'>Architecture:</span>
        <span class='spec-val'>Logistic Regression</span>
      </div>
      <div class='spec-row'>
        <span class='spec-key'>Regularisation:</span>
        <span class='spec-val'>L2 Penalty (C=0.1)</span>
      </div>
      <div class='spec-row'>
        <span class='spec-key'>Scaling:</span>
        <span class='spec-val'>StandardScaler (in-fold)</span>
      </div>
      <div class='spec-row'>
        <span class='spec-key'>Features:</span>
        <span class='spec-val'>52 Stationary Ratios</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Dark Financial Terminal CSS Engine
# ---------------------------------------------------------------------------
bg_main = "#0b0f19"
card_bg = "#121824"
card_border = "#1f293d"
text_primary = "#f8fafc"
text_muted = "#94a3b8"
heading_border = "#1f293d"
sidebar_bg = "#070a12"
input_bg = "#121824"
input_border = "#1f293d"
input_text = "#f8fafc"
popover_bg = "#121824"
popover_text = "#cbd5e1"
hover_bg = "#1e293b"

banner_noact_bg = "rgba(245, 158, 11, 0.08)"
banner_noact_border = "rgba(245, 158, 11, 0.3)"
banner_noact_title = "#f59e0b"
banner_noact_desc = "#cbd5e1"

banner_act_bg = "rgba(16, 185, 129, 0.08)"
banner_act_border = "rgba(16, 185, 129, 0.3)"
banner_act_title = "#10b981"
banner_act_desc = "#cbd5e1"

pill_noact_color = "#f59e0b"

chart_plot_bg = "rgba(18, 24, 36, 0.6)"
chart_paper_bg = "rgba(0,0,0,0)"
chart_text_color = "#cbd5e1"
chart_title_color = "#f8fafc"
chart_grid_color = "#1f293d"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }}

  .stApp {{
    background-color: {bg_main};
    color: {text_primary};
  }}

  h1, h2, h3 {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: {text_primary} !important;
  }}

  .mono-text {{
    font-family: 'Space Mono', monospace !important;
  }}

  /* Sidebar Styling */
  div[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    border-right: 1px solid {card_border};
  }}

  div[data-testid="stSidebar"] label, div[data-testid="stSidebar"] span, div[data-testid="stSidebar"] p {{
    color: {text_muted} !important;
  }}

  .sidebar-group-title {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {text_muted};
    margin-bottom: 8px;
  }}

  /* Primary Blue Action Button */
  .stButton > button {{
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    border: none !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 10px 16px !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.2s ease-in-out !important;
    width: 100% !important;
  }}

  .stButton > button:hover {{
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    box-shadow: 0 6px 12px -2px rgba(37, 99, 235, 0.4) !important;
    color: #ffffff !important;
  }}

  /* Sidebar Checklist & Specs Layout */
  .checklist-row {{
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    font-size: 12px;
    color: {text_primary};
  }}

  .check-circle {{
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: bold;
    margin-right: 10px;
    flex-shrink: 0;
  }}

  .spec-row {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
    font-size: 12px;
  }}

  .spec-key {{
    font-weight: 700;
    color: {text_primary};
  }}

  .spec-val {{
    color: {text_muted};
  }}

  /* Terminal Card Component */
  .terminal-card {{
    background: {card_bg};
    border: 1px solid {card_border};
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  }}

  .terminal-card-header {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {text_muted};
    margin-bottom: 8px;
  }}

  .terminal-card-value {{
    font-family: 'Space Mono', monospace;
    font-size: 24px;
    font-weight: 700;
    color: {text_primary};
    line-height: 1.2;
  }}

  .terminal-card-sub {{
    font-size: 12px;
    color: {text_muted};
    margin-top: 6px;
    line-height: 1.4;
  }}

  /* Validation Status Banners */
  .status-banner-actionable {{
    background: {banner_act_bg};
    border: 1px solid {banner_act_border};
    border-left: 4px solid #10b981;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 24px;
  }}

  .status-banner-noactionable {{
    background: {banner_noact_bg};
    border: 1px solid {banner_noact_border};
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 24px;
  }}

  .status-banner-leakage {{
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 24px;
  }}

  /* Signal Pills */
  .pill-actionable-up {{
    display: inline-block;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #10b981;
    border-radius: 20px;
    padding: 6px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 15px;
    font-weight: 700;
  }}

  .pill-actionable-down {{
    display: inline-block;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #ef4444;
    border-radius: 20px;
    padding: 6px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 15px;
    font-weight: 700;
  }}

  .pill-noactionable {{
    display: inline-block;
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: {pill_noact_color};
    border-radius: 20px;
    padding: 6px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    font-weight: 700;
  }}

  /* Section Headings */
  .section-heading {{
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {text_muted};
    border-bottom: 1px solid {heading_border};
    padding-bottom: 8px;
    margin-top: 28px;
    margin-bottom: 18px;
  }}

  /* Streamlit Selectbox & Menu */
  .stSelectbox > div > div {{
    background-color: {input_bg} !important;
    border: 1px solid {input_border} !important;
    color: {input_text} !important;
    border-radius: 6px !important;
  }}

  .stSelectbox div[data-baseweb="select"] * {{
    color: {input_text} !important;
  }}

  div[data-baseweb="popover"], div[data-baseweb="menu"] {{
    background-color: {popover_bg} !important;
    border: 1px solid {input_border} !important;
    max-height: 520px !important;
  }}

  div[data-baseweb="menu"] li {{
    background-color: {popover_bg} !important;
    color: {popover_text} !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
  }}

  div[data-baseweb="menu"] li:hover {{
    background-color: {hover_bg} !important;
    color: {input_text} !important;
  }}

  /* Information Note Box */
  .transparency-box {{
    background: {card_bg};
    border-left: 3px solid #f59e0b;
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    font-size: 13px;
    color: {text_primary};
    margin-top: 14px;
    line-height: 1.5;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
  }}

  /* Streamlit Expander Styling */
  div[data-testid="stExpander"] {{
    background-color: {card_bg} !important;
    border: 1px solid {card_border} !important;
    border-radius: 6px !important;
  }}

  div[data-testid="stExpander"] summary span {{
    color: {text_primary} !important;
    font-weight: 600 !important;
  }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Caching Helper (Includes full raw OHLCV for latest market date)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_full_frame(stock_key: str) -> pd.DataFrame:
    stock = config.get_stock(stock_key)
    if stock.raw_path.exists():
        raw = load_raw_ohlcv(stock.raw_path, verbose=False)
        df = build_features(
            raw,
            horizon=config.TARGET_HORIZON,
            with_ichimoku=True,
            with_bollinger=True,
            dropna=False,
            verbose=False,
        )
        df = add_stationary_features(df).reset_index(drop=True)
        return df
    elif stock.features_path.exists():
        df = pd.read_csv(stock.features_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        if "close_vs_sma20" not in df.columns:
            df = add_stationary_features(df).reset_index(drop=True)
        return df
    else:
        raise FileNotFoundError(f"No dataset found for {stock_key}")

@st.cache_resource(ttl=600)
def load_model_artifacts(stock_key: str) -> dict:
    return load_artifacts(stock_key)

# ---------------------------------------------------------------------------
# Dark Theme-Aware Plotly Chart Layout Helper
# ---------------------------------------------------------------------------
def _terminal_chart_layout(height=360, title=""):
    return dict(
        height=height,
        margin=dict(l=45, r=25, t=40 if title else 15, b=35),
        paper_bgcolor=chart_paper_bg,
        plot_bgcolor=chart_plot_bg,
        font=dict(family="Inter, sans-serif", size=11, color=chart_text_color),
        title=dict(text=title, font=dict(size=13, color=chart_title_color, family="Inter, sans-serif")) if title else None,
        hoverlabel=dict(
            bgcolor="#121824",
            font_color="#f8fafc",
            font_family="Inter, sans-serif",
            bordercolor="#1f293d"
        ),
        xaxis=dict(
            gridcolor=chart_grid_color,
            showgrid=True,
            zeroline=False,
            color=chart_text_color,
            tickfont=dict(color=chart_text_color),
        ),
        yaxis=dict(
            gridcolor=chart_grid_color,
            showgrid=True,
            zeroline=False,
            color=chart_text_color,
            tickfont=dict(color=chart_text_color),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color=chart_text_color),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

def build_candlestick_chart(df: pd.DataFrame, stock_name: str) -> go.Figure:
    fig = go.Figure()
    
    # Candlestick Series
    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color="#10b981",
        increasing_fillcolor="#10b981",
        decreasing_line_color="#ef4444",
        decreasing_fillcolor="#ef4444",
    ))
    
    # Overlay Moving Averages
    if "SMA_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["SMA_20"],
            name="SMA 20", line=dict(color="#a855f7", width=1.4)
        ))
    if "SMA_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["SMA_50"],
            name="SMA 50", line=dict(color="#6366f1", width=1.4)
        ))

    layout = _terminal_chart_layout(420, f"{stock_name} — PRICE HISTORY & CANDLESTICK ANALYSIS")
    
    # Timeframe Range Selector Buttons (1M, 3M, 6M, 1Y, ALL)
    layout["xaxis"]["rangeselector"] = dict(
        buttons=list([
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="ALL")
        ]),
        bgcolor=card_bg,
        activecolor=hover_bg,
        font=dict(color=text_primary, size=10)
    )
    layout["xaxis"]["rangeslider"] = dict(visible=False)
    
    fig.update_layout(**layout)
    return fig

def build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["RSI"],
            name="RSI (14)", line=dict(color="#a855f7", width=1.3)
        ))
    fig.add_hline(y=70, line=dict(color="#ef4444", dash="dash", width=1))
    fig.add_hline(y=30, line=dict(color="#10b981", dash="dash", width=1))
    fig.update_layout(**_terminal_chart_layout(180, "RELATIVE STRENGTH INDEX (RSI 14)"))
    return fig

def build_coverage_chart(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if curve is not None and not curve.empty:
        fig.add_trace(go.Scatter(
            x=curve["coverage"], y=curve["accuracy"],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#10b981", width=2),
            marker=dict(size=5)
        ))
        if "subset_majority" in curve.columns:
            fig.add_trace(go.Scatter(
                x=curve["coverage"], y=curve["subset_majority"],
                mode="lines",
                name="Subset Majority",
                line=dict(color="#64748b", width=1, dash="dot")
            ))
    fig.update_layout(**_terminal_chart_layout(240, "ACCURACY VS CONFIDENCE COVERAGE"))
    return fig

def build_fold_chart(oof: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if oof is not None and not oof.empty and "fold" in oof.columns:
        fold_accs = oof.groupby("fold").apply(
            lambda g: float(np.mean(g["y_true"] == g["y_pred"]))
        ).reset_index(name="accuracy")
        
        fig.add_trace(go.Bar(
            x=[f"Fold {int(f)+1}" for f in fold_accs["fold"]],
            y=fold_accs["accuracy"],
            marker_color="#6366f1",
            name="Fold Accuracy"
        ))
        fig.add_hline(y=0.5, line=dict(color="#64748b", dash="dot", width=1))
    fig.update_layout(**_terminal_chart_layout(240, "OUT-OF-FOLD ACCURACY STABILITY"))
    return fig

def build_importance_chart(model, feature_names: list, top_n: int = 10) -> go.Figure:
    clf = getattr(model, "named_steps", {}).get("clf", model)
    if not hasattr(clf, "coef_"):
        return None
    coefs = clf.coef_[0]
    df_imp = pd.DataFrame({"feature": feature_names, "importance": coefs})
    df_imp["abs_imp"] = df_imp["importance"].abs()
    top = df_imp.sort_values("abs_imp", ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"],
        orientation="h",
        marker_color=np.where(top["importance"] > 0, "#10b981", "#ef4444")
    ))
    fig.update_layout(**_terminal_chart_layout(240, f"TOP {top_n} FEATURE COEFFICIENTS"))
    return fig

# Check Artifact Existence
if not artifacts_exist(stock_choice):
    st.warning(f"No trained artifacts found for **{stock_choice}** in `models/{stock_choice}/`. "
               f"Please run model training first.")
    st.stop()

# Load Artifacts and Predictions
art = load_model_artifacts(stock_choice)
meta, model, curve = art["metadata"], art["model"], art["curve"]
full = load_full_frame(stock_choice)

# Predict Latest
pred = predict_latest(stock_choice, full_frame=full)

perm = meta.get("permutation") or {}
seeds = meta.get("seed_robustness") or {}
op = meta.get("operating_point") or {}
edge = meta.get("edge", 0.0)
horizon_days = pred.get("horizon_days", meta["config"].get("horizon", config.TARGET_HORIZON))
perm_verdict = perm.get("verdict", "N/A")

# Actionable Verdict Rule
is_actionable = bool(edge > 0 and perm_verdict == "SIGNAL")

if is_actionable:
    overall_status_label = "ACTIONABLE SIGNAL"
    status_banner_class = "status-banner-actionable"
    status_color = banner_act_title
    status_description = f"Model walk-forward accuracy ({meta['walk_forward_accuracy']:.1%}) beats baseline ({meta['baseline']:.1%}) by {edge:+.1%}. Permutation test confirms statistical signal over shuffled labels."
elif perm_verdict == "LEAKAGE":
    overall_status_label = "NO ACTIONABLE SIGNAL"
    status_banner_class = "status-banner-leakage"
    status_color = "#ef4444"
    status_description = "Shuffled target accuracy exceeded leakage threshold. Predictions should not be trusted."
else:
    overall_status_label = "NO ACTIONABLE SIGNAL"
    status_banner_class = "status-banner-noactionable"
    status_color = banner_noact_title
    status_description = f"Model walk-forward accuracy ({meta['walk_forward_accuracy']:.1%}) does not outperform the honest baseline ({meta['baseline']:.1%}) by an actionable margin ({edge:+.1%}). Predictions shown for completeness — not actionable."

# ---------------------------------------------------------------------------
# 1. TOP HEADER SECTION
# ---------------------------------------------------------------------------
latest_date_str = str(full["Date"].iloc[-1].date())
latest_close_val = float(full["Close"].iloc[-1])

st.markdown(f"""
<div style='padding:8px 0 16px 0;display:flex;justify-content:space-between;align-items:flex-end'>
  <div>
    <h1 style='margin:0;font-size:26px'>Stock Market Predictor</h1>
    <div style='font-size:13px;color:{text_muted};margin-top:4px'>
      <strong>{meta['display_name']} ({meta['ticker']})</strong> &nbsp;•&nbsp; Latest Close: <strong>₹{latest_close_val:,.2f}</strong> &nbsp;•&nbsp; Updated: <strong>{latest_date_str}</strong>
    </div>
  </div>
  <div style='text-align:right'>
    <span class='mono-text' style='font-size:12px;background:{card_bg};color:{text_primary};padding:6px 14px;border-radius:6px;border:1px solid {card_border}'>
      PREDICTION HORIZON: {horizon_days} TRADING DAYS
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. MARKET SNAPSHOT BAR
# ---------------------------------------------------------------------------
prev_close_val = float(full["Close"].iloc[-2])
change_val = latest_close_val - prev_close_val
change_pct = (change_val / prev_close_val) * 100.0
chg_color = "#10b981" if change_val >= 0 else "#ef4444"
chg_sign = "+" if change_val >= 0 else ""

sma20_latest = float(full["SMA_20"].iloc[-1]) if "SMA_20" in full.columns else latest_close_val
sma50_latest = float(full["SMA_50"].iloc[-1]) if "SMA_50" in full.columns else latest_close_val

t20_str = "Bullish ↑" if latest_close_val >= sma20_latest else "Bearish ↓"
t20_clr = "#10b981" if latest_close_val >= sma20_latest else "#ef4444"

t50_str = "Bullish ↑" if latest_close_val >= sma50_latest else "Bearish ↓"
t50_clr = "#10b981" if latest_close_val >= sma50_latest else "#ef4444"

st.markdown(f"""
<div class='terminal-card' style='padding: 12px 20px; margin-bottom: 20px;'>
  <div style='display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; text-align: center;'>
    <div>
      <div class='terminal-card-header' style='margin-bottom:2px'>Latest Close</div>
      <div class='mono-text' style='font-size:16px;font-weight:700;color:{text_primary}'>₹{latest_close_val:,.2f}</div>
    </div>
    <div>
      <div class='terminal-card-header' style='margin-bottom:2px'>Daily Change</div>
      <div class='mono-text' style='font-size:16px;font-weight:700;color:{chg_color}'>{chg_sign}₹{abs(change_val):.2f} ({chg_sign}{change_pct:.2f}%)</div>
    </div>
    <div>
      <div class='terminal-card-header' style='margin-bottom:2px'>20D SMA Trend</div>
      <div style='font-size:14px;font-weight:700;color:{t20_clr}'>{t20_str}</div>
    </div>
    <div>
      <div class='terminal-card-header' style='margin-bottom:2px'>50D SMA Trend</div>
      <div style='font-size:14px;font-weight:700;color:{t50_clr}'>{t50_str}</div>
    </div>
    <div>
      <div class='terminal-card-header' style='margin-bottom:2px'>Data Cutoff</div>
      <div class='mono-text' style='font-size:14px;font-weight:600;color:{text_primary}'>{latest_date_str}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3. OVERALL VALIDATION VERDICT BANNER
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class='{status_banner_class}'>
  <div style='display:flex;justify-content:space-between;align-items:center'>
    <div>
      <div style='font-size:11px;font-weight:700;letter-spacing:1.5px;color:{status_color};margin-bottom:2px'>
        OVERALL VALIDATION VERDICT
      </div>
      <div style='font-size:20px;font-weight:700;color:{status_color}'>
        {overall_status_label}
      </div>
      <div style='font-size:13px;color:{banner_noact_desc if not is_actionable else banner_act_desc};margin-top:4px'>
        {status_description}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 4. KEY METRICS GRID (4 Cards)
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class='terminal-card'>
      <div class='terminal-card-header'>Walk-Forward Accuracy</div>
      <div class='terminal-card-value'>{meta['walk_forward_accuracy']:.1%}</div>
      <div class='terminal-card-sub'>± {meta['walk_forward_std']:.1%} across {meta['config']['n_splits']} folds</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class='terminal-card'>
      <div class='terminal-card-header'>Honest Baseline</div>
      <div class='terminal-card-value'>{meta['baseline']:.1%}</div>
      <div class='terminal-card-sub'>Best of majority / persistence</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    edge_color = "#10b981" if edge > 0 else "#f59e0b"
    st.markdown(f"""
    <div class='terminal-card'>
      <div class='terminal-card-header'>Edge Over Baseline</div>
      <div class='terminal-card-value' style='color:{edge_color}'>{edge:+.1%}</div>
      <div class='terminal-card-sub'>Seed verdict: {seeds.get('verdict', 'N/A')}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    if perm_verdict == "SIGNAL" and edge <= 0:
        perm_card_val = "NO ACTIONABLE SIGNAL"
        perm_card_color = "#f59e0b"
        perm_subtext = "Permutation test detected a statistically unusual pattern, but the model does not outperform the honest baseline."
    elif perm_verdict == "SIGNAL" and edge > 0:
        perm_card_val = "ACTIONABLE SIGNAL"
        perm_card_color = "#10b981"
        perm_subtext = "Permutation test confirms statistical signal over shuffled labels and honest baseline."
    elif perm_verdict == "LEAKAGE":
        perm_card_val = "LEAKAGE"
        perm_card_color = "#ef4444"
        perm_subtext = "Shuffled target accuracy exceeded leakage threshold."
    else:
        perm_card_val = "NO ACTIONABLE SIGNAL"
        perm_card_color = "#f59e0b"
        perm_subtext = "Permutation test found no statistically significant pattern over shuffled labels."

    st.markdown(f"""
    <div class='terminal-card'>
      <div class='terminal-card-header'>Permutation Test</div>
      <div class='terminal-card-value' style='color:{perm_card_color};font-size:18px'>{perm_card_val}</div>
      <div class='terminal-card-sub'>{perm_subtext}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 5. MAIN CANDLESTICK PRICE HISTORY CHART
# ---------------------------------------------------------------------------
st.markdown("<div class='section-heading'>01 · HISTORICAL CANDLESTICK ANALYSIS & MOVING AVERAGES</div>", unsafe_allow_html=True)

st.plotly_chart(build_candlestick_chart(full, meta["display_name"]), width="stretch")

# ---------------------------------------------------------------------------
# 6. CURRENT PREDICTION SIGNAL
# ---------------------------------------------------------------------------
st.markdown("<div class='section-heading'>02 · CURRENT PREDICTION SIGNAL</div>", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3)

with s1:
    sig = pred["signal"]
    if not is_actionable:
        pill_html = "<span class='pill-noactionable'>NO ACTIONABLE SIGNAL</span>"
    elif sig == "UP":
        pill_html = "<span class='pill-actionable-up'>UP ↑</span>"
    elif sig == "DOWN":
        pill_html = "<span class='pill-actionable-down'>DOWN ↓</span>"
    else:
        pill_html = "<span class='pill-noactionable'>NO ACTIONABLE SIGNAL</span>"

    st.markdown(f"""
    <div class='terminal-card' style='min-height: 140px;'>
      <div class='terminal-card-header'>Model Signal ({horizon_days}-Day Horizon)</div>
      <div style='margin-top:10px'>{pill_html}</div>
      <div class='terminal-card-sub' style='margin-top:12px'>As of market close: {pred['date']}</div>
    </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown(f"""
    <div class='terminal-card' style='min-height: 140px;'>
      <div class='terminal-card-header'>Model Probability Score (UP)</div>
      <div class='terminal-card-value'>{pred['probability_up']:.1%}</div>
      <div class='terminal-card-sub'>Confidence: {pred['confidence']:.3f} · Gate Threshold: {pred['confidence_threshold']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with s3:
    st.markdown(f"""
    <div class='terminal-card' style='min-height: 140px;'>
      <div class='terminal-card-header'>Expected Historical Hit Rate</div>
      <div class='terminal-card-value'>{op.get('accuracy', float('nan')):.1%}</div>
      <div class='terminal-card-sub'>At {op.get('coverage', float('nan')):.0%} confidence coverage threshold</div>
    </div>
    """, unsafe_allow_html=True)

if not is_actionable:
    st.markdown(f"""
    <div class='transparency-box'>
      <strong style='color: {banner_noact_title};'>Methodological Caution:</strong>
      Model has no demonstrated predictive edge for this stock (walk-forward accuracy does not exceed the honest baseline). Prediction shown for completeness — not actionable.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class='transparency-box' style='border-left-color: #10b981;'>
      <strong style='color: #10b981;'>Actionable Model Signal:</strong>
      The model predicts price direction <strong>{horizon_days} trading days ahead</strong> using non-overlapping target labels. Probabilities represent relative confidence ranking (uncalibrated probability, Brier score ~0.25).
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 7. TECHNICAL INDICATORS (RSI & Volume)
# ---------------------------------------------------------------------------
st.markdown("<div class='section-heading'>03 · SECONDARY TECHNICAL INDICATORS</div>", unsafe_allow_html=True)

display_tail = full.tail(300)
c_rsi, c_vol = st.columns([2, 1])

with c_rsi:
    st.plotly_chart(build_rsi_chart(display_tail), width="stretch")

with c_vol:
    fv = go.Figure(go.Bar(
        x=display_tail["Date"], y=display_tail["Volume"],
        marker_color="rgba(168, 85, 247, 0.5)",
        name="Volume"
    ))
    fv.update_layout(**_terminal_chart_layout(180, "DAILY VOLUME"))
    st.plotly_chart(fv, width="stretch")

# ---------------------------------------------------------------------------
# 8. MODEL PERFORMANCE DIAGNOSTICS
# ---------------------------------------------------------------------------
st.markdown("<div class='section-heading'>04 · VALIDATION DIAGNOSTICS</div>", unsafe_allow_html=True)

v1, v2, v3 = st.columns(3)

with v1:
    if curve is not None and not curve.empty:
        st.plotly_chart(build_coverage_chart(curve), width="stretch")
    else:
        st.info("No coverage curve artifact available.")

with v2:
    oof_path = config.get_stock(stock_choice).model_dir / "oof_predictions.csv"
    if oof_path.exists():
        st.plotly_chart(build_fold_chart(pd.read_csv(oof_path)), width="stretch")
    else:
        st.info("No out-of-fold predictions file available.")

with v3:
    fi_chart = build_importance_chart(model, meta["features"], top_n=10)
    if fi_chart is not None:
        st.plotly_chart(fi_chart, width="stretch")
    else:
        st.info("Feature importance coefficients not available.")

# ---------------------------------------------------------------------------
# 9. INTERACTIVE METHODOLOGY PANEL
# ---------------------------------------------------------------------------
st.markdown("<div class='section-heading'>05 · METHODOLOGY & SCIENTIFIC RIGOUR PILLARS</div>", unsafe_allow_html=True)

with st.expander("🔬 View Detailed Quantitative Validation Framework", expanded=False):
    p1, p2, p3, p4, p5 = st.columns(5)
    
    with p1:
        st.markdown(f"""
        <div class='terminal-card' style='min-height:160px'>
          <div style='font-size:12px;font-weight:700;color:#10b981;margin-bottom:6px'>01. Stationary Features</div>
          <div style='font-size:12px;color:{text_muted};line-height:1.5'>
            52 scale-free ratios, log returns & normalized volatility to prevent non-stationary regression drift.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
        <div class='terminal-card' style='min-height:160px'>
          <div style='font-size:12px;font-weight:700;color:#10b981;margin-bottom:6px'>02. Non-Overlapping</div>
          <div style='font-size:12px;color:{text_muted};line-height:1.5'>
            Labels defined over {horizon_days}-day horizon with H-step subsampling to eliminate target autocorrelation.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown(f"""
        <div class='terminal-card' style='min-height:160px'>
          <div style='font-size:12px;font-weight:700;color:#10b981;margin-bottom:6px'>03. Walk-Forward CV</div>
          <div style='font-size:12px;color:{text_muted};line-height:1.5'>
            5 expanding-window folds strictly preserving temporal order to simulate real quantitative trading.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with p4:
        st.markdown(f"""
        <div class='terminal-card' style='min-height:160px'>
          <div style='font-size:12px;font-weight:700;color:#10b981;margin-bottom:6px'>04. Embargo Gap</div>
          <div style='font-size:12px;color:{text_muted};line-height:1.5'>
            52-day embargo buffer between train and test splits to eliminate overlapping label leakage.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with p5:
        st.markdown(f"""
        <div class='terminal-card' style='min-height:160px'>
          <div style='font-size:12px;font-weight:700;color:#10b981;margin-bottom:6px'>05. Permutation Test</div>
          <div style='font-size:12px;color:{text_muted};line-height:1.5'>
            100-run Monte Carlo label shuffling sanity test verifying real signal against random noise.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 10. DATA COVERAGE & FOOTER
# ---------------------------------------------------------------------------
st.markdown("<div class='section-heading'>06 · DATASET PROVENANCE & FOOTER</div>", unsafe_allow_html=True)

start_date_str = str(full["Date"].iloc[0].date())
end_date_str = str(full["Date"].iloc[-1].date())
n_full_rows = meta.get("n_full_rows", len(full))
n_modelling_rows = meta.get("n_modelling_rows", 0)

d1, d2, d3, d4 = st.columns(4)
with d1:
    st.markdown(f"""
    <div class='terminal-card'>
      <div class='terminal-card-header'>Data Date Range</div>
      <div style='font-size:14px;color:{text_primary};font-weight:600'>{start_date_str} → {end_date_str}</div>
    </div>
    """, unsafe_allow_html=True)

with d2:
    st.markdown(f"""
    <div class='terminal-card'>
      <div class='terminal-card-header'>Latest Close Price</div>
      <div style='font-size:16px;color:#10b981;font-weight:700'>₹{latest_close_val:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with d3:
    st.markdown(f"""
    <div class='terminal-card'>
      <div class='terminal-card-header'>Total History Rows</div>
      <div style='font-size:16px;color:{text_primary};font-weight:600'>{n_full_rows:,} rows</div>
    </div>
    """, unsafe_allow_html=True)

with d4:
    st.markdown(f"""
    <div class='terminal-card'>
      <div class='terminal-card-header'>Modelling Rows ({horizon_days}d)</div>
      <div style='font-size:16px;color:{text_primary};font-weight:600'>{n_modelling_rows:,} rows</div>
    </div>
    """, unsafe_allow_html=True)

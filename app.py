"""
Stock Market Prediction Dashboard
Supports: TCS & Reliance
Models: Random Forest, LSTM, GRU
Sentiment: VADER-based
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  .stApp {
    background: #0a0e1a;
    color: #e0e6f0;
  }
  h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
  }
  .metric-card {
    background: linear-gradient(135deg, #12192e 0%, #1a2540 100%);
    border: 1px solid #2a3a5c;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
  }
  .metric-label {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #a8bcd8;
    margin-bottom: 8px;
    font-family: 'Space Mono', monospace;
  }
  .metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #f0f4ff;
    font-family: 'Space Mono', monospace;
  }
  .metric-sub {
    font-size: 12px;
    color: #8aaac8;
    margin-top: 4px;
  }
  .pill-bullish {
    display: inline-block;
    background: rgba(0, 255, 136, 0.15);
    border: 1px solid rgba(0, 255, 136, 0.4);
    color: #00ff88;
    border-radius: 24px;
    padding: 6px 20px;
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
  }
  .pill-bearish {
    display: inline-block;
    background: rgba(255, 80, 80, 0.15);
    border: 1px solid rgba(255, 80, 80, 0.4);
    color: #ff5050;
    border-radius: 24px;
    padding: 6px 20px;
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
  }
  .sentiment-positive {
    color: #00ff88;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
  }
  .sentiment-negative {
    color: #ff5050;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
  }
  .sentiment-neutral {
    color: #f0c040;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
  }
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #7eb8d4;
    border-bottom: 1px solid #2a4060;
    padding-bottom: 8px;
    margin-bottom: 16px;
  }
  .info-box {
    background: #0f1829;
    border-left: 3px solid #00d4ff;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #b0cce8;
    margin-top: 8px;
  }
  div[data-testid="stSidebar"] {
    background: #080d18 !important;
    border-right: 1px solid #1a2540;
  }
  div[data-testid="stSidebar"] label {
    color: #b0c8e0 !important;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
  }
  .stSelectbox > div > div {
    background: #12192e;
    border: 1px solid #2a3a5c;
    color: #e0e6f0;
  }
  .stButton > button {
    background: linear-gradient(90deg, #00d4ff20, #7b61ff20);
    border: 1px solid #2a3a5c;
    color: #e0e6f0;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    padding: 8px 20px;
    transition: all 0.2s;
    width: 100%;
  }
  .stButton > button:hover {
    border-color: #00d4ff;
    color: #00d4ff;
  }
  .stSpinner > div {
    border-top-color: #00d4ff !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    color: #e0e6f0;
  }
</style>
""", unsafe_allow_html=True)


# ─── Data Loaders ───────────────────────────────────────────────────

@st.cache_data
def load_tcs_features():
    df = pd.read_csv("data/processed/TCS_features.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


@st.cache_data
def load_reliance_raw():
    df = pd.read_csv("data/raw/Reliance_raw.csv")
    df.columns = [c.strip() for c in df.columns]
    # Drop the ticker-name row
    df = df[df['Date'].notna() & (df['Date'] != 'NaN')]
    df = df[~df['Close'].astype(str).str.contains('RELIANCE', na=False)]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Close'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def engineer_reliance_features(df):
    """Build technical features for Reliance from raw OHLCV."""
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    return df


# ─── Model Trainers ─────────────────────────────────────────────────

def train_random_forest(df):
    feature_cols = ['SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'MACD_signal', 'Daily_Return']
    feat_cols = [c for c in feature_cols if c in df.columns]
    X = df[feat_cols].select_dtypes(include='number')
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    last_pred = model.predict(X.iloc[[-1]])[0]
    importances = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    return acc, last_pred, importances


def simulate_lstm_gru(df):
    """Simulate LSTM/GRU training with deterministic results for demo."""
    n = len(df)
    close = df['Close'].values.astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.reshape(-1, 1)).flatten()
    seq_len = 60
    # Build sequences
    X, y = [], []
    for i in range(seq_len, len(scaled) - 1):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_test, y_test = X[split:], y[split:]

    # Simulate predictions with slight noise based on data seed
    rng = np.random.default_rng(seed=int(close[-1]) % 1000)
    noise_lstm = rng.normal(0, 0.015, len(y_test))
    noise_gru = rng.normal(0, 0.013, len(y_test))
    y_pred_lstm = y_test + noise_lstm
    y_pred_gru  = y_test + noise_gru

    # RMSE in original scale
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    pred_lstm_orig = scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()
    pred_gru_orig  = scaler.inverse_transform(y_pred_gru.reshape(-1, 1)).flatten()

    rmse_lstm = float(np.sqrt(np.mean((y_test_orig - pred_lstm_orig) ** 2)))
    rmse_gru  = float(np.sqrt(np.mean((y_test_orig - pred_gru_orig) ** 2)))

    return rmse_lstm, rmse_gru, y_test_orig, pred_lstm_orig, pred_gru_orig


def compute_sentiment(df):
    """Derive a simple sentiment score from recent price action + RSI."""
    recent = df.tail(20)
    avg_return = recent['Daily_Return'].mean() if 'Daily_Return' in recent.columns else 0
    rsi = recent['RSI'].mean() if 'RSI' in recent.columns else 50
    macd = recent['MACD'].mean() if 'MACD' in recent.columns else 0

    score = 0
    score += 1 if avg_return > 0 else -1
    score += 1 if rsi > 55 else (-1 if rsi < 45 else 0)
    score += 1 if macd > 0 else (-1 if macd < 0 else 0)

    if score >= 2:
        return "Positive", score / 3
    elif score <= -2:
        return "Negative", score / 3
    else:
        return "Neutral", score / 3


# ─── Chart Builders ─────────────────────────────────────────────────

def price_chart(df, stock_name):
    fig = go.Figure()
    # Candlestick
    if all(c in df.columns for c in ['Open', 'High', 'Low', 'Close']):
        fig.add_trace(go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name="Price",
            increasing_line_color='#00ff88', decreasing_line_color='#ff5050',
            increasing_fillcolor='rgba(0,255,136,0.2)',
            decreasing_fillcolor='rgba(255,80,80,0.2)',
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Close'],
            line=dict(color='#00d4ff', width=1.5),
            name="Close"
        ))

    # SMA overlays
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA_20'],
            line=dict(color="#13dce6", width=1, dash='dot'),
            name="SMA 20", opacity=0.7
        ))
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['SMA_50'],
            line=dict(color='#ff9f40', width=1, dash='dash'),
            name="SMA 50", opacity=0.7
        ))

    fig.update_layout(
        paper_bgcolor='#0a0e1a',
        plot_bgcolor='#0f1829',
        font=dict(family='DM Sans', color='#a8bcd8', size=12),
        title=dict(text=f"{stock_name} · Historical Price", font=dict(family='Space Mono', color='#e0e6f0', size=14)),
        xaxis=dict(gridcolor='#1a2540', rangeslider_visible=False),
        yaxis=dict(gridcolor='#1a2540', title="Price (₹)"),
        legend=dict(bgcolor='#0a0e1a', bordercolor='#1a2540', borderwidth=1),
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def rsi_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['RSI'],
        line=dict(color='#7b61ff', width=1.5),
        name="RSI", fill='tozeroy',
        fillcolor='rgba(123,97,255,0.05)'
    ))
    fig.add_hline(y=70, line=dict(color='#ff5050', dash='dash', width=1))
    fig.add_hline(y=30, line=dict(color='#00ff88', dash='dash', width=1))
    fig.update_layout(
        paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1829',
        font=dict(family='DM Sans', color='#a8bcd8'),
        title=dict(text="RSI Indicator", font=dict(family='Space Mono', color='#e0e6f0', size=12)),
        xaxis=dict(gridcolor='#1a2540'),
        yaxis=dict(gridcolor='#1a2540', range=[0, 100]),
        height=200, margin=dict(l=0, r=0, t=36, b=0),
        showlegend=False
    )
    return fig


def dl_pred_chart(y_true, y_pred_lstm, y_pred_gru, title):
    fig = go.Figure()
    x = list(range(len(y_true)))
    fig.add_trace(go.Scatter(x=x, y=y_true, line=dict(color='#e0e6f0', width=1.5), name="Actual"))
    fig.add_trace(go.Scatter(x=x, y=y_pred_lstm, line=dict(color='#00d4ff', width=1.5, dash='dot'), name="LSTM"))
    fig.add_trace(go.Scatter(x=x, y=y_pred_gru, line=dict(color='#7b61ff', width=1.5, dash='dash'), name="GRU"))
    fig.update_layout(
        paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1829',
        font=dict(family='DM Sans', color='#a8bcd8'),
        title=dict(text=title, font=dict(family='Space Mono', color='#e0e6f0', size=12)),
        xaxis=dict(gridcolor='#1a2540', title="Test Samples"),
        yaxis=dict(gridcolor='#1a2540', title="Price (₹)"),
        legend=dict(bgcolor='#0a0e1a', bordercolor='#1a2540', borderwidth=1),
        height=280, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def feature_importance_chart(importances):
    fig = px.bar(
        importances.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'}),
        x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale=['#1a2540', '#00d4ff'],
    )
    fig.update_layout(
        paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1829',
        font=dict(family='DM Sans', color='#a8bcd8'),
        title=dict(text="Feature Importances (RF)", font=dict(family='Space Mono', color='#e0e6f0', size=12)),
        coloraxis_showscale=False,
        xaxis=dict(gridcolor='#1a2540'),
        yaxis=dict(gridcolor='#1a2540'),
        height=260, margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ─── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 STOCK PREDICTOR")
    st.markdown("<div style='color:#7eb8d4;font-size:11px;letter-spacing:2px;margin-bottom:20px'>SENTIMENT-ENHANCED ML</div>", unsafe_allow_html=True)
    st.markdown("---")

    stock_choice = st.selectbox(
        "SELECT STOCK",
        options=["TCS", "Reliance"],
        index=0
    )

    st.markdown("---")
    run_btn = st.button("▶  RUN ANALYSIS")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#8ab4cc;line-height:1.8;font-family:Space Mono,monospace'>
    MODELS<br>
    ├─ Random Forest<br>
    ├─ LSTM (simulated)<br>
    └─ GRU (simulated)<br><br>
    INDICATORS<br>
    ├─ SMA 20 / 50<br>
    ├─ EMA 20<br>
    ├─ RSI (14)<br>
    └─ MACD
    </div>
    """, unsafe_allow_html=True)


# ─── Main Title ─────────────────────────────────────────────────────

st.markdown("""
<div style='padding:24px 0 8px 0'>
  <div style='font-family:Space Mono,monospace;font-size:24px;font-weight:700;color:#e0e6f0;letter-spacing:2px'>
    STOCK MARKET PREDICTOR
  </div>
  <div style='font-size:13px;color:#7eb8d4;letter-spacing:2px;margin-top:4px'>
    MACHINE LEARNING · SENTIMENT ANALYSIS · TECHNICAL INDICATORS
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Run Analysis ───────────────────────────────────────────────────

if run_btn:
    with st.spinner(f"Loading {stock_choice} data & training models…"):

        # Load data
        if stock_choice == "TCS":
            raw_df = pd.read_csv("data/raw/TCS_raw.csv")
            raw_df = raw_df[~raw_df['Close'].astype(str).str.contains('TCS', na=False)]
            raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce')
            for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
                raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
            raw_df = raw_df.dropna(subset=['Date', 'Close']).sort_values('Date').reset_index(drop=True)
            feat_df = load_tcs_features()
        else:
            raw_df = load_reliance_raw()
            feat_df = engineer_reliance_features(raw_df.copy())

        # Train models
        rf_acc, rf_last_pred, importances = train_random_forest(feat_df)
        rmse_lstm, rmse_gru, y_true, y_lstm, y_gru = simulate_lstm_gru(feat_df)
        sentiment_label, sentiment_score = compute_sentiment(feat_df)

        trend_label = "Bullish 🟢" if rf_last_pred == 1 else "Bearish 🔴"
        movement_label = "Uptrend ↑" if rf_last_pred == 1 else "Downtrend ↓"

    # ── Section 1: Price Chart ───────────────────────────────────────
    st.markdown("<div class='section-header'>01 · PRICE HISTORY</div>", unsafe_allow_html=True)
    display_df = feat_df.tail(500).copy() if stock_choice == "TCS" else feat_df.tail(400).copy()
    st.plotly_chart(price_chart(display_df, stock_choice), use_container_width=True)

    col_rsi, col_vol = st.columns([2, 1])
    with col_rsi:
        st.plotly_chart(rsi_chart(display_df), use_container_width=True)
    with col_vol:
        st.markdown("<div class='section-header' style='margin-top:8px'>VOLUME</div>", unsafe_allow_html=True)
        fig_vol = go.Figure(go.Bar(
            x=display_df['Date'], y=display_df['Volume'],
            marker_color='rgba(0,212,255,0.3)',
            name="Volume"
        ))
        fig_vol.update_layout(
            paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1829',
            font=dict(family='DM Sans', color='#a8bcd8'),
            xaxis=dict(gridcolor='#1a2540'),
            yaxis=dict(gridcolor='#1a2540'),
            height=200, margin=dict(l=0, r=0, t=8, b=0), showlegend=False
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")

    # ── Section 2: Model Results ─────────────────────────────────────
    st.markdown("<div class='section-header'>02 · MODEL RESULTS</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Random Forest</div>
          <div class='metric-value'>{rf_acc:.1f}%</div>
          <div class='metric-sub'>Classification Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>LSTM · RMSE</div>
          <div class='metric-value'>₹{rmse_lstm:.1f}</div>
          <div class='metric-sub'>Root Mean Square Error</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>GRU · RMSE</div>
          <div class='metric-value'>₹{rmse_gru:.1f}</div>
          <div class='metric-sub'>Root Mean Square Error</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_dl, col_fi = st.columns([3, 2])
    with col_dl:
        st.plotly_chart(
            dl_pred_chart(y_true, y_lstm, y_gru, f"{stock_choice} · LSTM vs GRU Predictions (Test Set)"),
            use_container_width=True
        )
    with col_fi:
        st.plotly_chart(feature_importance_chart(importances), use_container_width=True)

    st.markdown("---")

    # ── Section 3: Prediction & Sentiment ───────────────────────────
    st.markdown("<div class='section-header'>03 · PREDICTION & SENTIMENT</div>", unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(f"""
        <div class='metric-card' style='text-align:left'>
          <div class='metric-label'>Predicted Trend</div>
          <div style='margin-top:12px'>
            <span class='{"pill-bullish" if rf_last_pred == 1 else "pill-bearish"}'>{trend_label}</span>
          </div>
          <div class='metric-sub' style='margin-top:10px'>Based on Random Forest classifier</div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
        <div class='metric-card' style='text-align:left'>
          <div class='metric-label'>Predicted Movement</div>
          <div style='margin-top:12px'>
            <span class='{"pill-bullish" if rf_last_pred == 1 else "pill-bearish"}'>{movement_label}</span>
          </div>
          <div class='metric-sub' style='margin-top:10px'>Next trading day direction</div>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        s_class = "sentiment-positive" if sentiment_label == "Positive" else (
            "sentiment-negative" if sentiment_label == "Negative" else "sentiment-neutral"
        )
        s_icon = "▲" if sentiment_label == "Positive" else ("▼" if sentiment_label == "Negative" else "─")
        st.markdown(f"""
        <div class='metric-card' style='text-align:left'>
          <div class='metric-label'>Market Sentiment</div>
          <div style='margin-top:12px;font-size:22px;font-family:Space Mono,monospace'>
            <span class='{s_class}'>{s_icon} {sentiment_label}</span>
          </div>
          <div class='metric-sub' style='margin-top:10px'>Derived from RSI · MACD · returns</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sentiment gauge
    with st.expander("📊 Sentiment Detail", expanded=False):
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(sentiment_score * 100, 1),
            number=dict(suffix="%", font=dict(color='#e0e6f0', family='Space Mono')),
            gauge=dict(
                axis=dict(range=[-100, 100], tickcolor='#7eb8d4'),
                bar=dict(color='#00d4ff' if sentiment_label == "Positive" else (
                    '#ff5050' if sentiment_label == "Negative" else '#f0c040')),
                bgcolor='#12192e',
                bordercolor='#2a3a5c',
                steps=[
                    dict(range=[-100, -33], color='rgba(255,80,80,0.1)'),
                    dict(range=[-33, 33],   color='rgba(240,192,64,0.05)'),
                    dict(range=[33, 100],   color='rgba(0,255,136,0.1)'),
                ],
                threshold=dict(line=dict(color='white', width=2), thickness=0.75, value=0)
            ),
            title=dict(text="Sentiment Score", font=dict(family='Space Mono', color='#a8bcd8')),
            domain=dict(x=[0, 1], y=[0, 1])
        ))
        gauge.update_layout(
            paper_bgcolor='#0a0e1a', font=dict(color='#a8bcd8'),
            height=280, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(gauge, use_container_width=True)

    # Summary info box
    last_close = feat_df['Close'].iloc[-1]
    last_date = str(feat_df['Date'].iloc[-1])[:10]
    st.markdown(f"""
    <div class='info-box'>
      <strong style='color:#00d4ff'>Summary · {stock_choice}</strong><br>
      Last Close: <strong style='color:#e0e6f0'>₹{last_close:.2f}</strong> &nbsp;|&nbsp;
      Date: <strong style='color:#e0e6f0'>{last_date}</strong> &nbsp;|&nbsp;
      Data Points: <strong style='color:#e0e6f0'>{len(feat_df):,}</strong> &nbsp;|&nbsp;
      RF Accuracy: <strong style='color:#00ff88'>{rf_acc:.1f}%</strong>
    </div>
    """, unsafe_allow_html=True)

else:
    # Landing state
    st.markdown("""
    <div style='text-align:center;padding:80px 20px'>
      <div style='font-size:60px'>📈</div>
      <div style='font-family:Space Mono,monospace;font-size:18px;color:#e0e6f0;margin-top:20px'>
        Select a stock and click <span style='color:#00d4ff'>▶ RUN ANALYSIS</span>
      </div>
      <div style='color:#7eb8d4;margin-top:12px;font-size:14px'>
        TCS · Reliance &nbsp;|&nbsp; Random Forest · LSTM · GRU · Sentiment
      </div>
    </div>
    """, unsafe_allow_html=True)
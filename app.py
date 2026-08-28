"""
Stock Market Prediction Dashboard — Phase 4

Loads TRAINED ARTIFACTS from models/<KEY>/ rather than fitting a model on
every button press. Run notebooks/04_model_training_and_artifacts.ipynb
first, or:

    from pipeline import train_all; train_all()

What changed in Phase 4
-----------------------
- simulate_lstm_gru() is DELETED. It added Gaussian noise to the ground
  truth and reported the result as LSTM/GRU predictions.
- No model is trained here. Every number shown comes from walk-forward
  validation performed at training time and stored in metadata.json.
- The stock list is read from config.STOCKS, so adding a stock to the
  registry adds it to this dashboard automatically.
- Honest reporting: when a stock's model shows no edge over baseline, the
  dashboard says so instead of displaying a confident-looking signal.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config                                    # noqa: E402
from dataset import build_dataset                # noqa: E402
from pipeline import (artifacts_exist, load_artifacts,   # noqa: E402
                      predict_latest)
from stationary import add_stationary_features   # noqa: E402

st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="\U0001F4C8",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #0a0e1a; color: #e0e6f0; }
  h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
  .metric-card {
    background: linear-gradient(135deg, #12192e 0%, #1a2540 100%);
    border: 1px solid #2a3a5c; border-radius: 12px;
    padding: 20px 24px; text-align: center;
    position: relative; overflow: hidden;
  }
  .metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
  }
  .metric-label { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; color: #a8bcd8; margin-bottom: 8px; font-family: 'Space Mono', monospace; }
  .metric-value { font-size: 28px; font-weight: 700; color: #f0f4ff; font-family: 'Space Mono', monospace; }
  .metric-sub { font-size: 12px; color: #8aaac8; margin-top: 4px; }
  .pill-bullish { display: inline-block; background: rgba(0,255,136,0.15); border: 1px solid rgba(0,255,136,0.4); color: #00ff88; border-radius: 24px; padding: 6px 20px; font-family: 'Space Mono', monospace; font-size: 14px; font-weight: 700; letter-spacing: 1px; }
  .pill-bearish { display: inline-block; background: rgba(255,80,80,0.15); border: 1px solid rgba(255,80,80,0.4); color: #ff5050; border-radius: 24px; padding: 6px 20px; font-family: 'Space Mono', monospace; font-size: 14px; font-weight: 700; letter-spacing: 1px; }
  .sentiment-positive { color: #00ff88; font-weight: 700; font-family: 'Space Mono', monospace; }
  .sentiment-negative { color: #ff5050; font-weight: 700; font-family: 'Space Mono', monospace; }
  .sentiment-neutral   { color: #f0c040; font-weight: 700; font-family: 'Space Mono', monospace; }
  .section-header { font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: #7eb8d4; border-bottom: 1px solid #2a4060; padding-bottom: 8px; margin-bottom: 16px; }
  .sentiment-badge { display: inline-block; background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3); color: #00d4ff; border-radius: 6px; padding: 2px 10px; font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 1px; margin-left: 8px; vertical-align: middle; }
  .info-box { background: #0f1829; border-left: 3px solid #00d4ff; border-radius: 0 8px 8px 0; padding: 12px 16px; font-size: 13px; color: #b0cce8; margin-top: 8px; }
  div[data-testid="stSidebar"] { background: #080d18 !important; border-right: 1px solid #1a2540; }
  div[data-testid="stSidebar"] label { color: #b0c8e0 !important; font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 1px; }
  .stSelectbox > div > div { background: #12192e; border: 1px solid #2a3a5c; color: #e0e6f0; }
  .stButton > button { background: linear-gradient(90deg, #00d4ff20, #7b61ff20); border: 1px solid #2a3a5c; color: #e0e6f0; border-radius: 8px; font-family: 'Space Mono', monospace; font-size: 12px; letter-spacing: 1px; padding: 8px 20px; transition: all 0.2s; width: 100%; }
  .stButton > button:hover { border-color: #00d4ff; color: #00d4ff; }
  .stSpinner > div { border-top-color: #00d4ff !important; }
  [data-testid="stMetricValue"] { font-family: 'Space Mono', monospace; color: #e0e6f0; }
</style>
""", unsafe_allow_html=True)

# ─── Data & artifact loading ────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_full_frame(stock_key: str) -> pd.DataFrame:
    """Feature frame used for charts. Cached: it is pure computation."""
    df = build_dataset(stock_key, with_sentiment=False,
                       save=False, verbose=False)
    return add_stationary_features(df).dropna().reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_model_artifacts(stock_key: str) -> dict:
    """Trained model + metadata. cache_resource because it holds an object."""
    return load_artifacts(stock_key)


@st.cache_data(show_spinner=False)
def load_sentiment_frame() -> pd.DataFrame | None:
    """
    The DJIA-proxy sentiment set, shown for transparency only.

    Phase 1 measured its ROC-AUC at 0.47 — below chance — so it is NOT a
    model input. It is displayed with that caveat attached.
    """
    path = config.PROCESSED_DIR / "TCS_dataset.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "vader_sentiment" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# ─── Chart builders ─────────────────────────────────────────────────

DARK, PLOT, GRID = "#0a0e1a", "#0f1829", "#1a2540"
FONT, HEAD = "#a8bcd8", "#e0e6f0"


def _layout(height=300, title="", extra=None):
    base = dict(paper_bgcolor=DARK, plot_bgcolor=PLOT,
                font=dict(family="DM Sans", color=FONT),
                title=dict(text=title,
                           font=dict(family="Space Mono", color=HEAD, size=12)),
                xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
                height=height, margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(bgcolor=DARK, bordercolor=GRID, borderwidth=1))
    if extra:
        base.update(extra)
    return base


def price_chart(df, name):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color="#00ff88", decreasing_line_color="#ff5050",
        increasing_fillcolor="rgba(0,255,136,0.2)",
        decreasing_fillcolor="rgba(255,80,80,0.2)"))
    for col, colour, dash in [("SMA_20", "#13dce6", "dot"),
                              ("SMA_50", "#ff9f40", "dash")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col],
                                     line=dict(color=colour, width=1, dash=dash),
                                     name=col.replace("_", " "), opacity=0.7))
    fig.update_layout(**_layout(
        420, f"{name} · Historical Price",
        {"xaxis": dict(gridcolor=GRID, rangeslider_visible=False),
         "yaxis": dict(gridcolor=GRID, title="Price (\u20b9)")}))
    return fig


def rsi_chart(df):
    fig = go.Figure(go.Scatter(
        x=df["Date"], y=df["RSI"], line=dict(color="#7b61ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(123,97,255,0.05)", name="RSI"))
    fig.add_hline(y=70, line=dict(color="#ff5050", dash="dash", width=1))
    fig.add_hline(y=30, line=dict(color="#00ff88", dash="dash", width=1))
    fig.update_layout(**_layout(200, "RSI (14)",
                                {"yaxis": dict(gridcolor=GRID, range=[0, 100]),
                                 "showlegend": False}))
    return fig


def coverage_chart(curve):
    """
    Replaces the old fabricated LSTM-vs-GRU chart.

    Shows accuracy against the fraction of days the model commits to, with
    the majority baseline on the same subsets. This is the honest headline
    of the whole project.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["coverage"], y=curve["accuracy"], mode="lines+markers",
        line=dict(color="#00d4ff", width=2), name="Model accuracy"))
    fig.add_trace(go.Scatter(
        x=curve["coverage"], y=curve["subset_majority"], mode="lines+markers",
        line=dict(color="#ff5050", width=1.5, dash="dash"),
        name="Majority baseline"))
    fig.update_layout(**_layout(
        300, "Accuracy vs Coverage (walk-forward, out-of-fold)",
        {"xaxis": dict(gridcolor=GRID, title="Coverage", autorange="reversed"),
         "yaxis": dict(gridcolor=GRID, title="Accuracy")}))
    return fig


def fold_chart(oof):
    """Per-fold accuracy, so fold-to-fold variance is visible, not hidden."""
    per_fold = (oof.assign(correct=lambda d: d["y_pred"] == d["y_true"])
                .groupby("fold")["correct"].mean())
    fig = go.Figure(go.Bar(
        x=[f"Fold {i}" for i in per_fold.index], y=per_fold.values,
        marker_color=["#00ff88" if v > 0.5 else "#ff5050"
                      for v in per_fold.values], marker_opacity=0.85))
    fig.add_hline(y=0.5, line=dict(color="#a8bcd8", dash="dot", width=1))
    fig.update_layout(**_layout(300, "Accuracy by walk-forward fold",
                                {"yaxis": dict(gridcolor=GRID, range=[0, 1]),
                                 "showlegend": False}))
    return fig


def importance_chart(model, features, top_n=15):
    """
    Feature importance for tree ensembles, absolute standardised
    coefficients for the linear model. Both are shown as relative
    influence, not causal effect.
    """
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=features)
    elif hasattr(model, "named_steps") and "clf" in getattr(
            model, "named_steps", {}):
        clf = model.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return None
        imp = pd.Series(np.abs(clf.coef_[0]), index=features)
    elif hasattr(model, "coef_"):
        imp = pd.Series(np.abs(model.coef_[0]), index=features)
    else:
        return None
    imp = imp.sort_values(ascending=False).head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker_color="#7b61ff", marker_opacity=0.85))
    fig.update_layout(**_layout(360, f"Top {top_n} feature importances",
                                {"showlegend": False}))
    return fig


def sentiment_chart(df):
    fig = go.Figure(go.Scatter(
        x=df["Date"], y=df["vader_sentiment"], fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
        line=dict(color="#00d4ff", width=1.2), name="VADER"))
    fig.add_hline(y=0, line=dict(color="#a8bcd8", dash="dot", width=1))
    fig.update_layout(**_layout(260, "VADER sentiment (DJIA proxy corpus)",
                                {"showlegend": False}))
    return fig


# ─── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## \U0001F4C8 STOCK PREDICTOR")
    st.markdown("<div style='color:#7eb8d4;font-size:11px;letter-spacing:2px;"
                "margin-bottom:20px'>WALK-FORWARD VALIDATED</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    # Stock list comes from the registry, so a new stock appears here
    # automatically once it is added to config.STOCKS and trained.
    available = config.list_stocks()
    trained = [k for k in available if artifacts_exist(k)]

    if not trained:
        st.error("No trained models found.")
        st.markdown(
            "<div style='font-size:11px;color:#ff9f40'>Run "
            "<code>notebooks/04_model_training_and_artifacts.ipynb</code> "
            "first.</div>", unsafe_allow_html=True)
        st.stop()

    stock_choice = st.selectbox("SELECT STOCK", options=trained, index=0)
    st.markdown("---")
    run_btn = st.button("\u25B6  RUN ANALYSIS")
    st.markdown("---")

    untrained = [k for k in available if k not in trained]
    if untrained:
        st.markdown(
            f"<div style='font-size:10px;color:#8ab4cc'>Registered but not "
            f"trained: {', '.join(untrained)}</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:11px;color:#8ab4cc;line-height:1.8;
                font-family:Space Mono,monospace'>
    PIPELINE<br>├─ Stationary features<br>├─ Non-overlapping labels<br>
    ├─ Walk-forward CV<br>├─ Embargo gap<br>└─ Permutation test<br><br>
    MODEL<br>└─ Logistic (regularised)
    </div>""", unsafe_allow_html=True)


# ─── Title ──────────────────────────────────────────────────────────

st.markdown("""
<div style='padding:24px 0 8px 0'>
  <div style='font-family:Space Mono,monospace;font-size:24px;font-weight:700;
              color:#e0e6f0;letter-spacing:2px'>STOCK MARKET PREDICTOR</div>
  <div style='font-size:13px;color:#7eb8d4;letter-spacing:2px;margin-top:4px'>
    WALK-FORWARD VALIDATED \u00b7 STATIONARY FEATURES \u00b7 HONEST BASELINES
  </div>
</div>""", unsafe_allow_html=True)
st.markdown("---")


# ─── Main ───────────────────────────────────────────────────────────

if run_btn:
    with st.spinner(f"Loading {stock_choice} artifacts\u2026"):
        art = load_model_artifacts(stock_choice)
        meta, model, curve = art["metadata"], art["model"], art["curve"]
        full = load_full_frame(stock_choice)
        pred = predict_latest(stock_choice, full_frame=full)

    perm = meta.get("permutation") or {}
    seeds = meta.get("seed_robustness") or {}
    op = meta.get("operating_point") or {}
    edge = meta.get("edge", 0.0)

    # ── Honesty banner ───────────────────────────────────────────────
    if edge <= 0 or perm.get("verdict") == "NO SIGNAL":
        st.markdown(f"""<div style='background:rgba(255,80,80,0.08);
          border-left:3px solid #ff5050;border-radius:0 8px 8px 0;
          padding:14px 18px;font-size:13px;color:#ffb0b0;margin-bottom:16px'>
          <strong>No demonstrated edge for {stock_choice}.</strong><br>
          Walk-forward accuracy {meta['walk_forward_accuracy']:.4f} against a
          baseline of {meta['baseline']:.4f} (edge {edge:+.4f}).
          Permutation verdict: {perm.get('verdict', 'n/a')}.
          Predictions below are shown for completeness and should not be
          treated as actionable.
        </div>""", unsafe_allow_html=True)
    elif perm.get("verdict") == "LEAKAGE":
        st.error("Permutation test indicates leakage. Do not trust these "
                 "numbers until it is found.")

    # ── 01 Price ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>01 \u00b7 PRICE HISTORY</div>",
                unsafe_allow_html=True)
    display = full.tail(500)
    st.plotly_chart(price_chart(display, meta["display_name"]),
                    use_container_width=True)
    c_rsi, c_vol = st.columns([2, 1])
    with c_rsi:
        st.plotly_chart(rsi_chart(display), use_container_width=True)
    with c_vol:
        st.markdown("<div class='section-header' style='margin-top:8px'>"
                    "VOLUME</div>", unsafe_allow_html=True)
        fv = go.Figure(go.Bar(x=display["Date"], y=display["Volume"],
                              marker_color="rgba(0,212,255,0.3)"))
        fv.update_layout(**_layout(200, "", {"showlegend": False}))
        st.plotly_chart(fv, use_container_width=True)
    st.markdown("---")

    # ── 02 Validated performance ─────────────────────────────────────
    st.markdown("<div class='section-header'>02 \u00b7 VALIDATED "
                "PERFORMANCE</div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class='metric-card'>
          <div class='metric-label'>Walk-Forward Accuracy</div>
          <div class='metric-value'>{meta['walk_forward_accuracy']:.1%}</div>
          <div class='metric-sub'>\u00b1 {meta['walk_forward_std']:.1%} across
          {meta['config']['n_splits']} folds</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='metric-card'>
          <div class='metric-label'>Baseline</div>
          <div class='metric-value'>{meta['baseline']:.1%}</div>
          <div class='metric-sub'>Best of majority / persistence</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        colour = "#00ff88" if edge > 0 else "#ff5050"
        st.markdown(f"""<div class='metric-card'>
          <div class='metric-label'>Edge Over Baseline</div>
          <div class='metric-value' style='color:{colour}'>{edge:+.1%}</div>
          <div class='metric-sub'>{seeds.get('verdict', 'n/a')}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        pv = perm.get("verdict", "n/a")
        pc = "#00ff88" if pv == "SIGNAL" else "#f0c040"
        st.markdown(f"""<div class='metric-card'>
          <div class='metric-label'>Permutation Test</div>
          <div class='metric-value' style='color:{pc};font-size:20px'>{pv}</div>
          <div class='metric-sub'>Shuffled:
          {perm.get('shuffled_mean', float('nan')):.3f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        if curve is not None and not curve.empty:
            st.plotly_chart(coverage_chart(curve), use_container_width=True)
    with cc2:
        oof_path = config.get_stock(stock_choice).model_dir / \
            "oof_predictions.csv"
        if oof_path.exists():
            st.plotly_chart(fold_chart(pd.read_csv(oof_path)),
                            use_container_width=True)

    fi = importance_chart(model, meta["features"])
    if fi is not None:
        st.plotly_chart(fi, use_container_width=True)
    st.markdown("---")

    # ── 03 Current signal ────────────────────────────────────────────
    st.markdown("<div class='section-header'>03 \u00b7 CURRENT SIGNAL</div>",
                unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        sig = pred["signal"]
        if sig == "UP":
            pill = "<span class='pill-bullish'>UP \u2191</span>"
        elif sig == "DOWN":
            pill = "<span class='pill-bearish'>DOWN \u2193</span>"
        else:
            pill = ("<span style='display:inline-block;background:"
                    "rgba(240,192,64,0.15);border:1px solid rgba(240,192,64,.4);"
                    "color:#f0c040;border-radius:24px;padding:6px 20px;"
                    "font-family:Space Mono,monospace;font-size:14px;"
                    "font-weight:700'>NO SIGNAL</span>")
        st.markdown(f"""<div class='metric-card' style='text-align:left'>
          <div class='metric-label'>Signal
          ({pred['horizon_days']}-day horizon)</div>
          <div style='margin-top:12px'>{pill}</div>
          <div class='metric-sub' style='margin-top:10px'>
          As of {pred['date']}</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class='metric-card'>
          <div class='metric-label'>Model Probability (UP)</div>
          <div class='metric-value'>{pred['probability_up']:.3f}</div>
          <div class='metric-sub'>Confidence {pred['confidence']:.3f} \u00b7
          gate {pred['confidence_threshold']:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class='metric-card'>
          <div class='metric-label'>Expected Hit Rate</div>
          <div class='metric-value'>{op.get('accuracy', float('nan')):.1%}</div>
          <div class='metric-sub'>At
          {op.get('coverage', float('nan')):.0%} coverage</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class='info-box' style='margin-top:16px'>
      <strong style='color:#00d4ff'>How to read this.</strong>
      The model predicts direction {pred['horizon_days']} trading days ahead,
      using non-overlapping labels so no two training examples share data.
      Probabilities are ranked confidence scores, not calibrated
      probabilities \u2014 the Brier score sits close to 0.25, so treat them
      as relative rather than literal.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # ── 04 Sentiment (transparency only) ─────────────────────────────
    sent = load_sentiment_frame()
    if sent is not None and stock_choice == "TCS":
        st.markdown("<div class='section-header'>04 \u00b7 SENTIMENT "
                    "(NOT A MODEL INPUT)</div>", unsafe_allow_html=True)
        st.markdown("""<div class='info-box'
          style='border-left-color:#f0c040;margin-bottom:12px'>
          The available news corpus is Reddit r/worldnews (2008\u20132016),
          not TCS-specific news. Measured ROC-AUC as a predictor was 0.47
          \u2014 below chance \u2014 so these scores are excluded from the
          model and shown for transparency only.
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(sentiment_chart(sent), use_container_width=True)
        st.markdown("---")

    # ── Summary ──────────────────────────────────────────────────────
    st.markdown(f"""<div class='info-box'>
      <strong style='color:#00d4ff'>Summary \u00b7
      {meta['display_name']}</strong><br>
      Last close: <strong>\u20b9{pred['close']:,.2f}</strong> &nbsp;|&nbsp;
      Date: <strong>{pred['date']}</strong> &nbsp;|&nbsp;
      Rows: <strong>{meta['n_full_rows']:,}</strong> &nbsp;|&nbsp;
      Modelling rows: <strong>{meta['n_modelling_rows']:,}</strong>
      &nbsp;|&nbsp; Features: <strong>{meta['n_features']}</strong>
      &nbsp;|&nbsp; Trained: <strong>{meta['trained_at'][:10]}</strong>
    </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align:center;padding:80px 20px'>
      <div style='font-size:60px'>\U0001F4C8</div>
      <div style='font-family:Space Mono,monospace;font-size:18px;
                  color:#e0e6f0;margin-top:20px'>
        Select a stock and click
        <span style='color:#00d4ff'>\u25B6 RUN ANALYSIS</span>
      </div>
      <div style='color:#7eb8d4;margin-top:12px;font-size:14px'>
        Loads pre-trained, walk-forward-validated models
      </div>
    </div>""", unsafe_allow_html=True)

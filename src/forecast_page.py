"""
"Next-morning forecast" page of the dashboard (stock by stock).

For the stock picked in the sidebar it shows how well a model predicts
  * the NEXT MORNING'S OPENING PRICE (up or down vs today's close), and
  * the NEXT DAY'S CLOSING PRICE (up or down vs today's close),
using only information available before the Indian market opens at 09:15 IST:
the stock's own price indicators, Nifty, the S&P 500 overnight return, the VIX and the
opening moves of the Japanese, Korean, Taiwanese and Hong Kong markets.
"+ company news" adds company-targeted sentiment (newspaper headlines or NSE announcements).

Every number is out-of-sample: models were trained only on earlier years and tested on a year they
never saw (walk-forward). Tables come from scripts/build_forecast_results.py; nothing is retrained here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FDIR = ROOT / "reports" / "forecast"
GREEN, RED, GREY, BLUE, AMBER, LIGHT = "#10b981", "#ef4444", "#64748b", "#6366f1", "#f59e0b", "#94a3b8"
MODELS = {"Logistic regression": "LogReg", "LightGBM": "LightGBM", "XGBoost": "XGBoost"}
OPEN_T, CLOSE_T = "Next morning's opening price", "Next day's closing price"


@st.cache_data(ttl=600)
def _t(name: str) -> pd.DataFrame:
    p = FDIR / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _pct(x):
    return "—" if pd.isna(x) else f"{100 * x:.1f}%"


def _pts(x):
    if pd.isna(x):
        return "—"
    return "±0.0 pts" if abs(x) < 0.0005 else f"{100 * x:+.1f} pts"


def _pval(p):
    return "—" if pd.isna(p) else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}")


def _verdict(row):
    if row is None:
        return "NOT AVAILABLE", GREY
    if row["edge"] > 0 and row["p_vs_guess"] < 0.05:
        return "SIGNAL", GREEN
    return "NO SIGNAL", AMBER


# ----------------------------------------------------------------------------- charts
def _bars_this_stock(rows: dict, title: str, layout) -> go.Figure:
    fig = go.Figure()
    targets = [t for t in (OPEN_T, CLOSE_T) if rows.get(t) is not None]
    series = [("Simple guess (always the usual direction)", "acc_guess", "rgba(148,163,184,0.45)"),
              ("Model", "acc_model", LIGHT),
              ("Model + company news", "acc_news", BLUE)]
    for name, col, colr in series:
        vals = [100 * rows[t][col] for t in targets]
        fig.add_trace(go.Bar(x=targets, y=vals, name=name, marker_color=colr,
                             text=[f"{v:.1f}%" for v in vals], textposition="outside",
                             hovertemplate="%{x}<br>" + name + ": %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=50, line=dict(color="#475569", dash="dot", width=1))
    lay = layout(360, title)
    lay["barmode"] = "group"
    lay["xaxis"]["type"] = "category"
    lay["yaxis"]["range"] = [0, 100]
    lay["yaxis"]["title"] = dict(text="Correct up/down calls, %", font=dict(size=10))
    lay["legend"] = dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
                         font=dict(size=10, color="#cbd5e1"))
    lay["margin"] = dict(l=55, r=20, t=45, b=70)
    fig.update_layout(**lay)
    return fig


def _by_year(df: pd.DataFrame, title: str, layout) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["year"], y=100 * df["acc_guess"], name="Simple guess", mode="lines+markers",
                             line=dict(color=GREY, width=2, dash="dot"),
                             hovertemplate="%{x}: simple guess %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["year"], y=100 * df["acc_model"], name="Model", mode="lines+markers",
                             line=dict(color=GREEN, width=2.5),
                             customdata=df["n"],
                             hovertemplate="%{x}: model %{y:.1f}% (%{customdata} test days)<extra></extra>"))
    beat = int((df["acc_model"] > df["acc_guess"]).sum())
    lay = layout(330, title)
    lay["yaxis"]["title"] = dict(text="Correct up/down calls, %", font=dict(size=10))
    lay["xaxis"]["title"] = dict(text=f"Test year — the model beat the simple guess in {beat} of {len(df)} years",
                                 font=dict(size=10))
    lay["xaxis"]["dtick"] = 1
    lay["legend"] = dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5,
                         font=dict(size=10, color="#cbd5e1"))
    lay["margin"] = dict(l=55, r=20, t=45, b=85)
    fig.update_layout(**lay)
    return fig


def _all_stocks(df: pd.DataFrame, key: str, title: str, layout) -> go.Figure:
    d = df.sort_values("edge")
    sig = (d["edge"] > 0) & (d["p_vs_guess"] < 0.05)
    colr = [BLUE if t == key else (GREEN if s else AMBER) for t, s in zip(d["ticker"], sig)]
    fig = go.Figure(go.Bar(
        y=d["ticker"], x=100 * d["edge"], orientation="h", marker_color=colr,
        text=[f"{100 * e:+.1f}" for e in d["edge"]], textposition="outside",
        customdata=np.stack([100 * d["acc_model"], 100 * d["acc_guess"], d["p_vs_guess"]], axis=-1),
        hovertemplate="%{y}: model %{customdata[0]:.1f}% vs simple guess %{customdata[1]:.1f}%"
                      "<br>p = %{customdata[2]:.4f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#475569", width=1))
    lay = layout(max(300, 34 * len(d) + 80), title)
    lay["xaxis"]["title"] = dict(text="Points above the simple guess  ·  blue = selected stock, green = signal (p < 0.05), "
                                      "amber = no signal", font=dict(size=10))
    lay["yaxis"]["type"] = "category"
    lay["margin"] = dict(l=95, r=40, t=45, b=60)
    lay["showlegend"] = False
    fig.update_layout(**lay)
    return fig


# ----------------------------------------------------------------------------- page
def render_forecast_page(key: str, display_names: dict, layout, colors: dict) -> None:
    muted = colors.get("text_muted", "#94a3b8")
    name = display_names.get(key, key)
    st.markdown("""<style>
      [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
      [data-testid="stMain"] [data-testid="stRadioOption"] p,
      [data-testid="stMain"] [data-testid="stRadioOption"] div { color: #cbd5e1 !important; }
    </style>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='padding:8px 0 10px 0'>
      <h1 style='margin:0;font-size:26px'>Next-morning forecast</h1>
      <div style='font-size:13px;color:{muted};margin-top:4px'>
        <strong>{name}</strong> · will the price be up or down tomorrow? Predicted before the market opens,
        tested only on years the model never saw
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='transparency-box' style='border-left-color:{BLUE};'>
      <strong>What the model knows (all before 09:15 IST):</strong> this stock's own price indicators, the Nifty,
      last night's US market (S&amp;P 500) and its fear index (VIX), and how the Japanese, Korean, Taiwanese and
      Hong Kong markets opened this morning. The "+ company news" version also reads the sentiment of headlines or
      stock-exchange announcements about this company.
    </div>""", unsafe_allow_html=True)

    summ, years = _t("stock_summary"), _t("stock_by_year")
    if summ.empty:
        st.warning("No forecast tables found. Run `python scripts/build_forecast_results.py`.")
        return

    c1, c2 = st.columns([3, 2])
    with c1:
        periods = sorted(summ["period"].unique(), reverse=True)
        period = st.radio("Test period", periods, horizontal=True,
                          help="2013–2025 covers all 10 stocks and uses company announcements as news; "
                               "2007–2022 uses Times of India headlines (Eternal is too new for it).")
    with c2:
        model_label = st.selectbox("Model", list(MODELS), index=0)
    model = MODELS[model_label]

    s = summ[(summ["period"] == period) & (summ["model"] == model)]
    rows = {}
    for t in (OPEN_T, CLOSE_T):
        r = s[(s["target"] == t) & (s["ticker"] == key)]
        rows[t] = r.iloc[0] if len(r) else None
    if all(v is None for v in rows.values()):
        st.info(f"{name} has no test results for {period} with {model_label}. Try the other period or model.")
        return

    # --- cards
    cols = st.columns(4)

    def _card(col, head, val, sub, colr=None):
        col.markdown(f"<div class='terminal-card'><div class='terminal-card-header'>{head}</div>"
                     f"<div class='terminal-card-value' style='font-size:22px;{'color:' + colr if colr else ''}'>{val}</div>"
                     f"<div class='terminal-card-sub'>{sub}</div></div>", unsafe_allow_html=True)

    for col, t, head in [(cols[0], OPEN_T, "Opening price: model"), (cols[1], CLOSE_T, "Closing price: model")]:
        r = rows[t]
        if r is None:
            _card(col, head, "—", "not tested for this model / period")
        else:
            verdict, vc = _verdict(r)
            _card(col, head, _pct(r["acc_model"]),
                  f"simple guess {_pct(r['acc_guess'])} · {_pts(r['edge'])} · {_pval(r['p_vs_guess'])}"
                  f"<br><span style='color:{vc};font-weight:700'>{verdict}</span>")
    ro = rows[OPEN_T] if rows[OPEN_T] is not None else rows[CLOSE_T]
    _card(cols[2], "Adding company news", _pts(ro["news_effect"]),
          f"change in accuracy, {ro['target'].lower()} · {_pval(ro['p_news'])}",
          GREEN if (ro["news_effect"] > 0 and ro["p_news"] < 0.05) else None)
    _card(cols[3], "Test days", f"{int(ro['n']):,}", f"{int(ro['first_year'])}–{int(ro['last_year'])}, never seen in training")

    # --- 01 this stock
    st.markdown(f"<div class='section-heading'>01 · {key}: MODEL vs SIMPLE GUESS</div>", unsafe_allow_html=True)
    st.plotly_chart(_bars_this_stock(rows, f"{name.upper()} — CORRECT UP/DOWN CALLS ({model_label})", layout),
                    width="stretch")
    st.markdown(f"<div style='font-size:12px;color:{muted};margin-top:-6px'>The simple guess always predicts the "
                "direction that was most common in the training years; the dotted line is 50%, a coin flip. A model is only useful if it beats the simple guess. "
                "Opening-price accuracy is high for both because prices open higher more often than not.</div>",
                unsafe_allow_html=True)

    # --- 02 year by year
    st.markdown("<div class='section-heading'>02 · IS IT CONSISTENT? YEAR BY YEAR</div>", unsafe_allow_html=True)
    tgt = st.radio("Prediction", [OPEN_T, CLOSE_T], horizontal=True, key="fc_year_target")
    y = years[(years["period"] == period) & (years["model"] == model) & (years["target"] == tgt) & (years["ticker"] == key)]
    if y.empty:
        st.info("No yearly results for this choice.")
    else:
        st.plotly_chart(_by_year(y.sort_values("year"), f"{key} — {tgt.upper()}, EACH TEST YEAR", layout), width="stretch")

    # --- 03 all stocks
    st.markdown("<div class='section-heading'>03 · HOW THIS STOCK COMPARES WITH THE OTHER NINE</div>", unsafe_allow_html=True)
    a = s[(s["target"] == tgt) & (s["ticker"] != "ALL")]
    pooled = s[(s["target"] == tgt) & (s["ticker"] == "ALL")]
    if not a.empty:
        st.plotly_chart(_all_stocks(a, key, f"{tgt.upper()}: POINTS ABOVE THE SIMPLE GUESS, ALL STOCKS", layout),
                        width="stretch")
        if len(pooled):
            p0 = pooled.iloc[0]
            n_sig = int(((a["edge"] > 0) & (a["p_vs_guess"] < 0.05)).sum())
            st.markdown(f"<div style='font-size:12px;color:{muted};margin-top:-6px'>All stocks together: "
                        f"{_pct(p0['acc_model'])} vs {_pct(p0['acc_guess'])} for the simple guess "
                        f"({_pts(p0['edge'])}, {_pval(p0['p_vs_guess'])}). {n_sig} of {len(a)} stocks show a "
                        f"statistically significant signal. Adding company news changes the pooled accuracy by "
                        f"{_pts(p0['news_effect'])} ({_pval(p0['p_news'])}).</div>", unsafe_allow_html=True)

    # --- 04 market context
    st.markdown("<div class='section-heading'>04 · THE WHOLE MARKET, FOR CONTEXT</div>", unsafe_allow_html=True)
    m = _t("market_summary")
    if not m.empty:
        t = pd.DataFrame({
            "Prediction": m["task"], "Test days": m["n"],
            "Simple guess": (100 * m["acc_guess"]).round(1), "Model": (100 * m["acc_model"]).round(1),
            "Gain (pts)": (100 * m["edge"]).round(1), "Model vs guess": m["p_vs_guess"].map(_pval),
            "Model + news": (100 * m["acc_news"]).round(1), "News effect (pts)": (100 * m["news_effect"]).round(1),
            "News vs no news": m["p_news"].map(_pval)})
        st.dataframe(t, hide_index=True, width="stretch")

    # --- reading guide
    st.markdown("""
    <div class='transparency-box'>
      <strong>How to read this page.</strong> The real signal comes from information that arrives overnight, before
      the Indian market opens: what the US and Asian markets did while India was closed. Company news adds almost
      nothing on top of it: usually within ±0.5 points, no more than random noise, and for the Sensex opening price
      it even made the model slightly worse. The one place news clearly helps
      is the US, when New York Times news published before the opening bell is added (+2.5 points, p &lt; 0.001).
      These are tested historical results, not a live forecast: a live version would need fresh overnight prices each
      morning. A forecast made before the open also cannot be traded at yesterday's closing price, and trading costs are
      not included.
    </div>""", unsafe_allow_html=True)

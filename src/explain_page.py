"""
"Why did it move?" page of the dashboard.

Shows, for one stock, the news that appeared on each trading day next to that day's price move,
scored by the company-targeted (entity-level) FinBERT model, plus the evidence on how closely news
tone and price moves line up. Everything here is EXPLANATORY: it describes days that already happened.

Data (all shipped in the repository, nothing is recomputed here):
  data/raw/<STOCK>_raw.csv                         prices
  data/processed/news/<STOCK>_news_items.csv       headlines and filings with their scores
  sentiment_study/reports/explain/*.csv            statistics (built by sentiment_study/src/explain_moves.py)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from dataio import load_raw_ohlcv

ROOT = Path(__file__).resolve().parents[1]
NEWS_DIR = ROOT / "data" / "processed" / "news"
EXPLAIN_DIR = ROOT / "sentiment_study" / "reports" / "explain"
GREEN, RED, GREY, BLUE, AMBER = "#10b981", "#ef4444", "#64748b", "#6366f1", "#f59e0b"
VARIANTS = {"Excluding price-report headlines": "excluding price reports", "All headlines": "all headlines"}


# ----------------------------------------------------------------------------- data
@st.cache_data(ttl=600)
def _prices(key: str) -> pd.DataFrame:
    d = load_raw_ohlcv(config.get_stock(key).raw_path, verbose=False)
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").reset_index(drop=True)
    d["ret_pct"] = np.log(d["Close"]).diff() * 100
    return d


@st.cache_data(ttl=600)
def _news(key: str) -> pd.DataFrame:
    p = NEWS_DIR / f"{key}_news_items.csv"
    if not p.exists():
        return pd.DataFrame()
    n = pd.read_csv(p)
    n["trade_date"] = pd.to_datetime(n["trade_date"])
    return n


@st.cache_data(ttl=600)
def _table(name: str) -> pd.DataFrame:
    p = EXPLAIN_DIR / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# ----------------------------------------------------------------------------- charts
def _timeline(px_: pd.DataFrame, daily: pd.DataFrame, title: str, layout) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px_["Date"], y=px_["Close"], name="Close", line=dict(color="#94a3b8", width=1.2),
                             hovertemplate="%{x|%d %b %Y}<br>Close %{y:,.2f}<extra></extra>"))
    if not daily.empty:
        size = 6 + np.clip(daily["ret_pct"].abs().fillna(0), 0, 10) * 1.6
        fig.add_trace(go.Scatter(
            x=daily["trade_date"], y=daily["Close"], mode="markers", name="News day",
            marker=dict(size=size, color=daily["score"], colorscale=[[0, RED], [0.5, "#e2e8f0"], [1, GREEN]],
                        cmin=-1, cmax=1, line=dict(width=0.5, color="#0b0f19"),
                        colorbar=dict(title=dict(text="Sentiment<br>towards<br>company", font=dict(size=10)),
                                      thickness=10, len=0.7, tickfont=dict(size=9))),
            customdata=np.stack([daily["ret_pct"].round(2), daily["n"], daily["top"].str.slice(0, 110),
                                 daily["score"].round(2)], axis=-1),
            hovertemplate="%{x|%d %b %Y}<br>Move %{customdata[0]}%  ·  %{customdata[1]} item(s)"
                          "<br>Mean sentiment %{customdata[3]}<br><i>%{customdata[2]}</i><extra></extra>"))
    lay = layout(460, title)
    lay["margin"] = dict(l=55, r=25, t=70, b=35)
    lay["showlegend"] = False
    lay["yaxis"]["type"] = "log"
    lay["yaxis"]["title"] = dict(text="Close (log scale)", font=dict(size=10))
    lay["xaxis"]["rangeslider"] = dict(visible=False)
    lay["xaxis"]["rangeselector"] = dict(
        buttons=[dict(count=1, label="1Y", step="year", stepmode="backward"),
                 dict(count=3, label="3Y", step="year", stepmode="backward"),
                 dict(count=5, label="5Y", step="year", stepmode="backward"),
                 dict(step="all", label="ALL")],
        bgcolor="#121824", activecolor="#1e293b", font=dict(color="#f8fafc", size=10), x=0, y=1.02)
    fig.update_layout(**lay)
    return fig


def _lead_lag(ll: pd.DataFrame, title: str, layout) -> go.Figure:
    lab = {k: ("same day" if k == 0 else f"{abs(k)}d {'before' if k < 0 else 'after'}") for k in ll["lag"]}
    col = [GREEN if (p < 0.05 and k == 0) else (BLUE if p < 0.05 else GREY) for k, p in zip(ll["lag"], ll["p"])]
    fig = go.Figure(go.Bar(x=[lab[k] for k in ll["lag"]], y=ll["rho"], marker_color=col,
                           customdata=np.stack([ll["p"], ll["n"]], axis=-1),
                           hovertemplate="%{x}<br>Spearman %{y:.3f}<br>p = %{customdata[0]:.4f} · n = %{customdata[1]}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#475569", width=1))
    lay = layout(340, title)
    lay["xaxis"]["type"] = "category"
    lay["margin"] = dict(l=55, r=20, t=40, b=85)
    lay["yaxis"]["title"] = dict(text="Correlation of sentiment with return", font=dict(size=10))
    lay["xaxis"]["title"] = dict(text="Return on the day shown, relative to the news day · coloured bars: p < 0.05", font=dict(size=10))
    fig.update_layout(**lay)
    return fig


def _quintiles(q: pd.DataFrame, title: str, layout) -> go.Figure:
    names = ["Q1 most negative", "Q2", "Q3", "Q4", "Q5 most positive"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=q["same_day"], name="Same-day return", marker_color=GREEN))
    fig.add_trace(go.Bar(x=names, y=q["next_day"], name="Next-day return", marker_color=GREY))
    fig.add_hline(y=0, line=dict(color="#475569", width=1))
    lay = layout(300, title)
    lay["barmode"] = "group"
    lay["xaxis"]["type"] = "category"
    lay["legend"] = dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5, font=dict(size=10, color="#cbd5e1"))
    lay["margin"] = dict(l=55, r=20, t=40, b=80)
    lay["yaxis"]["title"] = dict(text="Mean return, %", font=dict(size=10))
    lay["xaxis"]["title"] = dict(text="News-day sentiment quintile", font=dict(size=10))
    fig.update_layout(**lay)
    return fig


def _event(ev: pd.DataFrame, title: str, layout) -> go.Figure:
    fig = go.Figure()
    for grp, c, fill in [("Positive news", GREEN, "rgba(16,185,129,0.15)"), ("Negative news", RED, "rgba(239,68,68,0.15)")]:
        g = ev[ev["group"] == grp].sort_values("day")
        if g.empty:
            continue
        fig.add_trace(go.Scatter(x=list(g["day"]) + list(g["day"][::-1]), y=list(g["hi"]) + list(g["lo"][::-1]),
                                 fill="toself", fillcolor=fill, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=g["day"], y=g["car"], name=f"{grp} (n = {int(g['n_events'].iloc[0])})",
                                 line=dict(color=c, width=2), mode="lines+markers",
                                 hovertemplate="day %{x}<br>cumulative %{y:.2f}%<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#94a3b8", dash="dot", width=1))
    fig.add_hline(y=0, line=dict(color="#475569", width=1))
    lay = layout(340, title)
    lay["legend"] = dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, font=dict(size=10, color="#cbd5e1"))
    lay["margin"] = dict(l=55, r=20, t=40, b=85)
    lay["xaxis"]["title"] = dict(text="Trading days from the news day (0)", font=dict(size=10))
    lay["yaxis"]["title"] = dict(text="Cumulative return vs Sensex, % (95% band)", font=dict(size=10))
    fig.update_layout(**lay)
    return fig


def _coverage(bc: pd.DataFrame, title: str, layout) -> go.Figure:
    colors = {"News sentiment matches the move": GREEN, "Neutral news": "#94a3b8",
              "News sentiment opposite to the move": RED, "No news that day": "#334155"}
    fig = go.Figure()
    for _, r in bc.iterrows():
        fig.add_trace(go.Bar(y=[""], x=[r["share"] * 100], name=r["category"], orientation="h",
                             marker_color=colors.get(r["category"], GREY),
                             hovertemplate=f"{r['category']}<br>{int(r['count'])} days ({r['share']:.0%})<extra></extra>"))
    lay = layout(300, title)
    lay["barmode"] = "stack"
    lay["legend"] = dict(orientation="h", yanchor="top", y=-0.45, xanchor="center", x=0.5, font=dict(size=10, color="#cbd5e1"),
                         traceorder="normal")
    lay["margin"] = dict(l=20, r=20, t=40, b=110)
    lay["xaxis"]["title"] = dict(text="% of the largest 5% of daily moves, 2001–2022", font=dict(size=10))
    lay["xaxis"]["range"] = [0, 100]
    fig.update_layout(**lay)
    return fig


# ----------------------------------------------------------------------------- page
def render_explain_page(key: str, display_names: dict, layout, colors: dict) -> None:
    muted, card = colors.get("text_muted", "#94a3b8"), colors.get("card_bg", "#121824")
    name = display_names.get(key, key)
    st.markdown(f"""
    <div style='padding:8px 0 10px 0'>
      <h1 style='margin:0;font-size:26px'>Why did it move?</h1>
      <div style='font-size:13px;color:{muted};margin-top:4px'>
        <strong>{name}</strong> · news on each trading day, scored for sentiment <em>towards this company</em>,
        next to that day's price move
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<style>
      [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
      [data-testid="stMain"] [data-testid="stRadioOption"] p,
      [data-testid="stMain"] [data-testid="stRadioOption"] div,
      [data-testid="stMain"] [data-baseweb="select"] div { color: #cbd5e1 !important; }
    </style>""", unsafe_allow_html=True)
    px_, news = _prices(key), _news(key)
    if news.empty:
        st.warning("No news file for this stock. Run `python sentiment_study/src/explain_moves.py`.")
        return
    summ = _table("summary")
    srow = summ[summ["scope"] == key].iloc[0] if (not summ.empty and key in set(summ["scope"])) else None

    # --- filters
    f1, f2, f3 = st.columns([2, 2, 1.4])
    with f1:
        sources = st.multiselect("News sources", ["Times of India headline", "NSE filing"],
                                 default=["Times of India headline", "NSE filing"])
    with f2:
        variant_label = st.radio("Headlines", list(VARIANTS), horizontal=True,
                                 help="Price-report headlines ('Infosys shares fall 3%') describe the move "
                                      "instead of explaining it, so they are hidden by default.")
    with f3:
        min_abs = st.slider("Min |move| %", 0.0, 10.0, 0.0, 0.5)
    variant = VARIANTS[variant_label]
    n = news[news["source"].isin(sources)]
    if variant == "excluding price reports":
        n = n[~n["price_report"]]

    # --- metric cards
    toi = news[news["source"] == "Times of India headline"]; fil = news[news["source"] == "NSE filing"]
    c1, c2, c3, c4 = st.columns(4)

    def _card(col, head, val, sub):
        col.markdown(f"<div class='terminal-card'><div class='terminal-card-header'>{head}</div>"
                     f"<div class='terminal-card-value' style='font-size:22px'>{val}</div>"
                     f"<div class='terminal-card-sub'>{sub}</div></div>", unsafe_allow_html=True)

    _card(c1, "Headlines naming the company", f"{len(toi):,}",
          f"Times of India, {toi.trade_date.min():%Y}–{toi.trade_date.max():%Y}" if len(toi) else "none")
    _card(c2, "Company filings", f"{len(fil):,}",
          f"NSE, {fil.trade_date.min():%Y}–{fil.trade_date.max():%Y}" if len(fil) else "none")
    if srow is not None and pd.notna(srow["same_day_rho"]):
        _card(c3, "Same-day link (no price reports)", f"{srow['same_day_rho']:+.3f}",
              f"Spearman, p = {srow['same_day_p']:.3f} · next day {srow['next_day_rho']:+.3f}")
        _card(c4, "Big moves with matching news", f"{srow['big_moves_matching_news']:.0%}",
              f"opposite {srow['big_moves_opposite_news']:.0%} · no news {srow['big_moves_no_news']:.0%}")
    else:
        _card(c3, "Same-day link", "—", "too few headlines for a reliable estimate")
        _card(c4, "Big moves with matching news", "—", "too few headlines")

    # --- 01 timeline
    st.markdown("<div class='section-heading'>01 · PRICE AND NEWS TIMELINE</div>", unsafe_allow_html=True)
    if n.empty:
        st.info("No news items for this selection.")
        return
    g = n.groupby("trade_date")
    daily = pd.DataFrame({"score": g["targeted"].mean(), "n": g.size(),
                          "top": g.apply(lambda x: x.loc[x["targeted"].abs().idxmax(), "text"])}).reset_index()
    daily = daily.merge(px_[["Date", "Close", "ret_pct"]], left_on="trade_date", right_on="Date", how="left")
    daily = daily[daily["ret_pct"].abs().fillna(0) >= min_abs]
    st.markdown(f"<div style='font-size:12px;color:{muted};margin:-4px 0 4px 0'>Each dot is a day with news about the "
                "company. Colour = mean sentiment towards the company (red negative, green positive); size = size of "
                "that day's price move. Hover a dot for the strongest item.</div>", unsafe_allow_html=True)
    st.plotly_chart(_timeline(px_, daily, f"{name.upper()} — CLOSE PRICE AND NEWS DAYS", layout), width="stretch")

    # --- 02 day explorer
    st.markdown("<div class='section-heading'>02 · DAY EXPLORER</div>", unsafe_allow_html=True)
    days = daily.assign(absr=daily["ret_pct"].abs()).sort_values("absr", ascending=False)
    if days.empty:
        st.info("No news days pass the filter.")
    else:
        opts = [f"{d:%d %b %Y}  ·  {r:+.2f}%  ·  {k} item(s)" for d, r, k in zip(days["trade_date"], days["ret_pct"].fillna(0), days["n"])]
        pick = st.selectbox("Pick a news day (largest moves first)", range(len(opts)), format_func=lambda i: opts[i])
        row = days.iloc[pick]
        items = n[n["trade_date"] == row["trade_date"]].copy()
        d1, d2 = st.columns([1, 3])
        clr = GREEN if row["ret_pct"] >= 0 else RED
        d1.markdown(f"<div class='terminal-card'><div class='terminal-card-header'>Move that day</div>"
                    f"<div class='terminal-card-value' style='color:{clr}'>{row['ret_pct']:+.2f}%</div>"
                    f"<div class='terminal-card-sub'>Mean sentiment towards the company {row['score']:+.2f}</div></div>",
                    unsafe_allow_html=True)
        agree = np.sign(row["score"]) == np.sign(row["ret_pct"]) and abs(row["score"]) >= 0.1
        verdict = ("News tone matches the move" if agree else
                   "News tone is neutral" if abs(row["score"]) < 0.1 else "News tone is opposite to the move")
        d2.markdown(f"<div class='transparency-box'><strong>{verdict}.</strong> Each item is scored for how it reads "
                    f"<em>for {name.split(' (')[0]}</em> (−1 negative … +1 positive). The plain-FinBERT column scores the "
                    f"whole text without targeting, for comparison.</div>", unsafe_allow_html=True)
        show = items[["published", "source", "text", "label", "targeted", "plain", "price_report"]].rename(columns={
            "published": "Published", "source": "Source", "text": "Text", "label": "Label",
            "targeted": "Towards company", "plain": "Plain FinBERT", "price_report": "Price report"})
        st.dataframe(show, hide_index=True, width="stretch")

    # --- 03 biggest moves
    st.markdown("<div class='section-heading'>03 · THE LARGEST MOVES AND THEIR NEWS (2001–2022)</div>", unsafe_allow_html=True)
    bt = _table("big_moves_top")
    bt = bt[bt["scope"] == key] if not bt.empty else bt
    if bt.empty:
        st.info("No large-move table for this stock.")
    else:
        bt = bt.copy()
        bt["headline"] = bt["headline"].fillna("— no headline naming the company that day")
        st.dataframe(bt.drop(columns=["scope"]).rename(columns={
            "date": "Date", "return_pct": "Move %", "sentiment": "Mean sentiment that day",
            "headline": "Strongest headline (no price reports)", "headline_score": "Its score"}),
            hide_index=True, width="stretch", height=320)

    # --- 04 evidence
    st.markdown("<div class='section-heading'>04 · HOW CLOSELY NEWS TONE AND PRICE MOVES LINE UP</div>", unsafe_allow_html=True)
    scope_label = st.radio("Evidence for", [f"This stock ({key})", "All 10 stocks (pooled)", "Market (Sensex)"],
                           horizontal=True)
    scope = key if scope_label.startswith("This") else ("ALL10" if scope_label.startswith("All") else "SENSEX")
    ll, qq, ev, bc = _table("lead_lag"), _table("quintiles"), _table("event_study"), _table("big_moves_coverage")
    e1, e2 = st.columns(2)
    with e1:
        x = ll[(ll["scope"] == scope) & (ll["variant"] == variant)].sort_values("lag") if not ll.empty else ll
        if x.empty or x["rho"].isna().all():
            st.info("Too few news days for this scope.")
        else:
            st.plotly_chart(_lead_lag(x, "SENTIMENT vs RETURN, DAYS BEFORE AND AFTER THE NEWS", layout), width="stretch")
    with e2:
        x = qq[(qq["scope"] == scope) & (qq["variant"] == variant)].sort_values("q") if not qq.empty else qq
        if x.empty:
            st.info("Too few news days for this scope." if scope != "ALL10" else "Quintiles are shown for single stocks and the Sensex.")
        else:
            st.plotly_chart(_quintiles(x, "MEAN RETURN BY SENTIMENT QUINTILE", layout), width="stretch")
    e3, e4 = st.columns(2)
    with e3:
        x = ev[ev["scope"] == scope] if not ev.empty else ev
        if x.empty:
            st.info("Too few strong news days for this scope." if scope != "ALL10" else "The event study is shown for single stocks and the Sensex.")
        else:
            st.plotly_chart(_event(x, "CUMULATIVE RETURN AROUND STRONG NEWS DAYS (no price reports)", layout), width="stretch")
    with e4:
        x = bc[bc["scope"] == scope] if not bc.empty else bc
        if x.empty:
            st.info("No coverage table for this scope." if scope != "ALL10" else "Coverage is shown for single stocks and the Sensex.")
        else:
            st.plotly_chart(_coverage(x, "THE LARGEST MOVES: WAS THERE NEWS, AND DID IT AGREE?", layout), width="stretch")

    # --- 05 same-day accuracy + reading guide
    st.markdown("<div class='section-heading'>05 · SAME-DAY UP/DOWN ACCURACY WITH AND WITHOUT SENTIMENT</div>", unsafe_allow_html=True)
    sd = _table("same_day_accuracy")
    if not sd.empty:
        t = sd.copy()
        t["Without"] = (t["acc_without"] * 100).round(1); t["With"] = (t["acc_with"] * 100).round(1)
        t["Gain (pts)"] = t["gain_pts"].round(2); t["p (McNemar)"] = t["p_mcnemar"].round(4)
        st.dataframe(t[["scope", "model", "sentiment", "rows", "n", "Without", "With", "Gain (pts)", "p (McNemar)"]]
                     .rename(columns={"scope": "Scope", "model": "Model", "sentiment": "Sentiment", "rows": "Rows", "n": "n"}),
                     hide_index=True, width="stretch")
    st.markdown(f"""
    <div class='transparency-box'>
      <strong>How to read this page.</strong> Everything here explains days that already happened; none of it is a forecast.
      Headlines that merely report the move ("Sensex down 200 points") make sentiment look far more explanatory than it is,
      so the honest numbers are the ones with price reports removed: news tone and same-day returns line up weakly but
      measurably (Sensex Spearman ≈ 0.06, p &lt; 0.001; 10 stocks pooled ≈ 0.03, p ≈ 0.005), and the link is gone the next day.
      Much of the move often shows up the day <em>before</em> the headline's date, because a newspaper dated d frequently reports
      news from day d−1.
    </div>""", unsafe_allow_html=True)

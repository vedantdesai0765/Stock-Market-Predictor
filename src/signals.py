"""
src/signals.py
Evidence-Based Actionable Trading Signal Engine.

Produces BUY, SELL, and HOLD signals combining:
- Calibrated model probabilities from walk-forward models
- Expected return vs. transaction cost hurdles
- Technical confirmation filters (RSI, MACD, Moving Averages)
- Entity-targeted FinBERT news sentiment confirmation
- Market regime filters (Trending, Sideways, High Volatility)

Adheres strictly to academic honesty: abstains (HOLD) whenever confidence is
insufficient, indicators conflict, or expected gross returns do not overcome
realistic transaction friction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------------------
# Default Configuration & Cost Hurdles (NSE Equities)
# ---------------------------------------------------------------------------
# Realistic Indian Delivery Round-Trip Friction:
# STT (0.1% buy + 0.1% sell = 0.20% for physical delivery, or ~0.025% intraday)
# Brokerage (0.03%), Exchange/SEBI/Stamp/GST (~0.03%), Bid-ask slippage (~0.04%)
# Default conservative hurdle: 15 bps (0.15% round-trip)
DEFAULT_FRICTION_BPS = 15.0  # basis points (0.15%)
DEFAULT_FRICTION_RATE = DEFAULT_FRICTION_BPS / 10000.0  # 0.0015

CONFIDENCE_GATE_DEFAULT = 0.03  # abs(prob - 0.5) >= 0.03
SENTIMENT_THRESHOLD = 0.05     # |sentiment score| >= 0.05 for directional bias


# ---------------------------------------------------------------------------
# Sentiment Extraction & Aggregation
# ---------------------------------------------------------------------------

def load_clean_news_sentiment(
    stock_key: str,
    base_dir: Optional[Path] = None,
    filter_price_reports: bool = True,
    lookback_window: int = 5
) -> pd.DataFrame:
    """
    Extracts entity-targeted FinBERT news sentiment for a given stock.
    
    Filters out price-recap articles (price_report == True) to prevent
    circular reasoning or lookahead bias.
    Computes rolling lookback mean sentiment over trading days.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    news_file = base_dir / "data" / "processed" / "news" / f"{stock_key}_news_items.csv"
    if not news_file.exists():
        return pd.DataFrame(columns=["Date", "sentiment_score", "sentiment_count", "sentiment_label"])

    df = pd.read_csv(news_file)
    if "trade_date" not in df.columns:
        return pd.DataFrame(columns=["Date", "sentiment_score", "sentiment_count", "sentiment_label"])

    df["Date"] = pd.to_datetime(df["trade_date"])
    
    if filter_price_reports and "price_report" in df.columns:
        df = df[~df["price_report"].fillna(False)].copy()

    # Aggregate by trade date
    sentiment_col = "targeted" if "targeted" in df.columns else "plain"
    daily = df.groupby("Date").agg(
        sentiment_score=(sentiment_col, "mean"),
        sentiment_count=(sentiment_col, "count"),
    ).reset_index()

    daily = daily.sort_values("Date").reset_index(drop=True)
    daily["sentiment_rolling"] = daily["sentiment_score"].rolling(
        window=lookback_window, min_periods=1
    ).mean()

    return daily


# ---------------------------------------------------------------------------
# Market Regime Identification
# ---------------------------------------------------------------------------

def classify_market_regimes(
    df: pd.DataFrame,
    vol_col: str = "realized_vol_20",
    sma20_col: str = "SMA_20",
    sma50_col: str = "SMA_50",
    close_col: str = "Close",
    vol_high_quantile: float = 0.80,
    vol_low_quantile: float = 0.30
) -> pd.DataFrame:
    """
    Classifies each trading day into a market regime:
    - TRENDING_BULL: Close > SMA20 > SMA50 and normal/moderate volatility
    - TRENDING_BEAR: Close < SMA20 < SMA50 and normal/moderate volatility
    - HIGH_VOLATILITY: Realized vol > 80th historical percentile
    - SIDEWAYS_CHOPPY: Flat moving averages / mixed trend and low-to-medium vol
    """
    out = df.copy()
    if vol_col not in out.columns or close_col not in out.columns:
        out["regime"] = "SIDEWAYS_CHOPPY"
        return out

    # Expanding historical volatility quantiles to prevent lookahead
    expanding_vol = out[vol_col].expanding(min_periods=60)
    high_threshold = expanding_vol.quantile(vol_high_quantile).shift(1)
    low_threshold = expanding_vol.quantile(vol_low_quantile).shift(1)

    regimes = []
    for idx, row in out.iterrows():
        vol = row.get(vol_col, np.nan)
        c = row.get(close_col, np.nan)
        s20 = row.get(sma20_col, np.nan)
        s50 = row.get(sma50_col, np.nan)
        v_high = high_threshold.iloc[idx] if idx < len(high_threshold) else np.nan

        if pd.notnull(vol) and pd.notnull(v_high) and vol > v_high:
            regimes.append("HIGH_VOLATILITY")
        elif pd.notnull(c) and pd.notnull(s20) and pd.notnull(s50):
            if c >= s20 and s20 >= s50:
                regimes.append("TRENDING_BULL")
            elif c <= s20 and s20 <= s50:
                regimes.append("TRENDING_BEAR")
            else:
                regimes.append("SIDEWAYS_CHOPPY")
        else:
            regimes.append("SIDEWAYS_CHOPPY")

    out["market_regime"] = regimes
    return out


# ---------------------------------------------------------------------------
# Out-of-Sample Probability Calibration (Platt / Isotonic)
# ---------------------------------------------------------------------------

def calibrate_oof_probabilities(oof: pd.DataFrame) -> pd.DataFrame:
    """
    Recalibrates model probabilities strictly fold-by-fold out-of-sample.
    For fold f, uses folds < f to fit isotonic regression.
    """
    df = oof.copy()
    df["y_proba_calibrated"] = df["y_proba"]
    
    if "fold" not in df.columns:
        return df

    folds = sorted(df["fold"].unique())
    for f in folds[1:]:
        past = df[df["fold"] < f]
        current = df[df["fold"] == f]
        if past["y_true"].nunique() >= 2 and len(past) >= 30:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
            iso.fit(past["y_proba"], past["y_true"])
            df.loc[current.index, "y_proba_calibrated"] = iso.predict(current["y_proba"])
            
    return df


# ---------------------------------------------------------------------------
# Multi-Factor Signal Engine
# ---------------------------------------------------------------------------

def generate_signals(
    stock_key: str,
    variant: str = "A",
    long_only: bool = True,
    friction_rate: float = DEFAULT_FRICTION_RATE,
    confidence_gate: float = CONFIDENCE_GATE_DEFAULT,
    base_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generates systematic trading signals over the out-of-sample evaluation period.

    Variants:
      'A' - Model Alone: Raw or calibrated probability threshold with friction gating.
      'B' - Model + Technical: Model signal confirmed by RSI, MACD histogram, and SMA trend.
      'C' - Model + Sentiment: Model signal confirmed by Entity-targeted FinBERT score.
      'D' - Multi-Factor Ensemble: Model + Technical + Sentiment all aligned.
      'E' - Regime-Filtered: Trades only in favorable regimes; stays in Cash during
            HIGH_VOLATILITY or choppy SIDEWAYS phases.

    Returns:
      DataFrame containing dates, prices, model prob, technical flags,
      sentiment score, market regime, signal (-1, 0, +1), expected return,
      and rationales.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    # 1. Load OOF predictions
    oof_path = base_dir / "models" / stock_key / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"No OOF predictions found for {stock_key} at {oof_path}")

    oof = pd.read_csv(oof_path)
    oof["Date"] = pd.to_datetime(oof["Date"])
    oof = calibrate_oof_probabilities(oof)

    # 2. Load Raw & Features Data
    from dataset import load_raw_ohlcv
    from features import build_features
    from stationary import add_stationary_features
    import config

    stock = config.get_stock(stock_key)
    raw = load_raw_ohlcv(stock.raw_path, verbose=False)
    feat_df = build_features(raw, horizon=config.TARGET_HORIZON, with_ichimoku=True, with_bollinger=True, dropna=False, verbose=False)
    feat_df = add_stationary_features(feat_df).reset_index(drop=True)
    feat_df["Date"] = pd.to_datetime(feat_df["Date"])

    # 3. Classify Regimes
    feat_df = classify_market_regimes(feat_df)

    # 4. Load Sentiment
    news_df = load_clean_news_sentiment(stock_key, base_dir=base_dir)

    # Merge all datasets into evaluation dataframe
    merged = pd.merge(oof, feat_df[[
        "Date", "Close", "Open", "High", "Low", "Volume",
        "SMA_20", "SMA_50", "RSI", "MACD", "MACD_signal",
        "macd_hist_norm", "close_vs_sma20", "close_vs_sma50",
        "realized_vol_20", "market_regime"
    ]], on="Date", how="left")

    if not news_df.empty:
        merged = pd.merge(merged, news_df[["Date", "sentiment_rolling", "sentiment_count"]], on="Date", how="left")
        merged["sentiment_rolling"] = merged["sentiment_rolling"].fillna(0.0)
        merged["sentiment_count"] = merged["sentiment_count"].fillna(0)
    else:
        merged["sentiment_rolling"] = 0.0
        merged["sentiment_count"] = 0

    # Estimate 3-day volatility / typical magnitude return for expected return calculation
    # In 3 days, expected return ~ (2 * proba - 1) * expected_magnitude
    # where expected_magnitude is approximated by realized_vol_20 * sqrt(3)
    merged["vol_3d"] = (merged["realized_vol_20"].fillna(0.015) * np.sqrt(3)).clip(lower=0.01, upper=0.08)
    merged["confidence"] = np.abs(merged["y_proba_calibrated"] - 0.5)
    merged["expected_gross_ret"] = (2.0 * merged["y_proba_calibrated"] - 1.0) * merged["vol_3d"]

    signals = []
    rationales = []
    conf_flags = []
    tech_flags = []
    sent_flags = []
    regime_flags = []

    variant_upper = variant.upper()

    for _, row in merged.iterrows():
        p_up = row["y_proba_calibrated"]
        exp_ret = row["expected_gross_ret"]
        conf = row["confidence"]
        rsi = row["RSI"]
        macd_hist = row["macd_hist_norm"]
        c_sma20 = row["close_vs_sma20"]
        c_sma50 = row["close_vs_sma50"]
        sent = row["sentiment_rolling"]
        regime = row["market_regime"]

        # Base Model Proposal
        # Upward bias if prob > 0.5 and exp_ret > friction
        model_action = 0
        if conf >= confidence_gate:
            if p_up > 0.50 and exp_ret > friction_rate:
                model_action = 1
            elif p_up < 0.50 and (-exp_ret) > friction_rate:
                model_action = -1

        # Technical Confirmation Evaluation
        # Bullish technicals: RSI not overbought (<70), MACD hist >= 0, Close >= SMA20
        # Bearish technicals: RSI not oversold (>30), MACD hist <= 0, Close <= SMA20
        tech_bull = bool(pd.notnull(rsi) and rsi < 70 and pd.notnull(macd_hist) and macd_hist >= 0 and pd.notnull(c_sma20) and c_sma20 >= -0.01)
        tech_bear = bool(pd.notnull(rsi) and rsi > 30 and pd.notnull(macd_hist) and macd_hist <= 0 and pd.notnull(c_sma20) and c_sma20 <= 0.01)

        # Sentiment Confirmation Evaluation
        sent_bull = bool(sent >= SENTIMENT_THRESHOLD)
        sent_bear = bool(sent <= -SENTIMENT_THRESHOLD)
        sent_neutral = bool(abs(sent) < SENTIMENT_THRESHOLD)

        # Regime Evaluation
        is_favorable_bull = regime in ("TRENDING_BULL", "SIDEWAYS_CHOPPY")
        is_favorable_bear = regime in ("TRENDING_BEAR", "SIDEWAYS_CHOPPY")
        is_high_vol = (regime == "HIGH_VOLATILITY")

        final_sig = 0
        reason = "HOLD: Default"

        if variant_upper == "A":
            # Model Alone (with cost hurdle)
            if model_action == 1:
                final_sig = 1
                reason = f"BUY: Model P(up)={p_up:.1%} exceeds cost hurdle ({exp_ret:+.2%} vs {friction_rate:.2%})"
            elif model_action == -1:
                final_sig = -1 if not long_only else 0
                reason = f"SELL: Model P(up)={p_up:.1%} downward return exceeds hurdle ({-exp_ret:+.2%} vs {friction_rate:.2%})"
            else:
                final_sig = 0
                reason = f"HOLD: Insufficient edge (E[R]={exp_ret:+.2%}) to overcome friction ({friction_rate:.2%})"

        elif variant_upper == "B":
            # Model + Technical Confirmation
            if model_action == 1:
                if tech_bull:
                    final_sig = 1
                    reason = f"BUY: Model ({p_up:.1%}) confirmed by RSI ({rsi:.1f}), MACD, and SMA"
                else:
                    final_sig = 0
                    reason = f"HOLD: Model bullish ({p_up:.1%}) but technicals vetoed (RSI={rsi:.1f}, MACD={macd_hist:+.3f})"
            elif model_action == -1:
                if tech_bear:
                    final_sig = -1 if not long_only else 0
                    reason = f"SELL: Model ({p_up:.1%}) confirmed by technical indicators"
                else:
                    final_sig = 0
                    reason = f"HOLD: Model bearish ({p_up:.1%}) but technicals not confirming"
            else:
                final_sig = 0
                reason = "HOLD: No base model signal"

        elif variant_upper == "C":
            # Model + Sentiment Confirmation
            if model_action == 1:
                if sent_bull or (sent_neutral and p_up >= 0.55):
                    final_sig = 1
                    reason = f"BUY: Model ({p_up:.1%}) confirmed by FinBERT sentiment ({sent:+.2f})"
                else:
                    final_sig = 0
                    reason = f"HOLD: Model bullish but negative FinBERT sentiment ({sent:+.2f})"
            elif model_action == -1:
                if sent_bear or (sent_neutral and p_up <= 0.45):
                    final_sig = -1 if not long_only else 0
                    reason = f"SELL: Model ({p_up:.1%}) confirmed by FinBERT sentiment ({sent:+.2f})"
                else:
                    final_sig = 0
                    reason = f"HOLD: Model bearish but positive FinBERT sentiment ({sent:+.2f})"
            else:
                final_sig = 0
                reason = "HOLD: No base model signal"

        elif variant_upper == "D":
            # Multi-Factor Consensus: Model + Technicals + Sentiment
            if model_action == 1 and tech_bull and not sent_bear:
                final_sig = 1
                reason = f"BUY: Multi-factor consensus (Model={p_up:.1%}, Tech=Bull, Sent={sent:+.2f})"
            elif model_action == -1 and tech_bear and not sent_bull:
                final_sig = -1 if not long_only else 0
                reason = f"SELL: Multi-factor consensus (Model={p_up:.1%}, Tech=Bear, Sent={sent:+.2f})"
            else:
                final_sig = 0
                reason = "HOLD: Multi-factor disagreement / partial confirmation"

        elif variant_upper == "E":
            # Regime-Filtered Strategy
            if is_high_vol:
                final_sig = 0
                reason = f"HOLD: High volatility regime veto ({regime})"
            elif regime == "SIDEWAYS_CHOPPY" and conf < 0.08:
                final_sig = 0
                reason = "HOLD: Sideways choppy market with low conviction"
            else:
                if model_action == 1 and tech_bull and is_favorable_bull:
                    final_sig = 1
                    reason = f"BUY: Favorable regime ({regime}) with model and technical confirmation"
                elif model_action == -1 and tech_bear and is_favorable_bear:
                    final_sig = -1 if not long_only else 0
                    reason = f"SELL: Favorable regime ({regime}) with bearish confirmation"
                else:
                    final_sig = 0
                    reason = f"HOLD: Regime ({regime}) conditions not satisfied"
        else:
            raise ValueError(f"Unknown strategy variant: {variant}")

        signals.append(final_sig)
        rationales.append(reason)
        conf_flags.append(model_action != 0)
        tech_flags.append(tech_bull if model_action == 1 else (tech_bear if model_action == -1 else False))
        sent_flags.append(sent_bull if model_action == 1 else (sent_bear if model_action == -1 else False))
        regime_flags.append(regime)

    merged["signal"] = signals
    merged["rationale"] = rationales
    merged["flag_model"] = conf_flags
    merged["flag_tech"] = tech_flags
    merged["flag_sentiment"] = sent_flags
    
    return merged


# ---------------------------------------------------------------------------
# Latest Real-Time Signal Generator (For Terminal Dashboard)
# ---------------------------------------------------------------------------

def get_latest_signal_status(
    stock_key: str,
    variant: str = "A",
    friction_rate: float = DEFAULT_FRICTION_RATE,
    base_dir: Optional[Path] = None,
) -> Dict[str, Union[str, float, bool]]:
    """
    Computes the current actionable signal for the latest market date.
    
    Used by the Streamlit terminal to display the prominent BUY / SELL / HOLD badge,
    expected return, friction hurdle, and multi-factor breakdown.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    # Generate signals for full out-of-sample history to calibrate and get latest row
    df = generate_signals(stock_key, variant=variant, long_only=True, friction_rate=friction_rate, base_dir=base_dir)
    latest = df.iloc[-1]

    sig = int(latest["signal"])
    sig_label = "BUY" if sig == 1 else ("SELL" if sig == -1 else "HOLD")

    p_up = float(latest["y_proba_calibrated"])
    exp_ret = float(latest["expected_gross_ret"])
    conf = float(latest["confidence"])
    
    rsi = float(latest["RSI"]) if pd.notnull(latest["RSI"]) else 50.0
    macd_hist = float(latest["macd_hist_norm"]) if pd.notnull(latest["macd_hist_norm"]) else 0.0
    close_val = float(latest["Close"])
    sma20_val = float(latest["SMA_20"]) if pd.notnull(latest["SMA_20"]) else close_val
    sma50_val = float(latest["SMA_50"]) if pd.notnull(latest["SMA_50"]) else close_val
    sent_score = float(latest["sentiment_rolling"])
    regime = str(latest["market_regime"])

    # Determine net expectancy
    if sig == 1:
        net_exp = exp_ret - friction_rate
    elif sig == -1:
        net_exp = -exp_ret - friction_rate
    else:
        net_exp = 0.0

    return {
        "stock": stock_key,
        "date": str(latest["Date"].date()),
        "signal": sig,
        "signal_label": sig_label,
        "calibrated_proba_up": p_up,
        "confidence": conf,
        "expected_gross_return": exp_ret,
        "friction_rate": friction_rate,
        "net_expectancy": net_exp,
        "rationale": latest["rationale"],
        "close": close_val,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "sma20": sma20_val,
        "sma50": sma50_val,
        "sentiment_score": sent_score,
        "market_regime": regime,
        "flag_model": bool(latest["flag_model"]),
        "flag_tech": bool(latest["flag_tech"]),
        "flag_sentiment": bool(latest["flag_sentiment"]),
    }

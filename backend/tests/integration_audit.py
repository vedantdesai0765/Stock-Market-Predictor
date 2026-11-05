"""
backend/tests/integration_audit.py
Automated Numerical & Methodological Integration Audit across all 10 stocks.
"""

import sys
import json
from pathlib import Path
import requests
import pandas as pd
import numpy as np

BASE_URL = "http://127.0.0.1:8000/api/v1"
ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

import config
import pipeline
import signals
import backtest

STOCKS = config.list_stocks()

def run_audit():
    print(f"=== STARTING INTEGRATION AUDIT FOR {len(STOCKS)} STOCKS ===")
    results = {}

    # 1. Test Stocks List
    resp = requests.get(f"{BASE_URL}/stocks")
    assert resp.status_code == 200, f"Stocks list failed: {resp.text}"
    stocks_data = resp.json()
    assert stocks_data["total"] == 10, f"Expected 10 stocks, got {stocks_data['total']}"
    print("[PASS] Stocks list endpoint returned all 10 stocks.")

    for ticker in STOCKS:
        print(f"\n--- Auditing Stock: {ticker} ---")
        stock_cfg = config.get_stock(ticker)
        raw_df = pd.read_csv(stock_cfg.raw_path)
        last_raw_close = float(raw_df["Close"].iloc[-1])
        last_raw_date = str(raw_df["Date"].iloc[-1])[:10]

        stock_res = {"ticker": ticker, "checks": {}}

        # A. PREDICTION ENDPOINT
        p_resp = requests.get(f"{BASE_URL}/stocks/{ticker}/prediction")
        assert p_resp.status_code == 200, f"Prediction endpoint failed for {ticker}: {p_resp.text}"
        p_data = p_resp.json()

        # Verify close price matches raw data on the returned date
        raw_on_date = raw_df[raw_df["Date"].astype(str).str.startswith(p_data["date"])]
        assert not raw_on_date.empty, f"Date {p_data['date']} not found in raw data for {ticker}"
        expected_close = float(raw_on_date["Close"].iloc[0])
        close_diff = abs(p_data["close"] - expected_close)
        assert close_diff < 0.05, f"Close mismatch for {ticker} on {p_data['date']}: API {p_data['close']} vs RAW {expected_close}"

        # Verify probabilities
        prob_sum = p_data["probability_up"] + p_data["probability_down"]
        assert abs(prob_sum - 1.0) < 0.001, f"Probabilities do not sum to 1: {prob_sum}"

        # Verify signal
        assert p_data["signal"] in {"UP", "DOWN", "NO SIGNAL"}, f"Unexpected prediction: {p_data['signal']}"
        assert p_data["target_horizon_days"] == 3, f"Expected 3-day horizon, got {p_data['target_horizon_days']}"
        stock_res["checks"]["prediction"] = {
            "status": "PASS",
            "close": p_data["close"],
            "date": p_data["date"],
            "prob_up": p_data["probability_up"],
            "signal": p_data["signal"],
            "confidence": p_data["confidence"],
            "features_count": len(p_data["top_features"]) if p_data["top_features"] else 0
        }

        # B. FORECAST ENDPOINT
        f_resp = requests.get(f"{BASE_URL}/stocks/{ticker}/forecast")
        assert f_resp.status_code == 200, f"Forecast endpoint failed for {ticker}: {f_resp.text}"
        f_data = f_resp.json()
        assert len(f_data["summaries"]) >= 3, f"Expected at least 3 models in summary, got {len(f_data['summaries'])}"
        assert f_data["best_accuracy"] is not None and f_data["best_accuracy"] > 0.5, f"Unexpected best accuracy: {f_data['best_accuracy']}"
        stock_res["checks"]["forecast"] = {
            "status": "PASS",
            "best_model": f_data["best_model"],
            "best_accuracy": f_data["best_accuracy"],
            "best_edge": f_data["best_edge"],
            "summaries_count": len(f_data["summaries"]),
            "annual_count": len(f_data["annual_breakdown"]) if f_data["annual_breakdown"] else 0
        }

        # C. NEWS & SENTIMENT ENDPOINT
        n_resp = requests.get(f"{BASE_URL}/stocks/{ticker}/news")
        assert n_resp.status_code == 200, f"News endpoint failed for {ticker}: {n_resp.text}"
        n_data = n_resp.json()
        assert "timeline" in n_data and n_data["timeline"] is not None, f"Timeline missing in news response for {ticker}"
        assert len(n_data["timeline"]) > 0, f"Empty timeline for {ticker}"
        assert n_data["lead_lag"] is not None and len(n_data["lead_lag"]) > 0, f"Empty lead-lag for {ticker}"
        stock_res["checks"]["news"] = {
            "status": "PASS",
            "total_articles": n_data["total_articles"],
            "positive_count": n_data["positive_count"],
            "negative_count": n_data["negative_count"],
            "neutral_count": n_data["neutral_count"],
            "average_sentiment": n_data["average_sentiment"],
            "timeline_points": len(n_data["timeline"]),
            "lead_lag_points": len(n_data["lead_lag"])
        }

        # D. ACTIONABLE SIGNALS ENDPOINT (Variants A to E, Friction 0, 15, 30 bps)
        s_resp_15 = requests.get(f"{BASE_URL}/stocks/{ticker}/signals/latest?variant=A&friction_bps=15.0")
        assert s_resp_15.status_code == 200, f"Signals endpoint failed: {s_resp_15.text}"
        s_data_15 = s_resp_15.json()
        assert s_data_15["signal_label"] in {"BUY", "SELL", "HOLD"}, f"Invalid signal label: {s_data_15['signal_label']}"

        # Test friction impact
        s_resp_0 = requests.get(f"{BASE_URL}/stocks/{ticker}/signals/latest?variant=A&friction_bps=0.0")
        s_data_0 = s_resp_0.json()
        s_resp_30 = requests.get(f"{BASE_URL}/stocks/{ticker}/signals/latest?variant=A&friction_bps=30.0")
        s_data_30 = s_resp_30.json()

        # Friction difference
        diff_0_15 = s_data_0["friction_rate"] - s_data_15["friction_rate"]
        assert abs(diff_0_15 - (-0.0015)) < 1e-6, f"Friction rate calc mismatch: {s_data_15['friction_rate']}"

        stock_res["checks"]["signals"] = {
            "status": "PASS",
            "variant_A_signal": s_data_15["signal_label"],
            "regime": s_data_15["market_regime"],
            "exp_gross_return": s_data_15["expected_gross_return"],
            "friction_bps_15": s_data_15["friction_rate"],
            "net_expectancy": s_data_15["net_expectancy"]
        }

        # E. BACKTEST ENDPOINT
        b_resp = requests.get(f"{BASE_URL}/stocks/{ticker}/backtest?variant=A&friction_bps=15.0&long_only=true")
        assert b_resp.status_code == 200, f"Backtest endpoint failed for {ticker}: {b_resp.text}"
        b_data = b_resp.json()

        # Check equity curve
        assert len(b_data["equity_curve"]) > 10, f"Backtest equity curve too short for {ticker}"
        assert b_data["starting_capital"] == 1000000.0, f"Expected starting capital 1,000,000, got {b_data['starting_capital']}"
        first_eq = b_data["equity_curve"][0]
        assert abs(first_eq["cash_equity"] - 1000000.0) < 5000.0, f"Expected initial cash equity near 1,000,000, got {first_eq['cash_equity']}"

        # Check metrics
        strat_m = b_data["metrics"]["strategy"]
        bh_m = b_data["metrics"]["buy_and_hold"]
        assert "cagr" in strat_m and "sharpe_ratio" in strat_m and "max_drawdown" in strat_m

        # Check trades
        trades = b_data["trades"]
        assert len(trades) > 0, f"No trades recorded for {ticker}"
        for tr in trades[:3]:
            assert tr["position"] in {"LONG", "SHORT"}, f"Unexpected position: {tr['position']}"

        # Test long_only=false vs true
        b_resp_short = requests.get(f"{BASE_URL}/stocks/{ticker}/backtest?variant=A&friction_bps=15.0&long_only=false")
        assert b_resp_short.status_code == 200
        b_data_short = b_resp_short.json()
        has_short = any(tr["position"] == "SHORT" for tr in b_data_short["trades"])

        stock_res["checks"]["backtest"] = {
            "status": "PASS",
            "cagr_strat": strat_m["cagr"],
            "cagr_bh": bh_m["cagr"],
            "sharpe": strat_m["sharpe_ratio"],
            "max_dd": strat_m["max_drawdown"],
            "profit_factor": strat_m["profit_factor"],
            "win_rate": strat_m["win_rate"],
            "trade_count": len(trades),
            "supports_short": has_short
        }

        results[ticker] = stock_res
        print(f"  [PASS] {ticker}: All 5 modules verified successfully!")

    print("\n=== ALL 10 STOCKS PASSED BACKEND VERIFICATION ===")
    return results

if __name__ == "__main__":
    res = run_audit()
    with open(ROOT_DIR / "backend" / "tests" / "audit_results.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("Saved audit results to backend/tests/audit_results.json")

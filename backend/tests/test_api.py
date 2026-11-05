"""
backend/tests/test_api.py
Automated end-to-end integration and numerical parity tests for FastAPI backend.
"""

import pytest
from fastapi.testclient import TestClient

import config
import pipeline
import backtest


def test_health(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["registered_stocks"] == 10
    assert data["api_prefix"] == "/api/v1"


def test_stocks_list(client: TestClient):
    response = client.get("/api/v1/stocks")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 10
    assert len(data["stocks"]) == 10
    tickers = [s["ticker"] for s in data["stocks"]]
    expected = [
        "ADANIENT", "ADANIGREEN", "ETERNAL", "HDFCBANK", "INFY",
        "ITC", "RELIANCE", "SBI", "TCS", "VEDL"
    ]
    for exp in expected:
        assert exp in tickers


def test_stock_detail(client: TestClient):
    response = client.get("/api/v1/stocks/TCS")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TCS"
    assert data["symbol"] == "TCS.NS"
    assert data["display_name"] == "Tata Consultancy Services"
    assert data["has_model"] is True
    assert data["has_raw_data"] is True


def test_stock_not_found(client: TestClient):
    response = client.get("/api/v1/stocks/INVALID_TICKER")
    assert response.status_code == 404
    data = response.json()
    assert data["success"] is False
    assert data["error_code"] == "STOCK_NOT_FOUND"


def test_prediction_endpoint_and_numerical_parity(client: TestClient):
    response = client.get("/api/v1/stocks/TCS/prediction")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TCS"
    assert data["target_horizon_days"] == 3
    assert data["signal"] in ["UP", "DOWN", "NO SIGNAL"]
    assert 0.0 <= data["probability_up"] <= 1.0
    assert 0.0 <= data["confidence"] <= 0.5
    assert len(data["top_features"]) > 0

    # Numerical Parity Check with direct pipeline call
    direct_pred = pipeline.predict_latest("TCS")
    assert data["probability_up"] == pytest.approx(direct_pred["probability_up"], abs=1e-4)
    assert data["confidence"] == pytest.approx(direct_pred["confidence"], abs=1e-4)
    assert data["signal"] == direct_pred["signal"]
    assert data["close"] == pytest.approx(direct_pred["close"], abs=1e-2)


def test_forecast_endpoint(client: TestClient):
    response = client.get("/api/v1/stocks/TCS/forecast")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TCS"
    assert len(data["summaries"]) > 0
    assert data["best_model"] is not None
    assert data["best_accuracy"] is not None
    assert data["annual_breakdown"] is not None
    assert len(data["annual_breakdown"]) > 0


def test_signals_latest_endpoint(client: TestClient):
    response = client.get("/api/v1/stocks/TCS/signals/latest?variant=A&friction_bps=15.0")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TCS"
    assert data["signal"] in [-1, 0, 1]
    assert data["signal_label"] in ["BUY", "SELL", "HOLD"]
    assert 0.0 <= data["calibrated_proba_up"] <= 1.0
    assert data["friction_rate"] == 0.0015
    assert len(data["rationale"]) > 0
    assert data["market_regime"] in ["TRENDING_BULL", "TRENDING_BEAR", "HIGH_VOLATILITY", "SIDEWAYS_CHOPPY"]


def test_signals_invalid_variant(client: TestClient):
    response = client.get("/api/v1/stocks/TCS/signals/latest?variant=INVALID")
    assert response.status_code == 400
    data = response.json()
    assert data["error_code"] == "INVALID_VARIANT"


def test_backtest_endpoint_and_numerical_parity(client: TestClient):
    response = client.get("/api/v1/stocks/TCS/backtest?variant=A&long_only=true&friction_bps=15.0")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TCS"
    assert data["variant"] == "A"
    assert data["long_only"] is True
    assert len(data["equity_curve"]) > 50
    assert "strategy" in data["metrics"]
    assert "buy_and_hold" in data["metrics"]
    assert "cash" in data["metrics"]

    # Numerical Parity Check with direct backtest engine
    direct_bt = backtest.run_walk_forward_backtest("TCS", variant="A", long_only=True, friction_rate=0.0015)
    direct_metrics = direct_bt["metrics"]["strategy"]
    api_metrics = data["metrics"]["strategy"]

    assert api_metrics["cagr"] == pytest.approx(direct_metrics["cagr"], abs=1e-4)
    assert api_metrics["sharpe_ratio"] == pytest.approx(direct_metrics["sharpe_ratio"], abs=1e-4)
    assert api_metrics["max_drawdown"] == pytest.approx(direct_metrics["max_drawdown"], abs=1e-4)
    assert api_metrics["total_trades"] == direct_metrics["total_trades"]


def test_data_status_endpoint(client: TestClient):
    response = client.get("/api/v1/stocks/TCS/data-status")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TCS"
    assert data["raw_data_exists"] is True
    assert data["model_exists"] is True
    assert data["raw_data_last_date"] is not None
    assert data["oof_eval_last_date"] is not None
    assert data["model_trained_at"] is not None
    assert data["is_live_market"] is False
    assert "Historical research snapshot" in data["status_summary"]


def test_news_endpoint(client: TestClient):
    response = client.get("/api/v1/stocks/TCS/news?limit=20")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TCS"
    assert data["total_articles"] > 0
    assert len(data["recent_articles"]) <= 20
    assert data["lead_lag"] is not None


def test_all_stocks_data_status(client: TestClient):
    stocks = config.list_stocks()
    for s in stocks:
        resp = client.get(f"/api/v1/stocks/{s}/data-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == s
        assert data["raw_data_exists"] is True

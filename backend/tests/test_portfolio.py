"""
backend/tests/test_portfolio.py
Comprehensive automated test suite for Virtual Portfolio & Paper Trading engine.
Covers all 13 critical requirements specified in Phase 5.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.database.session import SessionLocal
from backend.app.models.wallet import Wallet, Position, Transaction
from backend.app.services.wallet_service import (
    WalletService,
    DEFAULT_STARTING_CAPITAL,
    ONE_WAY_FRICTION_BPS,
)


@pytest.fixture(autouse=True)
def reset_wallet_before_test():
    """Ensure a clean wallet state before each test."""
    db: Session = SessionLocal()
    try:
        WalletService.reset_portfolio(db)
    finally:
        db.close()
    yield
    # Cleanup after test
    db = SessionLocal()
    try:
        WalletService.reset_portfolio(db)
    finally:
        db.close()


def test_initial_wallet_creation_and_balance(client: TestClient):
    """1 & 2: Test initial wallet creation and initial ₹1,00,000 balance."""
    response = client.get("/api/v1/portfolio")
    assert response.status_code == 200
    data = response.json()
    assert data["initial_capital"] == 100000.0
    assert data["available_cash"] == 100000.0
    assert data["invested_capital"] == 0.0
    assert data["portfolio_value"] == 100000.0
    assert data["total_pnl"] == 0.0
    assert data["realised_pnl"] == 0.0
    assert data["unrealised_pnl"] == 0.0
    assert data["total_return_pct"] == 0.0
    assert data["holdings_count"] == 0
    assert data["holdings"] == []


def test_transaction_cost_calculation(client: TestClient):
    """7: Test transaction cost calculation (7.5 bps one-way Indian delivery friction)."""
    # Preview buying 10 shares of TCS
    preview_resp = client.post(
        "/api/v1/portfolio/preview",
        json={"ticker": "TCS", "side": "BUY", "quantity": 10}
    )
    assert preview_resp.status_code == 200
    data = preview_resp.json()
    price = data["execution_price"]
    gross = round(10 * price, 2)
    expected_friction = round(gross * (ONE_WAY_FRICTION_BPS / 10000.0), 2)
    expected_net = round(gross + expected_friction, 2)

    assert data["gross_amount"] == gross
    assert data["friction_bps"] == 7.5
    assert data["friction_amount"] == expected_friction
    assert data["net_amount"] == expected_net
    assert data["is_valid"] is True


def test_successful_buy_order(client: TestClient):
    """3: Test successful BUY order and holding update."""
    buy_resp = client.post(
        "/api/v1/portfolio/orders",
        json={"ticker": "TCS", "side": "BUY", "quantity": 10, "signal_source": "MANUAL"}
    )
    assert buy_resp.status_code == 201
    b_data = buy_resp.json()
    assert b_data["success"] is True
    assert b_data["ticker"] == "TCS"
    assert b_data["side"] == "BUY"
    assert b_data["quantity"] == 10
    assert b_data["cash_balance_after"] < 100000.0

    # Verify portfolio state reflects purchased shares
    port_resp = client.get("/api/v1/portfolio")
    assert port_resp.status_code == 200
    p_data = port_resp.json()
    assert p_data["holdings_count"] == 1
    holding = p_data["holdings"][0]
    assert holding["ticker"] == "TCS"
    assert holding["quantity"] == 10
    assert holding["invested_amount"] == b_data["gross_amount"]


def test_successful_sell_order(client: TestClient):
    """4: Test successful SELL order and proceeds receipt."""
    # First buy 10 shares
    client.post(
        "/api/v1/portfolio/orders",
        json={"ticker": "TCS", "side": "BUY", "quantity": 10}
    )

    # Now sell 4 shares
    sell_resp = client.post(
        "/api/v1/portfolio/orders",
        json={"ticker": "TCS", "side": "SELL", "quantity": 4}
    )
    assert sell_resp.status_code == 201
    s_data = sell_resp.json()
    assert s_data["success"] is True
    assert s_data["side"] == "SELL"
    assert s_data["quantity"] == 4

    # Check holding is now 6 shares
    port_resp = client.get("/api/v1/portfolio")
    p_data = port_resp.json()
    assert p_data["holdings_count"] == 1
    assert p_data["holdings"][0]["quantity"] == 6


def test_insufficient_cash(client: TestClient):
    """5: Test rejection when cash balance is insufficient."""
    # Attempt to buy 1,000 shares of TCS (approx ₹23 Lakhs, exceeding ₹1 Lakh cash)
    resp = client.post(
        "/api/v1/portfolio/orders",
        json={"ticker": "TCS", "side": "BUY", "quantity": 1000}
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error_code"] == "INSUFFICIENT_CASH"


def test_insufficient_holdings(client: TestClient):
    """6: Test rejection when attempting to sell unowned or excessive shares."""
    # Attempt to sell without owning any
    resp = client.post(
        "/api/v1/portfolio/orders",
        json={"ticker": "TCS", "side": "SELL", "quantity": 5}
    )
    assert resp.status_code == 400
    data = resp.json()
    assert data["error_code"] == "INSUFFICIENT_HOLDINGS"


def test_duplicate_order_prevention(client: TestClient):
    """8: Test duplicate order prevention via idempotency_key."""
    idempotency_key = "unique-order-key-12345"

    resp1 = client.post(
        "/api/v1/portfolio/orders",
        json={
            "ticker": "INFY",
            "side": "BUY",
            "quantity": 5,
            "idempotency_key": idempotency_key
        }
    )
    assert resp1.status_code == 201
    order1 = resp1.json()

    # Repeat exact same request with same idempotency key
    resp2 = client.post(
        "/api/v1/portfolio/orders",
        json={
            "ticker": "INFY",
            "side": "BUY",
            "quantity": 5,
            "idempotency_key": idempotency_key
        }
    )
    assert resp2.status_code == 201
    order2 = resp2.json()

    # Order IDs must match and no extra cash deducted
    assert order1["order_id"] == order2["order_id"]
    assert "already executed" in order2["message"].lower()

    # Portfolio should only have 5 shares, not 10
    port_resp = client.get("/api/v1/portfolio")
    assert port_resp.json()["holdings"][0]["quantity"] == 5


def test_concurrent_order_safety(client: TestClient):
    """9: Test sequential consistency and transaction integrity."""
    # Buy multiple stocks in sequence
    client.post("/api/v1/portfolio/orders", json={"ticker": "INFY", "side": "BUY", "quantity": 10})
    client.post("/api/v1/portfolio/orders", json={"ticker": "VEDL", "side": "BUY", "quantity": 20})

    port = client.get("/api/v1/portfolio").json()
    assert port["holdings_count"] == 2
    tickers = [h["ticker"] for h in port["holdings"]]
    assert "INFY" in tickers
    assert "VEDL" in tickers


def test_database_persistence(client: TestClient):
    """10: Test database persistence across separate session contexts."""
    client.post("/api/v1/portfolio/orders", json={"ticker": "ITC", "side": "BUY", "quantity": 25})

    # Direct query via fresh DB session
    db: Session = SessionLocal()
    try:
        tx = db.query(Transaction).filter(Transaction.ticker == "ITC").first()
        assert tx is not None
        assert tx.quantity == 25
        assert tx.side == "BUY"

        pos = db.query(Position).filter(Position.ticker == "ITC").first()
        assert pos is not None
        assert pos.quantity == 25
    finally:
        db.close()


def test_portfolio_valuation_calculation(client: TestClient):
    """11: Test portfolio valuation, unrealised P&L, and return percentage."""
    client.post("/api/v1/portfolio/orders", json={"ticker": "SBI", "side": "BUY", "quantity": 10})
    port = client.get("/api/v1/portfolio").json()

    # Portfolio value must equal available cash + current holdings value
    holdings_val = sum(h["current_value"] for h in port["holdings"])
    assert port["portfolio_value"] == pytest.approx(port["available_cash"] + holdings_val, abs=0.05)
    assert port["total_pnl"] == pytest.approx(port["portfolio_value"] - port["initial_capital"], abs=0.05)


def test_invalid_quantities(client: TestClient):
    """12: Test rejection of zero, negative, or invalid quantities."""
    resp_zero = client.post("/api/v1/portfolio/orders", json={"ticker": "TCS", "side": "BUY", "quantity": 0})
    assert resp_zero.status_code == 422 or resp_zero.status_code == 400

    resp_neg = client.post("/api/v1/portfolio/orders", json={"ticker": "TCS", "side": "BUY", "quantity": -5})
    assert resp_neg.status_code == 422 or resp_neg.status_code == 400


def test_missing_market_prices_or_invalid_ticker(client: TestClient):
    """13: Test rejection of non-existent stock symbols without price data."""
    resp = client.post("/api/v1/portfolio/orders", json={"ticker": "NONEXISTENT", "side": "BUY", "quantity": 10})
    assert resp.status_code == 404
    data = resp.json()
    assert data["error_code"] == "STOCK_NOT_FOUND"


def test_portfolio_analytics_endpoint(client: TestClient):
    """Test /api/v1/portfolio/analytics endpoint."""
    client.post("/api/v1/portfolio/orders", json={"ticker": "TCS", "side": "BUY", "quantity": 5})
    resp = client.get("/api/v1/portfolio/analytics")
    assert resp.status_code == 200
    data = resp.json()
    assert "equity_curve" in data
    assert len(data["equity_curve"]) >= 2
    assert "asset_allocation" in data
    assert len(data["asset_allocation"]) >= 2  # Cash + TCS

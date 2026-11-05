"""
backend/app/schemas/wallet.py
Pydantic schemas for Virtual Wallet, paper trading, holdings, and portfolio analytics.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class PositionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    wallet_id: str
    ticker: str
    quantity: int
    avg_buy_price: float
    total_cost: float
    updated_at: datetime


class HoldingItem(BaseModel):
    ticker: str
    display_name: str
    quantity: int
    avg_buy_price: float
    current_price: float
    price_date: str
    invested_amount: float
    current_value: float
    unrealised_pnl: float
    return_pct: float


class TransactionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    wallet_id: str
    ticker: str
    side: str
    quantity: int
    execution_price: float
    gross_amount: float
    friction_bps: float
    friction_amount: float
    net_amount: float
    signal_source: Optional[str] = "MANUAL"
    idempotency_key: Optional[str] = None
    executed_at: datetime


class WalletSummaryResponse(BaseModel):
    id: str
    name: str
    currency: str
    initial_capital: float
    available_cash: float
    invested_capital: float
    portfolio_value: float
    total_pnl: float
    realised_pnl: float
    unrealised_pnl: float
    total_return_pct: float
    holdings: List[HoldingItem]
    holdings_count: int
    recent_transactions: List[TransactionSchema]


class OrderPreviewRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol key, e.g. TCS")
    side: str = Field(..., description="'BUY' or 'SELL'")
    quantity: int = Field(..., gt=0, description="Number of shares to trade")


class OrderPreviewResponse(BaseModel):
    ticker: str
    display_name: str
    side: str
    quantity: int
    execution_price: float
    price_date: str
    gross_amount: float
    friction_bps: float
    friction_amount: float
    net_amount: float
    available_cash: float
    owned_quantity: int
    is_valid: bool
    validation_error: Optional[str] = None
    max_shares_buyable: int


class ExecuteOrderRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol key, e.g. TCS")
    side: str = Field(..., description="'BUY' or 'SELL'")
    quantity: int = Field(..., gt=0, description="Number of shares to trade")
    signal_source: Optional[str] = Field("MANUAL", description="Source of trade signal, e.g. SIGNAL_ENGINE_A")
    idempotency_key: Optional[str] = Field(None, description="Optional unique key to prevent duplicate submissions")


class ExecuteOrderResponse(BaseModel):
    success: bool
    order_id: str
    ticker: str
    display_name: str
    side: str
    quantity: int
    execution_price: float
    price_date: str
    gross_amount: float
    friction_amount: float
    net_amount: float
    cash_balance_after: float
    message: str


class PortfolioEquityPoint(BaseModel):
    timestamp: str
    portfolio_value: float
    cash: float
    invested: float


class AssetAllocationItem(BaseModel):
    name: str
    ticker: str
    value: float
    percentage: float
    color: str


class PortfolioAnalyticsResponse(BaseModel):
    equity_curve: List[PortfolioEquityPoint]
    asset_allocation: List[AssetAllocationItem]
    cash: float
    invested: float
    portfolio_value: float
    realised_pnl: float
    unrealised_pnl: float
    total_pnl: float
    total_trades: int
    winning_trades: int
    win_rate: float

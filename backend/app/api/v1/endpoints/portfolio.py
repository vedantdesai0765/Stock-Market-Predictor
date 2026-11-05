"""
backend/app/api/v1/endpoints/portfolio.py
FastAPI router for Virtual Wallet, Paper Trading, Holdings, and Portfolio Analytics.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.wallet import (
    ExecuteOrderRequest,
    ExecuteOrderResponse,
    OrderPreviewRequest,
    OrderPreviewResponse,
    PortfolioAnalyticsResponse,
    TransactionSchema,
    WalletSummaryResponse,
)
from backend.app.services.wallet_service import WalletService

router = APIRouter(prefix="/portfolio", tags=["Virtual Portfolio & Paper Trading"])


@router.get("", response_model=WalletSummaryResponse, summary="Get Virtual Portfolio Summary")
def get_portfolio(db: Session = Depends(get_db)):
    """
    Returns current wallet capital, available cash, total portfolio valuation,
    active equity holdings, real-time unrealised & realised P&L, and return percentage.
    """
    return WalletService.get_portfolio_summary(db)


@router.post("/preview", response_model=OrderPreviewResponse, summary="Preview Order & Calculate Friction")
def preview_order(
    request: OrderPreviewRequest,
    db: Session = Depends(get_db),
):
    """
    Pre-computes trade price, statutory Indian delivery friction (7.5 bps one-way),
    net purchase amount or proceeds, and cash/holding validation without placing an order.
    """
    return WalletService.preview_order(request, db)


@router.post("/orders", response_model=ExecuteOrderResponse, status_code=status.HTTP_201_CREATED, summary="Execute Paper Trading Order")
def execute_order(
    request: ExecuteOrderRequest,
    db: Session = Depends(get_db),
):
    """
    Executes a simulated BUY or SELL order atomically on the backend.
    Enforces cash balance checks, holding ownership checks, and idempotency prevention.
    """
    return WalletService.execute_order(request, db)


@router.get("/transactions", response_model=List[TransactionSchema], summary="Get Transaction Ledger")
def get_transactions(
    limit: int = Query(50, ge=1, le=200, description="Max transactions to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
):
    """
    Returns the auditable chronological transaction ledger for the paper trading wallet.
    """
    wallet = WalletService.get_or_create_default_wallet(db)
    from backend.app.models.wallet import Transaction
    txs = db.query(Transaction).filter(
        Transaction.wallet_id == wallet.id
    ).order_by(Transaction.executed_at.desc()).offset(offset).limit(limit).all()
    return [TransactionSchema.model_validate(tx) for tx in txs]


@router.get("/analytics", response_model=PortfolioAnalyticsResponse, summary="Get Portfolio Analytics")
def get_analytics(db: Session = Depends(get_db)):
    """
    Generates historical portfolio equity progression, asset allocation breakdown,
    and cumulative trade performance metrics.
    """
    return WalletService.get_analytics(db)


@router.post("/reset", summary="Reset Virtual Wallet")
def reset_portfolio(db: Session = Depends(get_db)):
    """
    Resets the simulated trading wallet to ₹1,00,000 cash, purging holdings and trade history.
    """
    return WalletService.reset_portfolio(db)

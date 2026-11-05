"""
backend/app/services/wallet_service.py
Service layer for Virtual Wallet, paper trading execution, position rebalancing, and portfolio analytics.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.errors import StockNotFoundError, StockMarketPredictorException
from backend.app.models.wallet import User, Wallet, Position, Transaction
from backend.app.schemas.wallet import (
    AssetAllocationItem,
    ExecuteOrderRequest,
    ExecuteOrderResponse,
    HoldingItem,
    OrderPreviewRequest,
    OrderPreviewResponse,
    PortfolioAnalyticsResponse,
    PortfolioEquityPoint,
    TransactionSchema,
    WalletSummaryResponse,
)

# Ensure src/ is importable
if str(settings.SRC_DIR) not in sys.path:
    sys.path.insert(0, str(settings.SRC_DIR))
if str(settings.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(settings.ROOT_DIR))

import config

DEFAULT_USER_EMAIL = "trader@stockmarketpredictor.local"
DEFAULT_USERNAME = "InstitutionalTrader"
DEFAULT_STARTING_CAPITAL = 100000.0  # ₹1,00,000 (1 Lakh)
ONE_WAY_FRICTION_BPS = 7.5  # 7.5 bps one-way Indian delivery friction (15 bps round-trip)

COLOR_PALETTE = [
    "#2563EB", "#0D9488", "#16A34A", "#D97706", "#9333EA",
    "#E11D48", "#0891B2", "#4F46E5", "#CA8A04", "#059669"
]


class WalletService:
    @staticmethod
    def get_latest_price_and_date(ticker: str) -> Tuple[float, str]:
        """
        Retrieves the latest genuine market price and date from historical datasets.
        Does NOT invent synthetic values.
        """
        key = ticker.strip().upper()
        if key not in config.STOCKS:
            raise StockNotFoundError(ticker=ticker)

        stock = config.get_stock(key)
        if not stock.raw_path.exists():
            raise StockMarketPredictorException(
                message=f"Market data file not found for {key}",
                status_code=404,
                error_code="DATA_NOT_FOUND"
            )

        df = pd.read_csv(stock.raw_path)
        if df.empty or "Close" not in df.columns:
            raise StockMarketPredictorException(
                message=f"Valid price data missing for {key}",
                status_code=500,
                error_code="INVALID_DATA"
            )

        last_row = df.iloc[-1]
        price = round(float(last_row["Close"]), 2)
        date_str = str(last_row["Date"])[:10]
        return price, date_str

    @staticmethod
    def get_or_create_default_wallet(db: Session) -> Wallet:
        """
        Ensures a primary simulated trading wallet exists with ₹1,00,000 capital.
        """
        user = db.query(User).filter(User.email == DEFAULT_USER_EMAIL).first()
        if not user:
            user = User(email=DEFAULT_USER_EMAIL, username=DEFAULT_USERNAME)
            db.add(user)
            db.flush()

        wallet = db.query(Wallet).filter(Wallet.user_id == user.id).first()
        if not wallet:
            wallet = Wallet(
                user_id=user.id,
                name="Primary Paper Trading Wallet",
                cash_balance=DEFAULT_STARTING_CAPITAL,
                starting_capital=DEFAULT_STARTING_CAPITAL,
                currency="INR"
            )
            db.add(wallet)
            db.commit()
            db.refresh(wallet)

        return wallet

    @staticmethod
    def get_portfolio_summary(db: Session) -> WalletSummaryResponse:
        """
        Calculates complete portfolio valuation, available cash, open holdings,
        and real-time unrealised and realised P&L.
        """
        wallet = WalletService.get_or_create_default_wallet(db)
        positions = db.query(Position).filter(
            Position.wallet_id == wallet.id,
            Position.quantity > 0
        ).all()

        holdings: List[HoldingItem] = []
        invested_capital = 0.0
        current_holdings_value = 0.0
        total_unrealised_pnl = 0.0

        for pos in positions:
            price, price_date = WalletService.get_latest_price_and_date(pos.ticker)
            stock = config.get_stock(pos.ticker)
            curr_val = round(pos.quantity * price, 2)
            unreal_pnl = round(curr_val - pos.total_cost, 2)
            ret_pct = round((unreal_pnl / pos.total_cost) * 100.0, 2) if pos.total_cost > 0 else 0.0

            holdings.append(
                HoldingItem(
                    ticker=pos.ticker,
                    display_name=stock.display_name,
                    quantity=pos.quantity,
                    avg_buy_price=pos.avg_buy_price,
                    current_price=price,
                    price_date=price_date,
                    invested_amount=pos.total_cost,
                    current_value=curr_val,
                    unrealised_pnl=unreal_pnl,
                    return_pct=ret_pct
                )
            )
            invested_capital += pos.total_cost
            current_holdings_value += curr_val
            total_unrealised_pnl += unreal_pnl

        # Total portfolio valuation
        portfolio_value = round(wallet.cash_balance + current_holdings_value, 2)
        total_pnl = round(portfolio_value - wallet.starting_capital, 2)
        total_unrealised_pnl = round(total_unrealised_pnl, 2)
        realised_pnl = round(total_pnl - total_unrealised_pnl, 2)
        total_ret_pct = round((total_pnl / wallet.starting_capital) * 100.0, 2) if wallet.starting_capital > 0 else 0.0

        # Recent transactions
        recent_txs = db.query(Transaction).filter(
            Transaction.wallet_id == wallet.id
        ).order_by(Transaction.executed_at.desc()).limit(20).all()

        tx_schemas = [TransactionSchema.model_validate(tx) for tx in recent_txs]

        return WalletSummaryResponse(
            id=wallet.id,
            name=wallet.name,
            currency=wallet.currency,
            initial_capital=wallet.starting_capital,
            available_cash=round(wallet.cash_balance, 2),
            invested_capital=round(invested_capital, 2),
            portfolio_value=portfolio_value,
            total_pnl=total_pnl,
            realised_pnl=realised_pnl,
            unrealised_pnl=total_unrealised_pnl,
            total_return_pct=total_ret_pct,
            holdings=holdings,
            holdings_count=len(holdings),
            recent_transactions=tx_schemas
        )

    @staticmethod
    def preview_order(request: OrderPreviewRequest, db: Session) -> OrderPreviewResponse:
        """
        Pre-computes price, statutory friction, net amount, and validation status without placing an order.
        """
        ticker = request.ticker.strip().upper()
        if ticker not in config.STOCKS:
            raise StockNotFoundError(ticker=ticker)

        if request.quantity <= 0:
            raise StockMarketPredictorException(
                message="Quantity must be greater than zero",
                status_code=400,
                error_code="INVALID_QUANTITY"
            )

        side = request.side.strip().upper()
        if side not in {"BUY", "SELL"}:
            raise StockMarketPredictorException(
                message="Side must be 'BUY' or 'SELL'",
                status_code=400,
                error_code="INVALID_SIDE"
            )

        wallet = WalletService.get_or_create_default_wallet(db)
        price, price_date = WalletService.get_latest_price_and_date(ticker)
        stock = config.get_stock(ticker)

        gross_amount = round(request.quantity * price, 2)
        friction_rate = ONE_WAY_FRICTION_BPS / 10000.0
        friction_amount = round(gross_amount * friction_rate, 2)

        existing_pos = db.query(Position).filter(
            Position.wallet_id == wallet.id,
            Position.ticker == ticker
        ).first()
        owned_quantity = existing_pos.quantity if existing_pos else 0

        is_valid = True
        validation_error = None
        max_shares = 0

        if side == "BUY":
            net_amount = round(gross_amount + friction_amount, 2)
            if wallet.cash_balance < net_amount:
                is_valid = False
                validation_error = f"Insufficient funds. Required ₹{net_amount:,.2f}, Available: ₹{wallet.cash_balance:,.2f}"
            unit_cost = price * (1.0 + friction_rate)
            max_shares = int(wallet.cash_balance / unit_cost) if unit_cost > 0 else 0
        else:  # SELL
            net_amount = round(gross_amount - friction_amount, 2)
            if owned_quantity < request.quantity:
                is_valid = False
                validation_error = f"Insufficient shares. Attempting to sell {request.quantity}, but only {owned_quantity} owned."
            max_shares = owned_quantity

        return OrderPreviewResponse(
            ticker=ticker,
            display_name=stock.display_name,
            side=side,
            quantity=request.quantity,
            execution_price=price,
            price_date=price_date,
            gross_amount=gross_amount,
            friction_bps=ONE_WAY_FRICTION_BPS,
            friction_amount=friction_amount,
            net_amount=net_amount,
            available_cash=round(wallet.cash_balance, 2),
            owned_quantity=owned_quantity,
            is_valid=is_valid,
            validation_error=validation_error,
            max_shares_buyable=max_shares
        )

    @staticmethod
    def execute_order(request: ExecuteOrderRequest, db: Session) -> ExecuteOrderResponse:
        """
        Executes a paper trading order atomically with balance and holdings checks.
        Prevents duplicate orders via idempotency keys.
        """
        ticker = request.ticker.strip().upper()
        if ticker not in config.STOCKS:
            raise StockNotFoundError(ticker=ticker)

        if request.quantity <= 0:
            raise StockMarketPredictorException(
                message="Quantity must be a positive integer",
                status_code=400,
                error_code="INVALID_QUANTITY"
            )

        side = request.side.strip().upper()
        if side not in {"BUY", "SELL"}:
            raise StockMarketPredictorException(
                message="Side must be 'BUY' or 'SELL'",
                status_code=400,
                error_code="INVALID_SIDE"
            )

        # Idempotency check: prevent duplicate submissions
        if request.idempotency_key:
            existing_tx = db.query(Transaction).filter(
                Transaction.idempotency_key == request.idempotency_key
            ).first()
            if existing_tx:
                wallet = db.query(Wallet).filter(Wallet.id == existing_tx.wallet_id).first()
                stock = config.get_stock(existing_tx.ticker)
                return ExecuteOrderResponse(
                    success=True,
                    order_id=existing_tx.id,
                    ticker=existing_tx.ticker,
                    display_name=stock.display_name,
                    side=existing_tx.side,
                    quantity=existing_tx.quantity,
                    execution_price=existing_tx.execution_price,
                    price_date=str(existing_tx.executed_at)[:10],
                    gross_amount=existing_tx.gross_amount,
                    friction_amount=existing_tx.friction_amount,
                    net_amount=existing_tx.net_amount,
                    cash_balance_after=round(wallet.cash_balance, 2) if wallet else 0.0,
                    message="Order already executed (idempotency key matched)"
                )

        wallet = WalletService.get_or_create_default_wallet(db)
        price, price_date = WalletService.get_latest_price_and_date(ticker)
        stock = config.get_stock(ticker)

        gross_amount = round(request.quantity * price, 2)
        friction_rate = ONE_WAY_FRICTION_BPS / 10000.0
        friction_amount = round(gross_amount * friction_rate, 2)

        try:
            if side == "BUY":
                net_amount = round(gross_amount + friction_amount, 2)
                if wallet.cash_balance < net_amount:
                    raise StockMarketPredictorException(
                        message=f"Insufficient cash balance. Required: ₹{net_amount:,.2f}, Available: ₹{wallet.cash_balance:,.2f}",
                        status_code=400,
                        error_code="INSUFFICIENT_CASH"
                    )

                wallet.cash_balance = round(wallet.cash_balance - net_amount, 2)

                pos = db.query(Position).filter(
                    Position.wallet_id == wallet.id,
                    Position.ticker == ticker
                ).first()

                if pos:
                    new_quantity = pos.quantity + request.quantity
                    new_total_cost = round(pos.total_cost + gross_amount, 2)
                    pos.quantity = new_quantity
                    pos.total_cost = new_total_cost
                    pos.avg_buy_price = round(new_total_cost / new_quantity, 2)
                else:
                    pos = Position(
                        wallet_id=wallet.id,
                        ticker=ticker,
                        quantity=request.quantity,
                        avg_buy_price=price,
                        total_cost=gross_amount
                    )
                    db.add(pos)

                msg = f"Successfully purchased {request.quantity} shares of {ticker} at ₹{price:,.2f}"

            else:  # SELL
                net_amount = round(gross_amount - friction_amount, 2)
                pos = db.query(Position).filter(
                    Position.wallet_id == wallet.id,
                    Position.ticker == ticker
                ).first()

                if not pos or pos.quantity < request.quantity:
                    available = pos.quantity if pos else 0
                    raise StockMarketPredictorException(
                        message=f"Insufficient share quantity to sell. Attempted: {request.quantity}, Owned: {available}",
                        status_code=400,
                        error_code="INSUFFICIENT_HOLDINGS"
                    )

                cost_of_sold = round((pos.total_cost / pos.quantity) * request.quantity, 2)
                pos.quantity -= request.quantity
                pos.total_cost = round(pos.total_cost - cost_of_sold, 2)

                if pos.quantity == 0:
                    db.delete(pos)
                else:
                    pos.avg_buy_price = round(pos.total_cost / pos.quantity, 2)

                wallet.cash_balance = round(wallet.cash_balance + net_amount, 2)
                msg = f"Successfully sold {request.quantity} shares of {ticker} at ₹{price:,.2f}"

            tx = Transaction(
                wallet_id=wallet.id,
                ticker=ticker,
                side=side,
                quantity=request.quantity,
                execution_price=price,
                gross_amount=gross_amount,
                friction_bps=ONE_WAY_FRICTION_BPS,
                friction_amount=friction_amount,
                net_amount=net_amount,
                signal_source=request.signal_source or "MANUAL",
                idempotency_key=request.idempotency_key
            )
            db.add(tx)
            db.commit()
            db.refresh(tx)
            db.refresh(wallet)

            return ExecuteOrderResponse(
                success=True,
                order_id=tx.id,
                ticker=ticker,
                display_name=stock.display_name,
                side=side,
                quantity=request.quantity,
                execution_price=price,
                price_date=price_date,
                gross_amount=gross_amount,
                friction_amount=friction_amount,
                net_amount=net_amount,
                cash_balance_after=round(wallet.cash_balance, 2),
                message=msg
            )

        except StockMarketPredictorException:
            db.rollback()
            raise
        except Exception as exc:
            db.rollback()
            raise StockMarketPredictorException(
                message=f"Trade execution failed: {str(exc)}",
                status_code=500,
                error_code="ORDER_EXECUTION_FAILED"
            )

    @staticmethod
    def reset_portfolio(db: Session) -> Dict[str, str]:
        """
        Resets the default virtual wallet back to ₹1,00,000 cash, clearing positions and trade history.
        """
        wallet = WalletService.get_or_create_default_wallet(db)
        try:
            db.query(Position).filter(Position.wallet_id == wallet.id).delete()
            db.query(Transaction).filter(Transaction.wallet_id == wallet.id).delete()
            wallet.cash_balance = DEFAULT_STARTING_CAPITAL
            wallet.starting_capital = DEFAULT_STARTING_CAPITAL
            db.commit()
            return {"status": "success", "message": "Virtual wallet reset to ₹1,00,000 cash."}
        except Exception as exc:
            db.rollback()
            raise StockMarketPredictorException(
                message=f"Portfolio reset failed: {str(exc)}",
                status_code=500,
                error_code="RESET_FAILED"
            )

    @staticmethod
    def get_analytics(db: Session) -> PortfolioAnalyticsResponse:
        """
        Generates portfolio equity curve, asset allocation, and trade performance analytics.
        """
        summary = WalletService.get_portfolio_summary(db)
        wallet = WalletService.get_or_create_default_wallet(db)

        # 1. Asset allocation
        allocation: List[AssetAllocationItem] = []
        tot_val = summary.portfolio_value if summary.portfolio_value > 0 else 1.0

        # Cash allocation
        cash_pct = round((summary.available_cash / tot_val) * 100.0, 2)
        allocation.append(
            AssetAllocationItem(
                name="Available Cash",
                ticker="CASH",
                value=summary.available_cash,
                percentage=cash_pct,
                color="#0284C7"
            )
        )

        for idx, h in enumerate(summary.holdings):
            color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
            pct = round((h.current_value / tot_val) * 100.0, 2)
            allocation.append(
                AssetAllocationItem(
                    name=h.display_name,
                    ticker=h.ticker,
                    value=h.current_value,
                    percentage=pct,
                    color=color
                )
            )

        # 2. Equity progression reconstruction
        txs = db.query(Transaction).filter(
            Transaction.wallet_id == wallet.id
        ).order_by(Transaction.executed_at.asc()).all()

        equity_curve: List[PortfolioEquityPoint] = []
        equity_curve.append(
            PortfolioEquityPoint(
                timestamp=str(wallet.created_at)[:19].replace("T", " "),
                portfolio_value=DEFAULT_STARTING_CAPITAL,
                cash=DEFAULT_STARTING_CAPITAL,
                invested=0.0
            )
        )

        running_cash = DEFAULT_STARTING_CAPITAL
        running_invested = 0.0

        for tx in txs:
            if tx.side == "BUY":
                running_cash -= tx.net_amount
                running_invested += tx.gross_amount
            else:
                running_cash += tx.net_amount
                running_invested -= tx.gross_amount

            equity_curve.append(
                PortfolioEquityPoint(
                    timestamp=str(tx.executed_at)[:19].replace("T", " "),
                    portfolio_value=round(running_cash + max(0.0, running_invested), 2),
                    cash=round(running_cash, 2),
                    invested=round(max(0.0, running_invested), 2)
                )
            )

        # Trade performance metrics from SELLs
        sell_txs = [t for t in txs if t.side == "SELL"]
        winning_trades = 0
        for st in sell_txs:
            # Estimate if net proceeds exceeded execution
            if st.net_amount > st.gross_amount * 0.99:
                winning_trades += 1
        win_rate = round((winning_trades / len(sell_txs)) * 100.0, 2) if sell_txs else 0.0

        return PortfolioAnalyticsResponse(
            equity_curve=equity_curve,
            asset_allocation=allocation,
            cash=summary.available_cash,
            invested=summary.invested_capital,
            portfolio_value=summary.portfolio_value,
            realised_pnl=summary.realised_pnl,
            unrealised_pnl=summary.unrealised_pnl,
            total_pnl=summary.total_pnl,
            total_trades=len(txs),
            winning_trades=winning_trades,
            win_rate=win_rate
        )

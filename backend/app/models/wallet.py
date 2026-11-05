"""
backend/app/models/wallet.py
SQLAlchemy 2.0 ORM models for virtual wallet, paper trading, and watchlists.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    String,
    Float,
    Integer,
    DateTime,
    ForeignKey,
    CheckConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.base import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    wallets: Mapped[List["Wallet"]] = relationship("Wallet", back_populates="user", cascade="all, delete-orphan")
    watchlists: Mapped[List["Watchlist"]] = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")


class Wallet(Base):
    __tablename__ = "wallets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), default="Primary Paper Trading Wallet")
    cash_balance: Mapped[float] = mapped_column(Float, default=100000.0, nullable=False)  # Starting ₹100,000 (1 Lakh)
    starting_capital: Mapped[float] = mapped_column(Float, default=100000.0, nullable=False)
    currency: Mapped[str] = mapped_column(String(10), default="INR")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="wallets")
    positions: Mapped[List["Position"]] = relationship("Position", back_populates="wallet", cascade="all, delete-orphan")
    transactions: Mapped[List["Transaction"]] = relationship("Transaction", back_populates="wallet", cascade="all, delete-orphan")


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    wallet_id: Mapped[str] = mapped_column(String(36), ForeignKey("wallets.id", ondelete="CASCADE"), index=True, nullable=False)
    ticker: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_buy_price: Mapped[float] = mapped_column(Float, nullable=False)
    total_cost: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    wallet: Mapped["Wallet"] = relationship("Wallet", back_populates="positions")

    __table_args__ = (
        CheckConstraint("quantity >= 0", name="check_positive_quantity"),
        Index("idx_wallet_ticker", "wallet_id", "ticker", unique=True),
    )


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    wallet_id: Mapped[str] = mapped_column(String(36), ForeignKey("wallets.id", ondelete="CASCADE"), index=True, nullable=False)
    ticker: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY or SELL
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    execution_price: Mapped[float] = mapped_column(Float, nullable=False)
    gross_amount: Mapped[float] = mapped_column(Float, nullable=False)
    friction_bps: Mapped[float] = mapped_column(Float, default=7.5)  # 7.5 bps one-way Indian statutory friction
    friction_amount: Mapped[float] = mapped_column(Float, nullable=False)
    net_amount: Mapped[float] = mapped_column(Float, nullable=False)
    signal_source: Mapped[Optional[str]] = mapped_column(String(100), default="MANUAL")
    idempotency_key: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True, nullable=True)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    # Relationships
    wallet: Mapped["Wallet"] = relationship("Wallet", back_populates="transactions")

    __table_args__ = (
        CheckConstraint("side IN ('BUY', 'SELL')", name="check_valid_trade_side"),
    )


class Watchlist(Base):
    __tablename__ = "watchlists"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    ticker: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="watchlists")

    __table_args__ = (
        Index("idx_user_ticker", "user_id", "ticker", unique=True),
    )

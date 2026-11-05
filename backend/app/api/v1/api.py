"""
backend/app/api/v1/api.py
Consolidated API v1 router.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, stocks, portfolio

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(stocks.router)
api_router.include_router(portfolio.router)

"""
backend/app/api/v1/endpoints/health.py
Health check and system diagnostics endpoint.
"""

from datetime import datetime, timezone
import sys
from fastapi import APIRouter
from sqlalchemy import text

from backend.app.core.config import settings
from backend.app.database.session import SessionLocal

router = APIRouter()


@router.get("/health", summary="System Health Check", tags=["Diagnostics"])
def health_check():
    """
    Returns system status, environment details, registered stocks count,
    and database connectivity state.
    """
    # Check DB connectivity
    db_status = "unconnected"
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_status = "connected"
    except Exception as exc:
        db_status = f"error: {str(exc)}"

    import config
    stock_count = len(config.list_stocks())

    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "registered_stocks": stock_count,
        "database": db_status,
        "api_prefix": settings.API_V1_PREFIX,
    }

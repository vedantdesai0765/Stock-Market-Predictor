"""
backend/app/main.py
FastAPI application entrypoint for Stock Market Predictor API.
"""

from contextlib import asynccontextmanager
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse

from backend.app.core.config import settings
from backend.app.core.errors import StockMarketPredictorException, stock_exception_handler
from backend.app.database.session import init_db
from backend.app.api.v1.api import api_router

# Ensure src/ and repo root are in sys.path
if str(settings.SRC_DIR) not in sys.path:
    sys.path.insert(0, str(settings.SRC_DIR))
if str(settings.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(settings.ROOT_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables
    init_db()
    yield
    # Shutdown logic if any


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-style Financial Analytics and Stock Prediction API for Indian Equities. "
        "Exposes directional walk-forward predictions, confidence gating, actionable trading signals, "
        "next-morning forecasting, and out-of-sample backtesting."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
app.add_exception_handler(StockMarketPredictorException, stock_exception_handler)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Sanitize unexpected 500 errors so sensitive stack traces are never leaked."""
    import logging
    logging.getLogger("uvicorn.error").error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred. Request was safely halted.",
            "path": str(request.url.path),
        },
    )

# Mount API routers
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main.py:app", host="127.0.0.1", port=8000, reload=True)

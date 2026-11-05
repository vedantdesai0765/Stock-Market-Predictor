"""
backend/app/core/errors.py
Custom exceptions and consistent error response handlers.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse

class StockMarketPredictorException(Exception):
    """Base exception for the application."""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, error_code: str = "INTERNAL_ERROR"):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)

class StockNotFoundError(StockMarketPredictorException):
    def __init__(self, ticker: str):
        super().__init__(
            message=f"Stock ticker '{ticker}' is not supported or not found in registry.",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="STOCK_NOT_FOUND"
        )
        self.ticker = ticker

class ArtifactNotFoundError(StockMarketPredictorException):
    def __init__(self, ticker: str, artifact_type: str):
        super().__init__(
            message=f"Model artifact '{artifact_type}' for stock '{ticker}' was not found. Models may need to be trained first.",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="ARTIFACT_NOT_FOUND"
        )
        self.ticker = ticker
        self.artifact_type = artifact_type

class DataFileNotFoundError(StockMarketPredictorException):
    def __init__(self, ticker: str, file_type: str):
        super().__init__(
            message=f"Data file '{file_type}' for stock '{ticker}' was not found on disk.",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="DATA_FILE_NOT_FOUND"
        )
        self.ticker = ticker
        self.file_type = file_type

async def stock_exception_handler(request: Request, exc: StockMarketPredictorException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": exc.error_code,
            "message": exc.message,
            "path": str(request.url.path),
        }
    )

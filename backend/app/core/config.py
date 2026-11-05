"""
backend/app/core/config.py
Application configuration and path resolution for Stock Market Predictor API.
"""

import os
from pathlib import Path
from typing import List

class Settings:
    PROJECT_NAME: str = "Stock Market Predictor API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")

    # Paths
    # backend/app/core/config.py -> 3 parents up is repo root
    ROOT_DIR: Path = Path(__file__).resolve().parents[3]
    SRC_DIR: Path = ROOT_DIR / "src"
    DATA_DIR: Path = ROOT_DIR / "data"
    MODELS_DIR: Path = ROOT_DIR / "models"
    REPORTS_DIR: Path = ROOT_DIR / "reports"
    NEWS_DIR: Path = DATA_DIR / "processed" / "news"
    FORECAST_DIR: Path = REPORTS_DIR / "forecast"
    EXPLAIN_DIR: Path = ROOT_DIR / "sentiment_study" / "reports" / "explain"

    # Database
    DATABASE_PATH: Path = DATA_DIR / "app.db"
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH.as_posix()}")

    # CORS Origins for Next.js frontend and Streamlit terminal
    @property
    def CORS_ORIGINS(self) -> List[str]:
        env_origins = os.getenv("CORS_ORIGINS")
        if env_origins:
            return [o.strip() for o in env_origins.split(",") if o.strip()]
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ]

settings = Settings()

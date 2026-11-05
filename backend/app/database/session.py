"""
backend/app/database/session.py
Database session management and table initialization.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from backend.app.core.config import settings
from backend.app.database.base import Base

# Ensure parent directory for database exists
settings.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db() -> None:
    """Create all tables in the database if they do not exist."""
    # Import all models so that Base knows about them before create_all
    import backend.app.models.wallet  # noqa: F401
    Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    """Dependency that yields a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

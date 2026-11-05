"""
backend/tests/conftest.py
Pytest fixtures for FastAPI TestClient.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.main import app
from backend.app.database.session import init_db


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Ensure database tables are created once for test session."""
    init_db()


@pytest.fixture(scope="module")
def client():
    """Module-scoped FastAPI TestClient."""
    with TestClient(app) as test_client:
        yield test_client

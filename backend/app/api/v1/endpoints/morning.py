"""
backend/app/api/v1/endpoints/morning.py
REST endpoints for the Morning Call: the live next-morning opening forecast and its scorecard.
"""

from fastapi import APIRouter, Query

from backend.app.schemas.morning import MorningCallResponse, ScorecardResponse
from backend.app.services.morning_service import MorningService

router = APIRouter(prefix="/morning-call", tags=["Morning Call"])


@router.get("", response_model=MorningCallResponse, response_model_by_alias=True,
            summary="Get the Latest Morning Call")
def get_morning_call():
    """
    Returns the most recent forecast of whether each stock opens above or below its previous close,
    made before the NSE opens from last night's US market, the VIX and this morning's Asian opens.
    Once the market has opened, each call also carries what really happened.
    """
    return MorningService.get_morning_call()


@router.get("/scorecard", response_model=ScorecardResponse, response_model_by_alias=True,
            summary="Get the Live Scorecard")
def get_scorecard(recent_days: int = Query(5, ge=0, le=30, description="Number of recent mornings to list call by call")):
    """
    Returns the live record of the Morning Call (only forecasts made before 09:15 IST count),
    the same days scored for the simple guess, and the walk-forward test results for comparison.
    """
    return MorningService.get_scorecard(recent_days=recent_days)

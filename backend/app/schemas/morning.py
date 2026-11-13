"""
backend/app/schemas/morning.py
Pydantic schemas for the Morning Call (live next-morning opening forecast) and its scorecard.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class MarketInput(BaseModel):
    key: str
    label: str
    region: str
    change_pct: Optional[float] = Field(None, description="Overnight move or opening gap, in percent")
    level: Optional[float] = None
    session: Optional[str] = Field(None, description="Session date the number comes from")
    available: bool = True
    note: Optional[str] = None


class Driver(BaseModel):
    key: str
    label: str
    detail: str
    direction: str = Field(..., description="'up' pushes the forecast higher than on an average morning, 'down' lower")
    weight: float = Field(..., description="Share of the total push that comes from this group (0-1)")


class Confidence(BaseModel):
    level: str = ""
    tested_accuracy: Optional[float] = Field(None, description="How often calls this confident were right in walk-forward testing")
    tested_n: int = 0


class CloseCall(BaseModel):
    prob_up: float
    call: str
    baseline_call: str
    confidence: Confidence


class CallResult(BaseModel):
    open_price: Optional[float] = None
    actual: Optional[str] = Field(None, description="HIGHER, LOWER, FLAT (not scored) or VOID (market did not open)")
    correct: Optional[bool] = None
    close_price: Optional[float] = None
    close_actual: Optional[str] = None
    close_correct: Optional[bool] = None


class StockCall(BaseModel):
    ticker: str
    name: str
    available: bool
    reason: Optional[str] = None
    prev_close: Optional[float] = None
    prob_up: Optional[float] = Field(None, description="Model probability that the stock opens above yesterday's close")
    call: Optional[str] = Field(None, description="HIGHER or LOWER")
    baseline_call: Optional[str] = None
    confidence: Optional[Confidence] = None
    drivers: List[Driver] = []
    close: Optional[CloseCall] = None
    result: Optional[CallResult] = None


class TestedAccuracy(BaseModel):
    n: int
    accuracy: float
    baseline: float
    edge: float
    years: Optional[str] = None
    years_beating_guess: Optional[int] = None
    years_tested: Optional[int] = None
    from_: Optional[str] = Field(None, alias="from")
    to: Optional[str] = None

    model_config = {"populate_by_name": True}


class ModelInfo(BaseModel):
    name: str
    trained_through: Optional[str] = None
    tested: Optional[TestedAccuracy] = None
    after_study: Optional[TestedAccuracy] = None
    simple_guess: Optional[str] = None
    base_rate_up: Optional[float] = Field(None, description="Share of training mornings on which a stock opened higher")


class CallSummary(BaseModel):
    higher: int = 0
    lower: int = 0
    unavailable: int = 0


class MorningCallResponse(BaseModel):
    status: str = Field(..., description="ok, market_closed, unavailable or not_started")
    today: str = Field(..., description="Today's date in India")
    server_time: str
    is_today: bool = Field(..., description="True when the forecast below is for today's session")
    market_phase: str = Field(..., description="pre_open, open, closed or holiday (for today, in India)")
    next_call_date: str = Field(..., description="Date of the next scheduled forecast")
    target_date: Optional[str] = None
    based_on: Optional[str] = None
    generated_at: Optional[str] = None
    on_time: Optional[bool] = None
    reason: Optional[str] = None
    next_session: Optional[str] = None
    summary: Optional[CallSummary] = None
    market: List[MarketInput] = []
    calls: List[StockCall] = []
    model: Optional[ModelInfo] = None
    notes: List[str] = []
    source: str = Field("local", description="'github' when read from the public live folder, else 'local'")


class Score(BaseModel):
    n: int = 0
    correct: int = 0
    accuracy: Optional[float] = None
    baseline_correct: int = 0
    baseline_accuracy: Optional[float] = None


class DayScore(BaseModel):
    date: str
    n: int
    correct: int


class StockScore(BaseModel):
    ticker: str
    name: str
    n: int
    correct: int
    accuracy: float


class RecentResult(BaseModel):
    ticker: str
    call: str
    prob_up: float
    actual: Optional[str] = None
    correct: Optional[bool] = None


class RecentDay(BaseModel):
    date: str
    results: List[RecentResult]


class ConfidenceBand(BaseModel):
    from_: float = Field(..., alias="from")
    to: float
    n: int
    share: float
    accuracy: Optional[float] = None

    model_config = {"populate_by_name": True}


class ScorecardResponse(BaseModel):
    updated_at: Optional[str] = None
    since: Optional[str] = None
    mornings: int = 0
    open: Score = Score()
    close: Score = Score()
    days: List[DayScore] = []
    per_stock: List[StockScore] = []
    excluded_late: int = 0
    recent: List[RecentDay] = []
    tested: Optional[TestedAccuracy] = None
    after_study: Optional[TestedAccuracy] = None
    confidence_bands: List[ConfidenceBand] = []
    source: str = "local"

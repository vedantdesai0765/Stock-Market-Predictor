"""
backend/app/schemas/prediction.py
Pydantic schemas for directional predictions, confidence gating, and model metrics.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class CoverageCurvePoint(BaseModel):
    threshold: float
    coverage: float
    accuracy: float
    edge: float
    n_committed: int


class PredictionResponse(BaseModel):
    ticker: str = Field(..., description="Stock ticker key, e.g. TCS")
    display_name: str = Field(..., description="Company display name")
    date: str = Field(..., description="As-of market date for this prediction (YYYY-MM-DD)")
    close: float = Field(..., description="Latest closing price")
    target_horizon_days: int = Field(default=3, description="Directional prediction horizon in trading days")
    probability_up: float = Field(..., description="Calibrated/raw probability of upward price movement")
    probability_down: float = Field(..., description="Probability of downward price movement (1 - p_up)")
    confidence: float = Field(..., description="Conviction distance from neutral: abs(p_up - 0.5)")
    confidence_threshold: float = Field(..., description="Operating threshold required to emit a signal")
    signal: str = Field(..., description="Directional signal: 'UP', 'DOWN', or 'NO SIGNAL'")
    abstention_active: bool = Field(..., description="True if confidence is below operating threshold")
    expected_accuracy: Optional[float] = Field(None, description="Expected out-of-sample accuracy at operating point")
    expected_coverage: Optional[float] = Field(None, description="Expected fraction of days receiving an active signal")
    walk_forward_accuracy: Optional[float] = Field(None, description="Overall walk-forward accuracy across folds")
    walk_forward_baseline: Optional[float] = Field(None, description="Majority class baseline accuracy")
    walk_forward_edge: Optional[float] = Field(None, description="Edge over majority baseline (accuracy - baseline)")
    model_name: Optional[str] = Field(None, description="Model architecture (e.g. logistic, gradient_boosting)")
    trained_at: Optional[str] = Field(None, description="Timestamp when model was trained")
    features_count: Optional[int] = Field(None, description="Total number of stationary features used")
    top_features: Optional[List[FeatureImportanceItem]] = Field(None, description="Top influential features")
    coverage_curve: Optional[List[CoverageCurvePoint]] = Field(None, description="Operating curve points")

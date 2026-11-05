"""
backend/app/services/prediction_service.py
Service adapter for directional prediction, confidence gating, and model metrics.
"""

import sys
from typing import List, Optional
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.core.errors import ArtifactNotFoundError, StockNotFoundError
from backend.app.schemas.prediction import (
    CoverageCurvePoint,
    FeatureImportanceItem,
    PredictionResponse,
)

# Ensure src/ is importable
if str(settings.SRC_DIR) not in sys.path:
    sys.path.insert(0, str(settings.SRC_DIR))
if str(settings.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(settings.ROOT_DIR))

import config
import pipeline


class PredictionService:
    @staticmethod
    def get_prediction(ticker: str) -> PredictionResponse:
        key = ticker.strip().upper()
        if key not in config.STOCKS:
            raise StockNotFoundError(ticker=ticker)

        if not pipeline.artifacts_exist(key):
            raise ArtifactNotFoundError(ticker=ticker, artifact_type="model.joblib / metadata.json")

        # 1. Load artifacts
        try:
            art = pipeline.load_artifacts(key)
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError(ticker=ticker, artifact_type=str(exc))

        model = art["model"]
        metadata = art["metadata"]
        curve_df = art.get("curve")

        # 2. Get latest prediction
        latest_pred = pipeline.predict_latest(key)

        p_up = latest_pred["probability_up"]
        p_down = round(1.0 - p_up, 4)
        conf = latest_pred["confidence"]
        threshold = latest_pred["confidence_threshold"]
        signal = latest_pred["signal"]
        abstention = conf < threshold

        # 3. Extract top features
        features = metadata.get("features", [])
        top_features: List[FeatureImportanceItem] = []
        try:
            estimator = model
            if hasattr(model, "named_steps") and "clf" in model.named_steps:
                estimator = model.named_steps["clf"]
            elif hasattr(model, "steps") and len(model.steps) > 0:
                estimator = model.steps[-1][1]

            if hasattr(estimator, "coef_"):
                coefs = np.asarray(estimator.coef_).ravel()
                feature_pairs = sorted(
                    zip(features, coefs),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                top_features = [
                    FeatureImportanceItem(feature=f, importance=round(float(w), 4))
                    for f, w in feature_pairs[:10]
                ]
            elif hasattr(estimator, "feature_importances_"):
                imps = np.asarray(estimator.feature_importances_).ravel()
                feature_pairs = sorted(
                    zip(features, imps),
                    key=lambda x: x[1],
                    reverse=True
                )
                top_features = [
                    FeatureImportanceItem(feature=f, importance=round(float(w), 4))
                    for f, w in feature_pairs[:10]
                ]
        except Exception:
            top_features = []

        # 4. Extract coverage curve points
        curve_points: List[CoverageCurvePoint] = []
        if curve_df is not None and not curve_df.empty:
            for _, row in curve_df.head(20).iterrows():
                curve_points.append(
                    CoverageCurvePoint(
                        threshold=round(float(row.get("threshold", 0.0)), 4),
                        coverage=round(float(row.get("coverage", 0.0)), 4),
                        accuracy=round(float(row.get("accuracy", 0.0)), 4),
                        edge=round(float(row.get("edge", 0.0)), 4),
                        n_committed=int(row.get("n_committed", 0)),
                    )
                )

        stock_cfg = config.get_stock(key)
        op_point = metadata.get("operating_point", {})

        return PredictionResponse(
            ticker=key,
            display_name=stock_cfg.display_name,
            date=latest_pred["date"],
            close=latest_pred["close"],
            target_horizon_days=latest_pred["horizon_days"],
            probability_up=p_up,
            probability_down=p_down,
            confidence=conf,
            confidence_threshold=threshold,
            signal=signal,
            abstention_active=abstention,
            expected_accuracy=op_point.get("accuracy"),
            expected_coverage=op_point.get("coverage"),
            walk_forward_accuracy=metadata.get("walk_forward_accuracy"),
            walk_forward_baseline=metadata.get("baseline"),
            walk_forward_edge=metadata.get("edge"),
            model_name=metadata.get("config", {}).get("model_name"),
            trained_at=metadata.get("trained_at"),
            features_count=len(features),
            top_features=top_features,
            coverage_curve=curve_points,
        )

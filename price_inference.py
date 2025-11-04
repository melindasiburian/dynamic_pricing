"""Realtime inference helpers for the dynamic pricing pipeline."""
from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Dict
import sys

import joblib
import pandas as pd

if __package__ in (None, ""):
    CURRENT_DIR = Path(__file__).resolve().parent
    if str(CURRENT_DIR) not in sys.path:
        sys.path.append(str(CURRENT_DIR))
    live_feature_builder = import_module("live_feature_builder")
else:
    live_feature_builder = import_module(f"{__package__}.live_feature_builder")

build_realtime_features_for_service = live_feature_builder.build_realtime_features_for_service

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "dynamic_pricing_model.joblib"

FEATURE_COLUMNS = [
    "category",
    "competitive_price",
    "competitor_count",
    "category_quantity",
    "total_visitors",
    "monthly_event_days",
    "temperature_celsius",
    "prcp_mm",
]


@lru_cache(maxsize=1)
def _load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Could not locate dynamic pricing model at " f"{MODEL_PATH}. Run train_and_save_model first."
        )
    return joblib.load(MODEL_PATH)


def get_live_pricing_recommendation(service_id: str, lat: float, lon: float) -> Dict[str, float]:
    """Generate a pricing recommendation for the given service."""
    pipeline = _load_pipeline()
    feature_snapshot = build_realtime_features_for_service(service_id, lat, lon)
    feature_frame = pd.DataFrame([feature_snapshot])
    prediction_features = feature_frame[FEATURE_COLUMNS]
    recommended_price = float(pipeline.predict(prediction_features)[0])

    return {
        "service_id": feature_snapshot["category"],
        "current_base_price": float(feature_snapshot["base_price_idr"]),
        "recommended_price": recommended_price,
        "delta_pct": (
            (recommended_price - feature_snapshot["base_price_idr"]) / feature_snapshot["base_price_idr"]
        )
        * 100.0,
        "context": {
            "competitor_count": int(feature_snapshot["competitor_count"]),
            "total_visitors": int(feature_snapshot["total_visitors"]),
            "monthly_event_days": int(feature_snapshot["monthly_event_days"]),
            "temperature_celsius": float(feature_snapshot["temperature_celsius"]),
            "prcp_mm": float(feature_snapshot["prcp_mm"]),
        },
        "raw_features": feature_snapshot,
    }


__all__ = ["get_live_pricing_recommendation", "FEATURE_COLUMNS"]

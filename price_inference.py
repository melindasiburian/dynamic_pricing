"""Inference utilities for the dynamic pricing model."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from .live_feature_builder import build_realtime_features_for_service

_MODULE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = [
    Path("artifacts/dynamic_pricing_model.joblib"),
    Path("dynamic_pricing_model.joblib"),
    _MODULE_DIR / "dynamic_pricing_model.joblib",
]

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
    for model_path in MODEL_PATHS:
        if model_path.exists():
            return joblib.load(model_path)
    raise FileNotFoundError(
        "Could not locate dynamic pricing model. Expected one of: "
        + ", ".join(str(path) for path in MODEL_PATHS)
    )


def get_live_pricing_recommendation(service_id: str, lat: float, lon: float) -> Dict[str, float]:
    """Return a pricing recommendation for the specified service."""
    pipeline = _load_pipeline()
    feature_snapshot = build_realtime_features_for_service(service_id, lat, lon)
    feature_frame = pd.DataFrame([feature_snapshot])
    prediction_features = feature_frame[FEATURE_COLUMNS]
    recommended_price = float(pipeline.predict(prediction_features)[0])

    result = {
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
    return result


__all__ = [
    "get_live_pricing_recommendation",
    "FEATURE_COLUMNS",
]

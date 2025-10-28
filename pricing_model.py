"""Utilities for loading and using the trained dynamic pricing model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = (
    PROJECT_DIR / "artifacts" / "dynamic_pricing_model.joblib",
    PROJECT_DIR / "dynamic_pricing_model.joblib",
)

REQUIRED_FEATURES = (
    "category",
    "competitive_price",
    "competitor_count",
    "category_quantity",
    "total_visitors",
    "monthly_event_days",
    "temperature_celsius",
    "prcp_mm",
)


class PricingModelNotFoundError(FileNotFoundError):
    """Raised when the trained pricing model artifact cannot be located."""


def find_model_path() -> Optional[Path]:
    """Return the first existing model path if available."""
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def get_model_path() -> Path:
    """Return the resolved path to the trained model or raise an error."""
    model_path = find_model_path()
    if model_path is None:
        raise PricingModelNotFoundError(
            "Trained pricing model not found. Run model_training.py to generate the artifact."
        )
    return model_path


@lru_cache(maxsize=1)
def load_model():
    """Load the trained pricing pipeline with simple memoisation."""
    model_path = get_model_path()
    return joblib.load(model_path)


def predict_price(features: Mapping[str, object]) -> float:
    """Predict the optimal price using the trained model.

    Args:
        features: Mapping containing the required feature values defined in
            :data:`REQUIRED_FEATURES`.

    Returns:
        Predicted price as a float.

    Raises:
        PricingModelNotFoundError: If the model artifact does not exist.
        ValueError: If the provided feature mapping is missing any required keys.
    """

    missing = [key for key in REQUIRED_FEATURES if key not in features]
    if missing:
        raise ValueError(f"Missing required features for prediction: {', '.join(missing)}")

    model = load_model()
    row = {name: features[name] for name in REQUIRED_FEATURES}
    frame = pd.DataFrame([row])
    prediction = model.predict(frame)
    return float(prediction[0])

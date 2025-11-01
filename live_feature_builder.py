"""Utilities for building realtime feature snapshots for dynamic pricing."""
from __future__ import annotations

import datetime as _dt
import random
from typing import Dict


def get_weather_snapshot(lat: float, lon: float) -> Dict[str, float]:
    """Return a dummy weather snapshot for the provided coordinates.

    In production this should be replaced with a call to a weather API such as
    OpenWeather or Tomorrow.io.
    """
    random.seed(f"{lat:.4f}:{lon:.4f}:{_dt.datetime.utcnow():%Y%m%d%H}")
    temperature_celsius = round(random.uniform(24.0, 34.0), 2)
    prcp_mm = round(random.uniform(0.0, 8.0), 2)
    return {
        "temperature_celsius": float(temperature_celsius),
        "prcp_mm": float(prcp_mm),
    }


def get_monthly_event_days(now_datetime: _dt.datetime) -> int:
    """Return a dummy count of event days for the month of ``now_datetime``."""
    # The heuristic below simulates more events during peak seasons.
    month = now_datetime.month
    base_events = {1: 2, 2: 3, 3: 4, 4: 3, 5: 5, 6: 4, 7: 6, 8: 6, 9: 3, 10: 4, 11: 5, 12: 7}
    variance = (now_datetime.day % 3) - 1  # -1, 0, or +1
    return max(0, base_events.get(month, 3) + variance)


def get_internal_service_snapshot(service_id: str) -> Dict[str, float]:
    """Return a dummy snapshot of internal metrics for ``service_id``."""
    service_catalog = {
        "VIP_TABLE": {
            "base_price_idr": 2_500_000.0,
            "competitive_price": 2_750_000.0,
            "competitor_count": 5,
            "category_quantity": 3,
            "total_visitors": 120,
        },
        "JETSKI_30MIN": {
            "base_price_idr": 850_000.0,
            "competitive_price": 900_000.0,
            "competitor_count": 8,
            "category_quantity": 12,
            "total_visitors": 65,
        },
    }

    snapshot = service_catalog.get(service_id.upper())
    if snapshot is None:
        random.seed(service_id)
        snapshot = {
            "base_price_idr": float(round(random.uniform(500_000, 3_000_000), 2)),
            "competitive_price": float(round(random.uniform(450_000, 3_200_000), 2)),
            "competitor_count": int(random.randint(2, 12)),
            "category_quantity": int(random.randint(1, 25)),
            "total_visitors": int(random.randint(25, 200)),
        }

    snapshot["category"] = service_id.upper()
    return snapshot


def build_realtime_features_for_service(service_id: str, lat: float, lon: float) -> Dict[str, object]:
    """Build the realtime feature dictionary for ``service_id`` at ``lat``/``lon``."""
    now = _dt.datetime.utcnow()
    weather_snapshot = get_weather_snapshot(lat, lon)
    internal_snapshot = get_internal_service_snapshot(service_id)

    features = {
        "date": now.strftime("%Y-%m-%d"),
        "base_price_idr": internal_snapshot["base_price_idr"],
        "category": internal_snapshot["category"],
        "competitive_price": internal_snapshot["competitive_price"],
        "competitor_count": internal_snapshot["competitor_count"],
        "category_quantity": internal_snapshot["category_quantity"],
        "total_visitors": internal_snapshot["total_visitors"],
        "monthly_event_days": get_monthly_event_days(now),
        "temperature_celsius": weather_snapshot["temperature_celsius"],
        "prcp_mm": weather_snapshot["prcp_mm"],
    }
    return features

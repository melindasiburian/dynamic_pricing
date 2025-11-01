"""Utilities to log pricing recommendations and simulated outcomes."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict

PROJECT_DIR = Path(__file__).resolve().parent
LOG_PATH = PROJECT_DIR / "artifacts" / "pricing_log.csv"


def log_pricing_event(state: Dict[str, object], recommendation: Dict[str, object], outcome: Dict[str, object]) -> None:
    """Append a pricing event to the log file."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()

    with LOG_PATH.open("a", newline="") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "timestamp",
                "service_id",
                "recommended_price",
                "base_price_idr",
                "units_sold",
                "revenue_idr",
                "profit_idr",
                "competitor_count",
                "total_visitors",
                "monthly_event_days",
                "temperature_celsius",
                "prcp_mm",
            ],
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "service_id": state["category"],
                "recommended_price": recommendation["recommended_price"],
                "base_price_idr": state["base_price_idr"],
                "units_sold": outcome["units_sold"],
                "revenue_idr": outcome["revenue_idr"],
                "profit_idr": outcome["profit_idr"],
                "competitor_count": state["competitor_count"],
                "total_visitors": state["total_visitors"],
                "monthly_event_days": state["monthly_event_days"],
                "temperature_celsius": state["temperature_celsius"],
                "prcp_mm": state["prcp_mm"],
            }
        )


__all__ = ["log_pricing_event", "LOG_PATH"]

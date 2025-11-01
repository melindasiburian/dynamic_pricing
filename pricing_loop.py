"""Simple live pricing loop that simulates recommendations and outcomes."""
from __future__ import annotations

import time
from typing import Dict

from .price_inference import get_live_pricing_recommendation
from .pricing_logger import log_pricing_event


def simulate_outcome(price_idr: float, visitors: int) -> Dict[str, int | float]:
    """Simulate a simple demand curve to produce outcome metrics."""
    base_demand = visitors * 0.2
    price_pressure = max(0.1, 5_000_000 / max(price_idr, 1))
    units_sold = max(0, int(base_demand * price_pressure))

    revenue_idr = units_sold * price_idr
    profit_idr = revenue_idr * 0.35

    return {
        "units_sold": units_sold,
        "revenue_idr": revenue_idr,
        "profit_idr": profit_idr,
    }


def run_pricing_loop(iterations: int = 5, sleep_seconds: float = 0.0) -> None:
    """Execute a toy pricing loop for demonstration purposes."""
    for _ in range(iterations):
        recommendation = get_live_pricing_recommendation("VIP_TABLE", -8.65, 115.13)
        state = recommendation["raw_features"]
        outcome = simulate_outcome(recommendation["recommended_price"], state["total_visitors"])
        log_pricing_event(state, recommendation, outcome)
        if sleep_seconds:
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    run_pricing_loop()

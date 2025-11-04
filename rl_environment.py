"""Gym-compatible environment for reinforcement learning on pricing logs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import gym
import numpy as np
import pandas as pd


class PricingEnv(gym.Env):
    """Environment that replays logged pricing experiences."""

    STATE_COLUMNS = [
        "competitor_count",
        "total_visitors",
        "monthly_event_days",
        "temperature_celsius",
        "prcp_mm",
    ]

    def __init__(self, log_path: Path):
        super().__init__()
        if not log_path.exists():
            raise FileNotFoundError(f"Pricing log not found at {log_path}")

        self.log_df = pd.read_csv(log_path)
        if self.log_df.empty:
            raise ValueError("Pricing log is empty. Run the pricing loop to generate data.")

        self.state_cols = self.STATE_COLUMNS
        lows = np.array([self.log_df[c].min() for c in self.state_cols], dtype=np.float32)
        highs = np.array([self.log_df[c].max() for c in self.state_cols], dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=np.array([-0.1], dtype=np.float32),
            high=np.array([0.1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low=lows, high=highs, dtype=np.float32)
        self.idx = 0

    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.idx = 0
        return self._get_state(), {}

    def _get_state(self) -> np.ndarray:
        row = self.log_df.iloc[self.idx]
        return np.array([row[c] for c in self.state_cols], dtype=np.float32)

    def step(self, action: np.ndarray):
        row = self.log_df.iloc[self.idx]

        # Clip the action to the defined bounds and interpret it as a percentage
        # adjustment to the logged recommended price.
        clipped_action = float(
            np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        )

        base_price = float(row["recommended_price"])
        adjusted_price = max(1.0, base_price * (1.0 + clipped_action))

        visitors = max(0, int(row["total_visitors"]))
        base_demand = visitors * 0.2
        price_pressure = max(0.1, 5_000_000 / max(adjusted_price, 1.0))
        units_sold = max(0, int(base_demand * price_pressure))

        revenue_idr = units_sold * adjusted_price
        reward = revenue_idr * 0.35

        self.idx += 1
        done = self.idx >= len(self.log_df)

        if done:
            next_state = np.zeros(len(self.state_cols), dtype=np.float32)
        else:
            next_state = self._get_state()

        info: Dict = {
            "logged_profit": float(row["profit_idr"]),
            "adjusted_price": adjusted_price,
            "units_sold": units_sold,
        }
        truncated = False
        return next_state, reward, done, truncated, info


__all__ = ["PricingEnv"]

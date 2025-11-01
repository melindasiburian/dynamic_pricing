"""Training helpers for the reinforcement learning pricing agent."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np
from stable_baselines3 import PPO

from .rl_environment import PricingEnv

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
LOG_PATH = ARTIFACTS_DIR / "pricing_log.csv"
AGENT_PATH = ARTIFACTS_DIR / "rl_agent.zip"


def train_rl_agent(total_timesteps: int = 10_000) -> PPO:
    """Train a PPO agent on the logged pricing data and persist it."""
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"Pricing log not found at {LOG_PATH}. Run the pricing loop to generate training data."
        )

    env = PricingEnv(LOG_PATH)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(AGENT_PATH))
    return model


@lru_cache(maxsize=1)
def _load_agent() -> PPO:
    if not AGENT_PATH.exists():
        raise FileNotFoundError(f"RL agent not found at {AGENT_PATH}. Train the agent first.")
    return PPO.load(str(AGENT_PATH))


def get_rl_price_suggestion(state: Dict[str, float]) -> float:
    """Return the percentage adjustment recommended by the RL agent."""
    agent = _load_agent()
    observation = np.array([float(state[col]) for col in PricingEnv.STATE_COLUMNS], dtype=np.float32)
    action, _ = agent.predict(observation, deterministic=True)
    adjustment = float(np.clip(action[0], -0.1, 0.1))
    return adjustment


__all__ = ["train_rl_agent", "get_rl_price_suggestion"]

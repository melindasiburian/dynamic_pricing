"""
Comprehensive Reinforcement Learning Environment for Dynamic Pricing
===================================================================

This module implements a sophisticated RL environment that simulates
real-world pricing and inventory management scenarios.
"""

import gym
from gym import spaces
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any


class DynamicPricingEnvironment(gym.Env):
    """
    Advanced RL Environment for Dynamic Pricing & Inventory Optimization

    Observation Space:
    - current_inventory: [0, 1000]
    - current_price: [10.0, 500.0]
    - competitor_avg_price: [5.0, 600.0]
    - demand_trend_7d: [-100, 100]
    - market_sentiment: [-1.0, 1.0]
    - day_of_week: [0, 6]
    - inventory_holding_cost: [0.0, 50.0]
    - stockout_penalty_rate: [0.0, 100.0]
    - profit_margin: [0.0, 0.8]
    - seasonality_factor: [0.5, 2.0]

    Action Space:
    - price_adjustment: Discrete(5) -> [-20%, -10%, 0%, +10%, +20%]
    - inventory_reorder: Discrete(5) -> [0, 50, 100, 200, 500] units
    """

    def __init__(self,
                 product_config: Dict[str, Any] = None,
                 market_config: Dict[str, Any] = None,
                 simulation_days: int = 365):

        super(DynamicPricingEnvironment, self).__init__()

        # Environment configuration
        self.simulation_days = simulation_days
        self.current_day = 0

        # Product configuration
        self.product_config = product_config or {
            'base_cost': 60.0,
            'base_price': 100.0,
            'price_elasticity': -1.2,  # % demand change per % price change
            'max_inventory': 1000,
            'holding_cost_per_unit_per_day': 0.5,
            'stockout_penalty_per_unit': 25.0,
            'lead_time_days': 3,
            'category': 'Electronics'
        }

        # Market configuration
        self.market_config = market_config or {
            'competitor_volatility': 0.05,  # Daily price change volatility
            'sentiment_volatility': 0.1,  # Daily sentiment change
            'base_demand': 50,  # Daily base demand
            'seasonality_amplitude': 0.3  # Seasonal variation amplitude
        }

        # Define observation space (10 dimensions)
        self.observation_space = spaces.Box(
            low=np.array([0, 10.0, 5.0, -100, -1.0, 0, 0.0, 0.0, 0.0, 0.5]),
            high=np.array([1000, 500.0, 600.0, 100, 1.0, 6, 50.0, 100.0, 0.8, 2.0]),
            dtype=np.float32
        )

        # Define action space (2 discrete actions)
        self.action_space = spaces.MultiDiscrete([5, 5])  # [price_adj, inventory_order]

        # Initialize state variables
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""

        self.current_day = 0

        # Initialize core state
        self.current_inventory = 200.0  # Starting inventory
        self.current_price = self.product_config['base_price']
        self.competitor_avg_price = self.current_price * (1 + random.uniform(-0.1, 0.1))

        # Initialize market conditions
        self.market_sentiment = random.uniform(-0.2, 0.2)  # Slightly random start
        self.demand_history = [self.market_config['base_demand']] * 7  # Last 7 days

        # Performance tracking
        self.total_profit = 0.0
        self.total_revenue = 0.0
        self.total_sales = 0.0
        self.stockout_days = 0
        self.excess_inventory_days = 0

        # Market simulation state
        self.pending_orders = []  # [(delivery_day, quantity), ...]

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step in the environment"""

        # Decode actions
        price_action = action[0]  # 0-4: [-20%, -10%, 0%, +10%, +20%]
        inventory_action = action[1]  # 0-4: [0, 50, 100, 200, 500] units

        # Apply price adjustment
        price_adjustments = [-0.20, -0.10, 0.0, 0.10, 0.20]
        new_price = self.current_price * (1 + price_adjustments[price_action])
        new_price = max(10.0, min(500.0, new_price))  # Price bounds

        # Apply inventory reorder
        reorder_quantities = [0, 50, 100, 200, 500]
        reorder_quantity = reorder_quantities[inventory_action]

        if reorder_quantity > 0:
            delivery_day = self.current_day + self.product_config['lead_time_days']
            self.pending_orders.append((delivery_day, reorder_quantity))

        # Simulate market evolution
        self._simulate_market_dynamics()

        # Process pending inventory deliveries
        self._process_inventory_deliveries()

        # Calculate demand based on current conditions
        demand = self._calculate_demand(new_price)

        # Process sales (limited by inventory)
        actual_sales = min(demand, self.current_inventory)
        unmet_demand = max(0, demand - self.current_inventory)

        # Update inventory
        self.current_inventory -= actual_sales

        # Calculate rewards (daily profit)
        reward = self._calculate_reward(new_price, actual_sales, unmet_demand)

        # Update state
        self.current_price = new_price
        self.demand_history.append(actual_sales)
        self.demand_history = self.demand_history[-7:]  # Keep last 7 days

        # Update performance metrics
        self._update_metrics(new_price, actual_sales, unmet_demand)

        # Check if episode is done
        self.current_day += 1
        done = self.current_day >= self.simulation_days

        # Prepare info dictionary
        info = {
            'day': self.current_day,
            'price': new_price,
            'inventory': self.current_inventory,
            'demand': demand,
            'sales': actual_sales,
            'unmet_demand': unmet_demand,
            'profit': reward,
            'total_profit': self.total_profit,
            'competitor_price': self.competitor_avg_price,
            'sentiment': self.market_sentiment
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation state"""

        demand_trend = np.mean(self.demand_history[-7:]) - np.mean(
            self.demand_history[-14:-7] if len(self.demand_history) >= 14 else self.demand_history[:7])

        return np.array([
            self.current_inventory,  # Current inventory level
            self.current_price,  # Current product price
            self.competitor_avg_price,  # Average competitor price
            demand_trend,  # 7-day demand trend
            self.market_sentiment,  # Market sentiment score
            self.current_day % 7,  # Day of week (0-6)
            self.product_config['holding_cost_per_unit_per_day'],  # Holding cost
            self.product_config['stockout_penalty_per_unit'],  # Stockout penalty
            self._calculate_current_margin(),  # Current profit margin
            self._get_seasonality_factor()  # Seasonal factor
        ], dtype=np.float32)

    def _simulate_market_dynamics(self):
        """Simulate market evolution (competitors, sentiment, etc.)"""

        # Competitor price evolution
        competitor_change = random.gauss(0, self.market_config['competitor_volatility'])
        self.competitor_avg_price *= (1 + competitor_change)
        self.competitor_avg_price = max(5.0, min(600.0, self.competitor_avg_price))

        # Market sentiment evolution
        sentiment_change = random.gauss(0, self.market_config['sentiment_volatility'])
        self.market_sentiment += sentiment_change
        self.market_sentiment = max(-1.0, min(1.0, self.market_sentiment))

        # Add external market shocks (rare events)
        if random.random() < 0.01:  # 1% chance per day
            shock_magnitude = random.choice([-0.3, -0.2, 0.2, 0.3])
            self.market_sentiment += shock_magnitude
            self.market_sentiment = max(-1.0, min(1.0, self.market_sentiment))

    def _calculate_demand(self, price: float) -> float:
        """Calculate demand based on price and market conditions"""

        # Base demand with seasonality
        base_demand = self.market_config['base_demand'] * self._get_seasonality_factor()

        # Price elasticity effect
        price_ratio = price / self.product_config['base_price']
        price_effect = (price_ratio - 1) * self.product_config['price_elasticity']

        # Competitive effect
        if self.competitor_avg_price > 0:
            competitive_advantage = (self.competitor_avg_price - price) / price
            competitive_effect = competitive_advantage * 0.5  # 50% of advantage translates to demand
        else:
            competitive_effect = 0

        # Sentiment effect
        sentiment_effect = self.market_sentiment * 0.2  # 20% demand change at extreme sentiment

        # Combine all effects
        total_demand_multiplier = 1 + price_effect + competitive_effect + sentiment_effect
        demand = base_demand * max(0.1, total_demand_multiplier)  # Minimum 10% of base demand

        # Add noise
        demand *= random.uniform(0.8, 1.2)

        return max(0, demand)

    def _calculate_reward(self, price: float, sales: float, unmet_demand: float) -> float:
        """Calculate reward (daily profit)"""

        # Revenue
        revenue = price * sales

        # Costs
        cost_of_goods_sold = self.product_config['base_cost'] * sales
        holding_cost = self.current_inventory * self.product_config['holding_cost_per_unit_per_day']
        stockout_penalty = unmet_demand * self.product_config['stockout_penalty_per_unit']

        # Daily profit
        profit = revenue - cost_of_goods_sold - holding_cost - stockout_penalty

        return profit

    def _process_inventory_deliveries(self):
        """Process pending inventory orders that arrive today"""

        arrived_orders = [(day, qty) for day, qty in self.pending_orders if day <= self.current_day]

        for day, quantity in arrived_orders:
            self.current_inventory += quantity
            self.current_inventory = min(self.current_inventory, self.product_config['max_inventory'])

        # Remove processed orders
        self.pending_orders = [(day, qty) for day, qty in self.pending_orders if day > self.current_day]

    def _calculate_current_margin(self) -> float:
        """Calculate current profit margin"""
        if self.current_price <= 0:
            return 0.0
        return (self.current_price - self.product_config['base_cost']) / self.current_price

    def _get_seasonality_factor(self) -> float:
        """Get seasonal demand factor"""
        # Simple sinusoidal seasonality
        day_of_year = self.current_day % 365
        seasonality = 1 + self.market_config['seasonality_amplitude'] * np.sin(2 * np.pi * day_of_year / 365)
        return seasonality

    def _update_metrics(self, price: float, sales: float, unmet_demand: float):
        """Update performance tracking metrics"""

        revenue = price * sales
        profit = self._calculate_reward(price, sales, unmet_demand)

        self.total_revenue += revenue
        self.total_profit += profit
        self.total_sales += sales

        if self.current_inventory == 0 and unmet_demand > 0:
            self.stockout_days += 1

        if self.current_inventory > self.product_config['max_inventory'] * 0.8:
            self.excess_inventory_days += 1

    def get_performance_summary(self) -> Dict[str, float]:
        """Get comprehensive performance summary"""

        if self.current_day == 0:
            return {}

        return {
            'total_profit': self.total_profit,
            'total_revenue': self.total_revenue,
            'total_sales': self.total_sales,
            'avg_daily_profit': self.total_profit / self.current_day,
            'profit_margin': (self.total_profit / self.total_revenue) if self.total_revenue > 0 else 0,
            'stockout_rate': self.stockout_days / self.current_day,
            'excess_inventory_rate': self.excess_inventory_days / self.current_day,
            'inventory_turnover': (self.total_sales * self.product_config['base_cost']) / (
                        200 * self.product_config['base_cost']) if self.current_day > 0 else 0,
            'final_inventory': self.current_inventory,
            'simulation_days': self.current_day
        }


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = DynamicPricingEnvironment()

    print("🏪 Dynamic Pricing RL Environment Test")
    print("=" * 45)

    # Test reset
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")

    # Test random actions for 10 steps
    print(f"\n🎮 Testing Random Actions:")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print(f"Day {info['day']}: Price=${info['price']:.2f}, Sales={info['sales']:.1f}, Profit=${reward:.2f}")

        if done:
            break

    # Performance summary
    print(f"\n📊 Performance Summary:")
    summary = env.get_performance_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    print(f"\n✅ Environment test completed successfully!")
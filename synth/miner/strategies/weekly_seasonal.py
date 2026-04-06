"""
weekly_seasonal.py — Weekly seasonality + GJR-GARCH strategy.

Wraps synth.miner.core.stock_simulator_v2.simulate_weekly_seasonal_optimized.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.core.stock_simulator_v2 import (
    simulate_weekly_seasonal_optimized,
)

class WeeklySeasonalStrategy(BaseStrategy):
    name = "weekly_seasonal"
    description = (
        "Weekly seasonality (day-of-week + hour) with GJR-GARCH "
        "for stocks and XAU with strong day-of-week patterns"
    )
    supported_asset_types = ["equity", "gold"]
    supported_regimes = ["market_open", "overnight"]
    default_params = {
        "lookback_days": 60,
    }

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        is_high = frequency == "high"
        if is_high:
            return {"lookback_days": [14, 30, 45]}
        return {"lookback_days": [45, 60, 90]}

    def simulate(
        self,
        prices_dict: dict,
        asset: str,
        time_increment: int,
        time_length: int,
        n_sims: int,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> np.ndarray:
        params = self.get_default_params()
        params.update(kwargs)
        return simulate_weekly_seasonal_optimized(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **params,
        )

strategy = WeeklySeasonalStrategy()

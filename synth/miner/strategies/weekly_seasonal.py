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
    supported_assets = ["TSLAX", "AAPLX", "GOOGLX", "SPYX", "NVDAX", "XAU"]
    supported_frequencies = ["high", "low"]
    default_params = {}
    param_grid = {}

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
        return simulate_weekly_seasonal_optimized(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
        )


strategy = WeeklySeasonalStrategy()

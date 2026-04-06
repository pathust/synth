"""
seasonal_stock.py — Intraday seasonality + GARCH strategy.

Wraps synth.miner.core.stock_simulator.simulate_seasonal_stock.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.core.stock_simulator import simulate_seasonal_stock

class SeasonalStockStrategy(BaseStrategy):
    name = "seasonal_stock"
    description = (
        "Intraday seasonality profile with GARCH simulation "
        "(designed for single-stock with strong intraday patterns)"
    )
    supported_asset_types = ["equity"]
    supported_regimes = ["market_open"]
    default_params = {
        "lookback_days": 20,
    }

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        is_high = frequency == "high"
        grid = {}
        if is_high:
            grid["lookback_days"] = [15, 30]
        else:
            grid["lookback_days"] = [45, 60]
        return grid

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
        return simulate_seasonal_stock(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **params,
        )

strategy = SeasonalStockStrategy()

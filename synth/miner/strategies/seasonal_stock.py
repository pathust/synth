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
    param_grid = {
        "lookback_days": [10, 20, 30, 45],
    }

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

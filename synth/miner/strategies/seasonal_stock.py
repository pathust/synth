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
    supported_assets = ["NVDAX"]
    supported_frequencies = ["low"]
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
        return simulate_seasonal_stock(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
        )


strategy = SeasonalStockStrategy()

"""
har_rv.py — HAR-RV + GARCH hybrid strategy.

Wraps synth.miner.core.HAR_RV_simulatior.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.core.HAR_RV_simulatior import (
    simulate_single_price_path_with_har_garch,
)


class HarRvStrategy(BaseStrategy):
    name = "har_rv"
    description = (
        "HAR-RV (Heterogeneous Autoregressive Realized Variance) "
        "combined with GARCH for volatility forecasting"
    )
    supported_assets = ["BTC", "ETH", "SOL", "XAU"]
    supported_frequencies = ["high", "low"]
    default_params = {}
    param_grid = {}  # HAR-RV has few tunable params

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
        return simulate_single_price_path_with_har_garch(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
        )


strategy = HarRvStrategy()

"""
garch_v2.py — Asset-adaptive GARCH with optimal config.

Wraps synth.miner.core.grach_simulator_v2.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.core.grach_simulator_v2 import (
    simulate_single_price_path_with_garch,
)


class GarchV2Strategy(BaseStrategy):
    name = "garch_v2"
    description = (
        "GARCH(1,1) with asset-adaptive lookback window and "
        "variance-targeting initialization"
    )
    supported_assets = []  # all assets
    supported_frequencies = ["high", "low"]
    default_params = {}  # uses get_optimal_config() internally
    param_grid = {
        "lookback_days": [7, 14, 30, 45, 60],
        "vol_multiplier": [0.8, 0.9, 1.0, 1.1],
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
        return simulate_single_price_path_with_garch(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **kwargs,
        )


strategy = GarchV2Strategy()

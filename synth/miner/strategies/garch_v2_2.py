"""
garch_v2_2.py — Asset-adaptive GARCH with optimal config v2.2.

Wraps synth.miner.core.garch_simulator_v2_2.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.core.garch_simulator_v2_2 import (
    simulate_single_price_path_with_garch,
)

class GarchV2_2Strategy(BaseStrategy):
    name = "garch_v2_2"
    description = (
        "GARCH(1,1) with asset-adaptive optimal config specialized for HFT v2.2"
    )
    supported_asset_types = []
    supported_regimes = []
    default_params = {}  # uses get_optimal_config() internally
    param_grid = {
        "lookback_days": [2.5, 3.0, 3.9],
        "vol_multiplier": [0.95, 1.0, 1.02, 1.08],
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

strategy = GarchV2_2Strategy()

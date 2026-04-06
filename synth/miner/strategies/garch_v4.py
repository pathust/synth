"""
garch_v4.py — GJR-GARCH + Skew-t + FHS + regime drift.

Wraps synth.miner.core.grach_simulator_v4.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.core.grach_simulator_v4 import (
    simulate_single_price_path_with_garch,
)

class GarchV4Strategy(BaseStrategy):
    name = "garch_v4"
    description = (
        "GJR-GARCH with Skew Student-t, FHS and regime drift "
        "(Z-Score momentum)"
    )
    supported_asset_types = []
    supported_regimes = []
    default_params = {}

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        from synth.miner.core.grach_simulator_v4 import get_optimal_param_grid
        time_inc = 60 if frequency == "high" else 300
        asset_val = asset if asset else "BTC"
        return get_optimal_param_grid(asset_val, time_inc)

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
        return simulate_single_price_path_with_garch(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **params,
        )

strategy = GarchV4Strategy()

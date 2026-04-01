"""
garch_v4.py — GJR-GARCH + Skew-t + FHS + regime drift.

Wraps synth.miner.strategies.grach_simulator_v4.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.strategies.grach_simulator_v4 import (
    simulate_single_price_path_with_garch,
)


class GarchV4Strategy(BaseStrategy):
    name = "garch_v4"
    description = (
        "GJR-GARCH with Skew Student-t, FHS and regime drift "
        "(Z-Score momentum)"
    )
    supported_assets = []  # all assets
    supported_frequencies = ["high", "low"]
    default_params = {
        "lookback_days": 25,
        "momentum_weight": 0.5,
        "drift_decay": 0.92,
    }
    param_grid = {
        "lookback_days": [7, 14, 25, 45],
        "momentum_weight": [0.2, 0.4, 0.6, 0.8],
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

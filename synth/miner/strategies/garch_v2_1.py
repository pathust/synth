"""
garch_v2_1.py — Regime-aware GJR/GARCH (grach_simulator_v2_1).

Same core as production fallback / entry mapping; supports **kwargs for tuning.
"""

from typing import Optional

import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.core.grach_simulator_v2_1 import (
    simulate_single_price_path_with_garch,
)


class GarchV2_1Strategy(BaseStrategy):
    name = "garch_v2_1"
    description = (
        "GARCH with ER/BBW regime routing, Student-t, optional GJR (grach_simulator_v2_1)"
    )
    supported_asset_types = []
    supported_regimes = []
    default_params = {}

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        from synth.miner.core.grach_simulator_v2_1 import get_optimal_param_grid

        time_inc = 60 if frequency == "high" else 300
        return get_optimal_param_grid(asset or "BTC", time_inc)

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


strategy = GarchV2_1Strategy()

"""
garch_v1.py — GARCH(p,q) Student-t strategy (original).

Wraps synth.miner.core.garch_simulator.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.core.garch_simulator import (
    simulate_single_price_path_with_garch,
)

class GarchV1Strategy(BaseStrategy):
    name = "garch_v1"
    description = "GARCH(p,q) with Student-t innovations (original simulator)"
    supported_asset_types = []
    supported_regimes = []
    default_params = {
        "mean": "Constant",
        "p": 1,
        "q": 1,
    }

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        is_high = frequency == "high"
        grid = {"mean": ["Zero" if is_high else "Constant", "Zero"], "p": [1, 2], "q": [1]}
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
        return simulate_single_price_path_with_garch(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **params,
        )

strategy = GarchV1Strategy()

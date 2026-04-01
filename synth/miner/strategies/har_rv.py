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
    supported_asset_types = ["crypto"]
    supported_regimes = []
    default_params = {
        "lookback_days": 30,
    }
    param_grid = {
        "lookback_days": [30, 45, 60],
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
        return simulate_single_price_path_with_har_garch(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **params,
        )


strategy = HarRvStrategy()

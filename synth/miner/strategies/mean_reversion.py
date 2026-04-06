"""
mean_reversion.py — Ornstein-Uhlenbeck (OU) mean-reversion strategy.

Models prices that tend to revert to a long-run mean. The OU process is:
    dX_t = θ(μ - X_t)dt + σ dW_t

Parameters estimated via MLE from historical log-prices.

Best for: XAU, SPYX — lower volatility, mean-reverting behavior.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"
    description = (
        "Ornstein-Uhlenbeck mean-reversion process — prices revert toward "
        "a long-run mean, good for range-bound assets"
    )
    supported_asset_types = ["gold"]
    supported_regimes = ["mean_reverting"]
    default_params = {
        "lookback_days": 30,
        "half_life_override": None,  # if set, overrides fitted θ
    }

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        is_high = frequency == "high"
        grid = {}
        if is_high:
            grid["lookback_days"] = [7, 14, 21]
            grid["half_life_override"] = [None, 3.0, 7.0]
        else:
            grid["lookback_days"] = [30, 45, 60]
            grid["half_life_override"] = [None, 7.0, 14.0]
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
        from synth.miner.core.mean_reversion_simulator import simulate_mean_reversion
        return simulate_mean_reversion(
            prices_dict=prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **params
        )

strategy = MeanReversionStrategy()

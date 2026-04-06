"""
egarch.py — EGARCH (Exponential GARCH) strategy.

Models volatility in log-space, naturally handling asymmetric effects
(negative shocks increase volatility more than positive ones). No
positivity constraints needed unlike standard GARCH.

Best for: BTC, ETH, SOL — high volatility with asymmetric reactions.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy

class EgarchStrategy(BaseStrategy):
    name = "egarch"
    description = (
        "Exponential GARCH — models asymmetric volatility in log-space, "
        "capturing leverage effects where drops cause higher vol than rallies"
    )
    supported_asset_types = ["crypto"]
    supported_regimes = []
    default_params = {
        "p": 1,
        "q": 1,
        "o": 1,
        "lookback_days": 30,
        "mean_model": "Zero",
        "scale": 10000.0,
    }

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        is_high = frequency == "high"
        grid = {"p": [1, 2], "q": [1]}
        if is_high:
            grid["lookback_days"] = [5, 7, 10]
        else:
            grid["lookback_days"] = [20, 30, 45]
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
        from synth.miner.core.egarch_simulator import simulate_egarch
        return simulate_egarch(
            prices_dict=prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **params
        )

strategy = EgarchStrategy()

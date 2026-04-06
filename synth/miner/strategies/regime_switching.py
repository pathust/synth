"""
regime_switching.py — Markov Regime-Switching GARCH strategy.

Uses regime detection to identify low-vol/high-vol market states, then
applies different GARCH parameters per regime. Blends simulations from
each regime weighted by the current regime probability.

Uses existing regime_detection.py utilities.

Best for: All assets — adapts to changing market conditions.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy

class RegimeSwitchingStrategy(BaseStrategy):
    name = "regime_switching"
    description = (
        "Markov Regime-Switching GARCH — detects market regime "
        "(trending/sideways) and adapts volatility model accordingly"
    )
    supported_asset_types = []
    supported_regimes = []
    default_params = {
        "lookback_days": 30,
        "regime_method": "er",  # "er" or "bbw"
        "scale": 10000.0,
        "trending_vol_mult": 1.2,
        "sideways_vol_mult": 0.8,
    }

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        is_high = frequency == "high"
        grid = {}
        if is_high:
            grid["lookback_days"] = [14, 30]
            grid["vol_ratio"] = [1.5, 2.0]
        else:
            grid["lookback_days"] = [45, 60]
            grid["vol_ratio"] = [2.0, 3.0]
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
        from synth.miner.core.regime_switching_simulator import simulate_regime_switching
        return simulate_regime_switching(
            prices_dict=prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **params
        )

strategy = RegimeSwitchingStrategy()

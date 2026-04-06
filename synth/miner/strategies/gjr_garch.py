"""
gjr_garch.py — GJR-GARCH (Glosten-Jagannathan-Runkle) strategy.

Threshold GARCH that adds an indicator for negative shocks, allowing
different volatility responses to positive vs negative returns.

Best for: Stocks (NVDAX, TSLAX, AAPLX, GOOGLX, SPYX) and XAU.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy

class GjrGarchStrategy(BaseStrategy):
    name = "gjr_garch"
    description = (
        "GJR-GARCH — threshold volatility model with leverage indicator "
        "for asymmetric volatility response to positive/negative shocks"
    )
    supported_asset_types = ["equity", "gold"]
    supported_regimes = []
    default_params = {
        "lookback_days": 45,
        "mean_model": "Constant",
        "scale": 10000.0,
        "vol_multiplier": 1.0,
    }

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        is_high = frequency == "high"
        asset_upper = (asset or "").upper()
        grid = {"p": [1, 2], "q": [1, 2], "o": [1]}
        if is_high:
            grid["lookback_days"] = [5, 7, 10] if asset_upper in ["BTC","ETH","SOL"] else [10, 15]
            grid["dist"] = ["StudentsT"]
        else:
            grid["lookback_days"] = [25, 30, 45] if asset_upper in ["BTC","ETH","SOL"] else [45, 60]
            grid["dist"] = ["StudentsT", "skewstudent"]
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
        from synth.miner.core.gjr_garch_simulator import simulate_gjr_garch
        return simulate_gjr_garch(
            prices_dict=prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **params
        )

strategy = GjrGarchStrategy()

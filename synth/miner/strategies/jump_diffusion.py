"""
jump_diffusion.py — Merton Jump-Diffusion strategy.

Extends GBM with Poisson jumps to capture sudden price discontinuities:
    dS/S = (μ - λk)dt + σ dW + J dN

where J ~ N(jump_mean, jump_std) and N is a Poisson process with
intensity λ.

Parameters estimated from historical kurtosis and skewness of returns.

Best for: BTC, ETH, SOL (sudden crashes/rallies), TSLAX, NVDAX (earnings jumps).
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy

class JumpDiffusionStrategy(BaseStrategy):
    name = "jump_diffusion"
    description = (
        "Merton Jump-Diffusion — GBM with Poisson jumps for capturing "
        "sudden price shocks (flash crashes, earnings, macro events)"
    )
    supported_asset_types = ["crypto", "equity"]
    supported_regimes = []
    default_params = {
        "lookback_days": 30,
        "jump_intensity_override": None,
        "jump_mean_override": None,
        "jump_std_override": None,
        "jump_vol_scale": 1.0,
    }

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        is_high = frequency == "high"
        grid = {}
        if is_high:
            grid["lookback_days"] = [10, 14, 20]
            grid["jump_vol_scale"] = [1.2, 1.5]
        else:
            grid["lookback_days"] = [30, 45, 60]
            grid["jump_vol_scale"] = [1.5, 2.0]
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
        from synth.miner.core.jump_diffusion_simulator import simulate_jump_diffusion
        return simulate_jump_diffusion(
            prices_dict=prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **params
        )

strategy = JumpDiffusionStrategy()

from typing import Optional
import numpy as np
from synth.miner.strategies.base import BaseStrategy

from synth.miner.core.stock_simulator_v4 import simulate_weekly_regime_switching

class WeeklyRegimeSwitchingStrategy(BaseStrategy):
    name = "weekly_regime_switching"
    description = "Regime Switching Strategy augmented with exact weekly trading hours masking."
    supported_asset_types = ["equity"]
    supported_regimes = ["market_open", "overnight", "earnings"]
    default_params = {}

    def simulate(self, prices_dict: dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: Optional[int] = 42, **kwargs) -> np.ndarray:
        return simulate_weekly_regime_switching(
            prices_dict=prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed
        )

strategy = WeeklyRegimeSwitchingStrategy()

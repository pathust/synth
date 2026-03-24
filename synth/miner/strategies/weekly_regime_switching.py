from typing import Optional
import numpy as np
from synth.miner.strategies.base import BaseStrategy

from synth.miner.core.stock_simulator_v4 import simulate_weekly_regime_switching

class WeeklyRegimeSwitchingStrategy(BaseStrategy):
    """
    Weekly Seasonality combined with Regime Switching. Detects trends and adjusts 
    volatility scaling, while successfully bypassing closed stock market hours.
    """
    name = "weekly_regime_switching"
    description = "Regime Switching Strategy augmented with exact weekly trading hours masking."
    supported_assets = ["NVDAX", "TSLAX", "AAPLX", "GOOGLX", "SPYX"]
    supported_frequencies = ["low"]
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

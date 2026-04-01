from typing import Optional
import numpy as np
from synth.miner.strategies.base import BaseStrategy

from synth.miner.core.stock_simulator_v3 import simulate_weekly_garch_v4

class WeeklyGarchV4Strategy(BaseStrategy):
    name = "weekly_garch_v4"
    description = "GARCH V4 blended with Weekly Empirical Seasonality for specific stock hours."
    supported_asset_types = ["equity"]
    supported_regimes = ["market_open", "overnight"]
    default_params = {}

    def simulate(self, prices_dict: dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: Optional[int] = 42, **kwargs) -> np.ndarray:
        return simulate_weekly_garch_v4(
            prices_dict=prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed
        )

strategy = WeeklyGarchV4Strategy()

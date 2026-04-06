from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.core.equity_simulator import simulate_us_equity_exact

class EquityExactHoursStrategy(BaseStrategy):
    name = "equity_exact_hours"
    description = (
        "Strictly enforces US Market hours (09:30-16:00 ET). "
        "Injects Gap variance at market open, zero variance overnight/weekends."
    )
    supported_asset_types = ["equity"]
    supported_regimes = ["market_open", "overnight"]
    default_params = {}

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        is_high = frequency == "high"
        grid = {}
        if is_high:
            grid["gap_ceiling_bps"] = [40.0, 55.0]
        else:
            grid["gap_ceiling_bps"] = [55.0, 80.0, 100.0]
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
        ext_hours = asset in ["NVDAX", "TSLAX", "SPYX"]
        return simulate_us_equity_exact(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            extended_hours=ext_hours,
            **kwargs,
        )

strategy = EquityExactHoursStrategy()

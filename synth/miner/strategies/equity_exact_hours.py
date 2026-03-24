from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.core.equity_simulator import simulate_us_equity_exact

class EquityExactHoursStrategy(BaseStrategy):
    """
    Wraps the US Equity Exact Hours simulator.
    Designed exclusively for stocks to achieve perfect CRPS on weekends.
    """
    name = "equity_exact_hours"
    description = (
        "Strictly enforces US Market hours (09:30-16:00 ET). "
        "Injects Gap variance at market open, zero variance overnight/weekends."
    )
    supported_assets = ["NVDAX", "TSLAX", "AAPLX", "GOOGLX", "SPYX"]
    supported_frequencies = ["low", "high"]
    default_params = {}

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

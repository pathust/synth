from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.core.arima_equity_simulator import simulate_arima_us_equity_exact

class ArimaEquityStrategy(BaseStrategy):
    name = "arima_equity"
    description = (
        "Strictly enforces US Market hours (09:30-16:00 ET). "
        "Intraday prices are modeled using an autoregressive moving average process (ARIMA 1,0,1)."
    )
    supported_asset_types = ["equity"]
    supported_regimes = ["market_open", "overnight"]
    default_params = {}

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        is_high = frequency == "high"
        grid = {"p": [1, 2], "q": [1]}
        if is_high:
            grid["lookback_days"] = [7, 14]
            grid["off_hours_vol_scale"] = [0.1, 0.2]
        else:
            grid["lookback_days"] = [30, 45]
            grid["off_hours_vol_scale"] = [0.2, 0.3]
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
        # Extended hours (04:00–20:00 ET): NVDAX, TSLAX, SPYX
        # Regular hours (09:30–16:00 ET): AAPLX, GOOGLX
        #   → AAPLX "đầm" hơn, gap mở phiên nhỏ → Regular hours là chuẩn
        ext_hours = asset in ["NVDAX", "TSLAX", "SPYX"]
        return simulate_arima_us_equity_exact(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            extended_hours=ext_hours,
            **kwargs,
        )

strategy = ArimaEquityStrategy()

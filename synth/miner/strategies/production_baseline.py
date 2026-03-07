"""
production_baseline.py — Wraps the current production generate_simulations()
logic as a BaseStrategy, so it can be used as baseline in duel comparisons.

This replicates the exact asset-routing and fallback chain from
synth/miner/simulations.py:generate_simulations().
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy

# Import the exact same simulate functions used in production
from synth.miner.core.garch_simulator import (
    simulate_single_price_path_with_garch as garch_v1,
)
from synth.miner.core.grach_simulator_v2 import (
    simulate_single_price_path_with_garch as garch_v2,
)
from synth.miner.core.grach_simulator_v2_1 import (
    simulate_single_price_path_with_garch as garch_v2_1,
)
from synth.miner.core.HAR_RV_simulatior import (
    simulate_single_price_path_with_har_garch as har_garch,
)
from synth.miner.core.stock_simulator import simulate_seasonal_stock
from synth.miner.core.stock_simulator_v2 import simulate_weekly_seasonal_optimized


def _get_production_fn_chain(asset: str) -> list[tuple]:
    """
    Return the exact same fallback chain as generate_simulations().
    Each entry is (simulate_fn, max_data_points).
    """
    if asset in ["TSLAX", "AAPLX", "GOOGLX", "XAU"]:
        return [(simulate_weekly_seasonal_optimized, None)]

    if asset in ["NVDAX"]:
        return [(simulate_seasonal_stock, None)]

    if asset in ["BTC", "ETH", "SOL"]:
        return [
            (garch_v2, None),
            (garch_v1, 500),
            (har_garch, 100000),
        ]

    # Fallback for other assets
    return [
        (garch_v2, None),
        (har_garch, 100000),
        (garch_v1, 500),
    ]


class ProductionBaselineStrategy(BaseStrategy):
    """
    Wraps the current production generate_simulations() logic.
    Uses the exact same asset-routing and fallback chain.
    """

    name = "production_baseline"
    description = (
        "Current production code (generate_simulations) — "
        "garch_v2 for crypto, weekly_seasonal for stocks/XAU"
    )
    supported_assets = []  # all
    supported_frequencies = ["high", "low"]
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
        fn_chain = _get_production_fn_chain(asset)

        for simulate_fn, max_data_points in fn_chain:
            try:
                # Apply max_data_points filtering (same as production)
                filtered = prices_dict
                if max_data_points is not None:
                    sorted_items = sorted(
                        prices_dict.items(), key=lambda x: int(x[0])
                    )
                    filtered = dict(sorted_items[-max_data_points:])

                paths = simulate_fn(
                    filtered,
                    asset=asset,
                    time_increment=time_increment,
                    time_length=time_length,
                    n_sims=n_sims,
                    seed=seed,
                    **kwargs,
                )

                if paths is not None and len(paths) > 0:
                    return paths

            except Exception as e:
                print(
                    f"  [production_baseline] {simulate_fn.__name__} failed: {e}"
                )
                continue

        raise RuntimeError(
            f"All production fallback functions failed for {asset}"
        )


strategy = ProductionBaselineStrategy()

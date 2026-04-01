"""
ensemble_garch_v2_v4.py — Fixed-weight ensemble of garch_v2 and garch_v4.

Strategy-level ensemble that combines the two core simulators:
- grach_simulator_v2  → strategy name \"garch_v2\"
- grach_simulator_v4  → strategy name \"garch_v4\"

It reuses EnsembleWeightedStrategy under the hood with
base_strategy_names = [\"garch_v2\", \"garch_v4\"] and equal weights
by default.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.strategies.ensemble_weighted import EnsembleWeightedStrategy


class EnsembleGarchV2V4Strategy(BaseStrategy):
    name = "ensemble_garch_v2_v4"
    description = (
        "Ensemble of garch_v2 (asset-adaptive GARCH) and "
        "garch_v4 (regime-aware GJR-GARCH + skew-t/FHS) with configurable weights"
    )
    supported_assets: list[str] = []  # all assets
    supported_frequencies: list[str] = ["high", "low"]
    # Only expose weights; base_strategy_names cố định là garch_v2 + garch_v4.
    default_params: dict = {
        "weights": [0.5, 0.5],
    }
    param_grid: dict = {
        "weights": [
            [0.5, 0.5],
            [0.4, 0.6],
            [0.6, 0.4],
        ],
    }

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
        # Merge default params with any overrides (e.g. custom weights).
        params = self.get_default_params()
        params.update(kwargs)

        # Delegate to EnsembleWeightedStrategy with fixed base_strategy_names.
        inner = EnsembleWeightedStrategy()
        return inner.simulate(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            base_strategy_names=["garch_v2", "garch_v4"],
            **params,
        )


strategy = EnsembleGarchV2V4Strategy()


"""
builder.py — Ensemble path construction from strategy configs.

Extracted from simulations_new_v3.py's _run_ensemble(). Uses StrategyConfig
objects and the strategy registry to build ensemble paths.
"""

from __future__ import annotations

import hashlib
import numpy as np
from typing import Optional, Callable

from synth.miner.strategies.base import StrategyConfig
from synth.miner.config.defaults import ENSEMBLE_TOP_N, OVER_REQUEST_FACTOR
from synth.miner.ensemble.trimmer import OutlierTrimmer


def _make_sub_seed(base_seed: int, strategy_name: str) -> int:
    """Deterministic but unique seed per strategy for independent innovations."""
    h = hashlib.sha256(strategy_name.encode()).hexdigest()
    # Convert hex string to integer and modulo it to fit 32-bit signed int range
    sub_seed = int(h, 16) % 1_000_000
    return (base_seed + sub_seed) & 0x7FFF_FFFF


class EnsembleBuilder:
    """
    Build ensemble simulation paths from multiple strategy configs.

    Given a list of StrategyConfig (with weights), runs each strategy
    with proportional path allocation, then trims outliers and selects
    the final ensemble.
    """

    def __init__(
        self,
        top_n: int = ENSEMBLE_TOP_N,
        over_request: float = OVER_REQUEST_FACTOR,
        trimmer: Optional[OutlierTrimmer] = None,
    ):
        self.top_n = top_n
        self.over_request = over_request
        self.trimmer = trimmer or OutlierTrimmer()

    def build(
        self,
        strategy_configs: list[StrategyConfig],
        simulate_fn_map: dict[str, Callable],
        prices_dict: dict[str, float],
        asset: str,
        start_time: str,
        time_increment: int,
        time_length: int,
        num_simulations: int,
        seed: int = 42,
    ) -> Optional[np.ndarray]:
        """
        Run ensemble of strategies and combine paths.

        Args:
            strategy_configs: List of StrategyConfig with weights.
            simulate_fn_map: Map of strategy_name -> simulate callable.
            prices_dict: Historical prices (already filtered).
            asset: Asset name.
            start_time: ISO format start time.
            time_increment: Seconds per step.
            time_length: Total sim duration in seconds.
            num_simulations: Target number of output paths.
            seed: Random seed.

        Returns:
            np.ndarray of shape (num_simulations, steps+1), or None on failure.
        """
        n_strategies = min(len(strategy_configs), self.top_n)
        active = strategy_configs[:n_strategies]

        # Normalize weights
        total_weight = sum(sc.weight for sc in active)
        if total_weight <= 0:
            return None
        normalized = [(sc, sc.weight / total_weight) for sc in active]

        # Over-request for trimming headroom
        target_total = int(num_simulations * self.over_request)

        all_paths: list[np.ndarray] = []
        used: list[str] = []
        sims_collected = 0

        for i, (sc, norm_weight) in enumerate(normalized):
            fn = simulate_fn_map.get(sc.strategy_name)
            if fn is None:
                print(f"[Ensemble] Strategy '{sc.strategy_name}' not found in map, skipping")
                continue

            # Allocate paths proportional to weight
            if i == len(normalized) - 1:
                n_sub = target_total - sims_collected
            else:
                n_sub = int(target_total * norm_weight)

            if n_sub <= 0:
                continue

            sub_seed = _make_sub_seed(seed, sc.strategy_name)

            try:
                paths = fn(
                    prices_dict,
                    asset=asset,
                    time_increment=time_increment,
                    time_length=time_length,
                    n_sims=n_sub,
                    seed=sub_seed,
                    **sc.params,
                )
                if (
                    paths is not None
                    and isinstance(paths, np.ndarray)
                    and paths.ndim == 2
                    and paths.shape[0] > 0
                ):
                    all_paths.append(paths)
                    sims_collected += paths.shape[0]
                    used.append(f"{sc.strategy_name}({paths.shape[0]})")
            except Exception as e:
                print(f"[Ensemble] {sc.strategy_name} failed: {e}")

        if not all_paths:
            return None

        combined = np.vstack(all_paths)

        # Trim outliers and select final ensemble
        final = self.trimmer.trim(combined, num_simulations, seed)

        print(
            f"[Ensemble] {asset}: {' + '.join(used)} => "
            f"Trimmed & Selected {final.shape[0]} paths"
        )
        return final

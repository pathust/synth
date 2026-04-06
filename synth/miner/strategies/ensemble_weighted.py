"""
ensemble_weighted.py — Adaptive Weighted Ensemble strategy.

Combines multiple base strategies by allocating n_sims across them
proportionally to weights. Weights can be set manually or will
default to equal weighting.

Best for: All assets — diversifies model risk.
"""

from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy

class EnsembleWeightedStrategy(BaseStrategy):
    name = "ensemble_weighted"
    description = (
        "Adaptive weighted ensemble — blends paths from multiple base "
        "strategies with configurable weights to diversify model risk"
    )
    supported_asset_types = []
    supported_regimes = []
    default_params = {
        # base_strategy_names will be resolved at runtime
        "base_strategy_names": ["garch_v1", "garch_v2"],
        "weights": None,  # None = equal weights
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
        params = self.get_default_params()
        params.update(kwargs)

        base_names = params["base_strategy_names"]
        weights = params["weights"]

        if not base_names:
            raise ValueError("ensemble_weighted requires base_strategy_names")

        # Import registry lazily to avoid circular imports
        from synth.miner.strategies.registry import StrategyRegistry

        registry = StrategyRegistry()
        registry.auto_discover()

        # Filter to strategies that actually support this asset
        available = []
        for name in base_names:
            try:
                strat = registry.get(name)
                if strat.supports_asset(asset):
                    available.append(strat)
            except KeyError:
                print(f"[Ensemble] Strategy '{name}' not found, skipping")

        if not available:
            raise ValueError(
                f"No available base strategies for asset={asset} "
                f"from {base_names}"
            )

        # Default equal weights
        if weights is None:
            weights = [1.0 / len(available)] * len(available)
        elif len(weights) != len(available):
            weights = [1.0 / len(available)] * len(available)

        # Normalize weights
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        # ── Allocate n_sims proportionally ──
        sims_per_strategy = []
        remaining = n_sims
        for i, w in enumerate(weights):
            if i == len(weights) - 1:
                sims_per_strategy.append(remaining)
            else:
                n = max(1, int(n_sims * w))
                sims_per_strategy.append(n)
                remaining -= n

        # ── Run each base strategy and concatenate paths ──
        all_paths = []
        for strat, n_s in zip(available, sims_per_strategy):
            sub_seed = None
            if seed is not None:
                import hashlib
                h = hashlib.sha256(strat.name.encode()).hexdigest()
                sub_seed = (seed + int(h, 16) % 1_000_000) & 0x7FFF_FFFF
            try:
                paths = strat.simulate(
                    prices_dict,
                    asset=asset,
                    time_increment=time_increment,
                    time_length=time_length,
                    n_sims=n_s,
                    seed=sub_seed,
                )
                if paths is not None and len(paths) > 0:
                    all_paths.append(paths)
                    print(
                        f"[Ensemble] {strat.name}: {paths.shape[0]} paths OK"
                    )
            except Exception as e:
                print(f"[Ensemble] {strat.name} failed: {e}")

        if not all_paths:
            raise RuntimeError("All base strategies failed in ensemble")

        # Concatenate all paths along the simulation axis
        combined = np.vstack(all_paths)

        # If we got more or fewer than n_sims, randomly subsample/resample
        if combined.shape[0] != n_sims:
            indices = np.random.choice(
                combined.shape[0], size=n_sims, replace=True
            )
            combined = combined[indices]

        return combined

strategy = EnsembleWeightedStrategy()

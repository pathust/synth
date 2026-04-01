"""
strategies/ — Modular strategy registry for price simulation.

Usage:
    from synth.miner.strategies import StrategyRegistry, StrategyConfig

    registry = StrategyRegistry()
    registry.auto_discover()

    strategy = registry.get("garch_v2")
    paths = strategy.simulate(prices_dict, "BTC", 300, 86400, 100, seed=42)
"""

from synth.miner.strategies.base import BaseStrategy, StrategyConfig
from synth.miner.strategies.registry import StrategyRegistry

__all__ = ["BaseStrategy", "StrategyConfig", "StrategyRegistry"]


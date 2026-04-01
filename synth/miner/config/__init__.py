"""
config/ — Centralized configuration for the miner module.

Replaces hardcoded strategy-asset mappings with structured StrategyConfig objects.
"""

from synth.miner.config.asset_strategy_config import (
    PRODUCTION_CONFIG,
    DEFAULT_FALLBACK_CHAIN,
    get_strategy_list,
)
from synth.miner.config.defaults import ENSEMBLE_TOP_N, TRIM_LOWER, TRIM_UPPER

__all__ = [
    "PRODUCTION_CONFIG",
    "DEFAULT_FALLBACK_CHAIN",
    "get_strategy_list",
    "ENSEMBLE_TOP_N",
    "TRIM_LOWER",
    "TRIM_UPPER",
]

"""
base.py — Abstract base class for all simulation strategies.

Every strategy must subclass BaseStrategy and implement `simulate()`.
The registry uses the class-level attributes to filter strategies
by asset_type and regime, following the Asset × Regime architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


ASSET_TYPES = ["crypto", "gold", "equity"]
REGIME_TYPES = {
    "crypto": ["bull", "high_vol", "ranging"],
    "gold": ["mean_reverting", "trending"],
    "equity": ["market_open", "overnight", "earnings"],
}

CRYPTO_ASSETS = ["BTC", "ETH", "SOL"]
GOLD_ASSETS = ["XAU"]
EQUITY_ASSETS = ["SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]


def get_asset_type(asset: str) -> str:
    if asset in CRYPTO_ASSETS:
        return "crypto"
    elif asset in GOLD_ASSETS:
        return "gold"
    elif asset in EQUITY_ASSETS:
        return "equity"
    return "crypto"


@dataclass
class StrategyConfig:
    """
    Configuration for deploying a strategy to a specific asset_type/regime.

    Attributes:
        strategy_name: Registry name of the strategy (e.g., "garch_v2").
        weight: Ensemble weight in [0, 1]. Weights are normalized at runtime.
        params: Override parameters passed as **kwargs to strategy.simulate().
    """
    strategy_name: str
    weight: float = 1.0
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be in [0, 1], got {self.weight}")

    def to_tuple(self) -> tuple[str, float]:
        return (self.strategy_name, self.weight)


class BaseStrategy(ABC):
    """Base class for all price simulation strategies."""

    name: str = ""
    version: str = "1.0"
    description: str = ""
    supported_asset_types: list[str] = []
    supported_regimes: list[str] = []
    supported_frequencies: list[str] = []
    default_params: dict = {}
    high_param_grid: dict = {}
    low_param_grid: dict = {}

    @abstractmethod
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
        """
        Run simulation and return price paths.

        Args:
            prices_dict: {timestamp_str: price_float} historical data
            asset: Asset name (e.g. "BTC")
            time_increment: Seconds per step (60 or 300)
            time_length: Total sim duration in seconds (3600 or 86400)
            n_sims: Number of simulation paths
            seed: Random seed for reproducibility

        Returns:
            np.ndarray of shape (n_sims, steps+1) — price paths
        """
        ...

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        if frequency == "high" and self.high_param_grid:
            return self.high_param_grid
        elif frequency == "low" and self.low_param_grid:
            return self.low_param_grid
        return {}

    def get_default_params(self) -> dict:
        return self.default_params.copy()

    def supports_asset(self, asset: str) -> bool:
        if not self.supported_asset_types:
            return True
        asset_type = get_asset_type(asset)
        return asset_type in self.supported_asset_types

    def supports_regime(self, regime: str) -> bool:
        if not self.supported_regimes:
            return True
        return regime in self.supported_regimes

    def supports_frequency(self, frequency: str) -> bool:
        if not self.supported_frequencies:
            return True
        return frequency in self.supported_frequencies

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name='{self.name}' "
            f"asset_types={self.supported_asset_types} "
            f"regimes={self.supported_regimes}>"
        )
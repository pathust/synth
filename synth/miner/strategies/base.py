"""
base.py — Abstract base class for all simulation strategies.

Every strategy must subclass BaseStrategy and implement `simulate()`.
The registry uses the class-level attributes to filter strategies
by asset and frequency.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseStrategy(ABC):
    """Base class for all price simulation strategies."""

    # --- Must be set by subclasses ---
    name: str = ""                          # e.g. "garch_v2"
    description: str = ""                   # Human-readable description
    supported_assets: list[str] = []        # e.g. ["BTC","ETH"]; empty = all
    supported_frequencies: list[str] = []   # ["high","low"]; empty = all
    default_params: dict = {}               # Default hyperparameters
    param_grid: dict = {}                   # Param ranges for GridSearch

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

    def get_param_grid(self) -> dict:
        """Return parameter grid for tuning. Override to customize."""
        return self.param_grid

    def get_default_params(self) -> dict:
        """Return default parameters. Override to customize."""
        return self.default_params.copy()

    def supports_asset(self, asset: str) -> bool:
        """Check if this strategy supports the given asset."""
        if not self.supported_assets:
            return True  # empty list = supports all
        return asset in self.supported_assets

    def supports_frequency(self, frequency: str) -> bool:
        """Check if this strategy supports the given frequency label."""
        if not self.supported_frequencies:
            return True  # empty list = supports all
        return frequency in self.supported_frequencies

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name='{self.name}' "
            f"assets={self.supported_assets} freq={self.supported_frequencies}>"
        )

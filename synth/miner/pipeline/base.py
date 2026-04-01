"""
base.py — Abstract base class for data providers.

All data sources (MySQL, CoinMetrics, Pyth) implement this interface
so they can be composed in PriceLoader.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


class DataProvider(ABC):
    """
    Abstract base for all price data sources.

    Implementations must fetch historical price data and return it
    as a dict mapping timestamp strings to float prices.
    """

    @abstractmethod
    def fetch(
        self,
        asset: str,
        start: Optional[datetime],
        end: Optional[datetime],
        resolution: str = "5m",
    ) -> dict[str, float]:
        """
        Fetch price data for the given asset and time range.

        Args:
            asset: Asset symbol (e.g., "BTC", "ETH", "NVDAX").
            start: Start of the time range (inclusive). None = earliest available.
            end: End of the time range (inclusive). None = latest available.
            resolution: Time resolution ("1m" or "5m").

        Returns:
            Dict mapping timestamp strings to float prices,
            e.g., {"1711234500": 67123.45, ...}

        Must be idempotent — calling with the same args returns the same result.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this data provider (e.g., 'MySQL', 'CoinMetrics')."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"

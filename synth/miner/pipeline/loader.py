"""
loader.py — Unified PriceLoader that replaces scattered data loading logic.

Replaces my_simulation.py's fetch_price_data() and data_handler.py's dispatch
with a clean, composable interface. Enforces anti-leakage guard by default.
"""

from __future__ import annotations

import datetime
from typing import Optional

from synth.miner.pipeline.base import DataProvider


class PriceLoader:
    """
    Unified price data loader.

    Composes multiple DataProvider instances and provides a single
    `load()` method with built-in anti-leakage protection.

    Usage:
        loader = PriceLoader(providers=[MySQLProvider()])
        prices = loader.load("BTC", before=datetime(...), resolution="5m")
    """

    def __init__(
        self,
        providers: list[DataProvider] | None = None,
        max_data_points: int = 100_000,
    ):
        if providers is None:
            # Default: MySQL only (fast, local)
            from synth.miner.pipeline.providers.mysql_provider import MySQLProvider
            providers = [MySQLProvider()]

        self._providers = providers
        self._max_data_points = max_data_points

    def load(
        self,
        asset: str,
        before: datetime.datetime,
        resolution: str = "5m",
        max_points: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Load historical prices strictly BEFORE the given timestamp.

        This is the core anti-leakage guard: no price at or after `before`
        will ever be returned to a strategy.

        Args:
            asset: Asset symbol (e.g., "BTC").
            before: Cutoff timestamp — only prices with ts < before are returned.
            resolution: Time resolution ("1m" or "5m").
            max_points: Max data points to return (most recent). Defaults to
                        self._max_data_points.

        Returns:
            Dict of {timestamp_str: price}, sorted ascending, truncated to
            max_points most recent entries.
        """
        max_pts = max_points or self._max_data_points
        cutoff_ts = int(before.timestamp())

        # Try providers in order, merge results
        all_prices: dict[str, float] = {}
        for provider in self._providers:
            try:
                data = provider.fetch(asset, start=None, end=before, resolution=resolution)
                if data:
                    all_prices.update(data)
                    break  # Use first successful provider
            except Exception as e:
                print(f"[PriceLoader] {provider.name} failed for {asset}: {e}")
                continue

        if not all_prices:
            return {}

        # Anti-leakage: strict filter — only prices BEFORE cutoff
        filtered = {
            k: v for k, v in all_prices.items()
            if int(k) < cutoff_ts
        }

        if not filtered:
            return {}

        # Sort ascending and take last max_points
        sorted_items = sorted(filtered.items(), key=lambda x: int(x[0]))
        if max_pts is not None and len(sorted_items) > max_pts:
            sorted_items = sorted_items[-max_pts:]

        return dict(sorted_items)

    def load_raw(
        self,
        asset: str,
        resolution: str = "5m",
    ) -> dict[str, float]:
        """
        Load all available data without any filtering.
        Used for data inspection/debugging only — NOT for strategies.
        """
        for provider in self._providers:
            try:
                data = provider.fetch(asset, start=None, end=None, resolution=resolution)
                if data:
                    return data
            except Exception as e:
                print(f"[PriceLoader] {provider.name} failed: {e}")
                continue
        return {}

    def preload(
        self,
        asset: str,
        resolution: str = "5m",
    ) -> int:
        """
        Preload data for an asset (warm the cache).
        Returns the number of data points loaded.
        """
        data = self.load_raw(asset, resolution)
        return len(data)

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self._providers]

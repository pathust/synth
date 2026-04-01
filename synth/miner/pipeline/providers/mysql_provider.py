"""
mysql_provider.py — DataProvider wrapping the existing MySQLHandler.

This is the primary data source for backtesting — it reads from the local
MySQL database that the fetch daemon populates.
"""

from datetime import datetime
from typing import Optional

from synth.miner.pipeline.base import DataProvider


class MySQLProvider(DataProvider):
    """Load price data from the local MySQL database."""

    def __init__(self):
        from synth.miner.data_handler import DataHandler
        self._handler = DataHandler()

    @property
    def name(self) -> str:
        return "MySQL"

    def fetch(
        self,
        asset: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        resolution: str = "5m",
    ) -> dict[str, float]:
        """
        Load price data from MySQL via the existing DataHandler.

        Optionally filters by start/end timestamps.
        """
        hist = self._handler.load_price_data(asset, resolution)
        if not hist or resolution not in hist:
            return {}

        prices = hist[resolution]

        # Apply time filters if provided
        if start is not None or end is not None:
            start_ts = int(start.timestamp()) if start else 0
            end_ts = int(end.timestamp()) if end else int(1e12)
            prices = {
                k: v for k, v in prices.items()
                if start_ts <= int(k) <= end_ts
            }

        return prices

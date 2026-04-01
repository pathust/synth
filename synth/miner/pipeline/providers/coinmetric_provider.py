"""
coinmetric_provider.py — DataProvider wrapping the CoinMetrics API client.

Secondary data source; useful for crypto assets.
"""

from datetime import datetime
from typing import Optional

from synth.miner.pipeline.base import DataProvider


class CoinMetricProvider(DataProvider):
    """Fetch price data from the CoinMetrics API."""

    def __init__(self):
        from synth.miner.data_handler import DataHandler
        self._handler = DataHandler()

    @property
    def name(self) -> str:
        return "CoinMetrics"

    def fetch(
        self,
        asset: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        resolution: str = "5m",
    ) -> dict[str, float]:
        """Fetch from CoinMetrics API via the existing DataHandler."""
        from synth.miner.constants import ASSETS_PERIODIC_FETCH_PRICE_DATA

        # CoinMetrics doesn't support periodic-fetch assets (stocks)
        if asset in ASSETS_PERIODIC_FETCH_PRICE_DATA:
            return {}

        if start is None or end is None:
            return {}

        # Map resolution to time_increment
        res_map = {"1m": 60, "5m": 300}
        time_increment = res_map.get(resolution, 300)

        try:
            return self._handler.fetch_cm_data(asset, start, end, time_increment)
        except Exception as e:
            print(f"[CoinMetricProvider] Error: {e}")
            return {}

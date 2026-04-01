"""
pyth_provider.py — DataProvider wrapping the Pyth TradingView API.

The "Synth" data source — fetches from the PriceDataProvider.
"""

from datetime import datetime
from typing import Optional

from synth.miner.pipeline.base import DataProvider


class PythProvider(DataProvider):
    """Fetch price data from Pyth TradingView API (Synth's price data provider)."""

    def __init__(self):
        from synth.miner.data_handler import DataHandler
        self._handler = DataHandler()

    @property
    def name(self) -> str:
        return "Pyth"

    def fetch(
        self,
        asset: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        resolution: str = "5m",
    ) -> dict[str, float]:
        """Fetch from Pyth API via the existing DataHandler."""
        if start is None or end is None:
            return {}

        # Map resolution to time_increment
        res_map = {"1m": 60, "5m": 300}
        time_increment = res_map.get(resolution, 300)
        time_length = int((end - start).total_seconds())

        try:
            return self._handler.fetch_synth_data(
                asset, start, time_length, time_increment
            )
        except Exception as e:
            print(f"[PythProvider] Error: {e}")
            return {}

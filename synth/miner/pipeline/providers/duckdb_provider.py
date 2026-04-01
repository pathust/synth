from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from synth.miner.pipeline.base import DataProvider


class DuckDBProvider(DataProvider):
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.getenv("SYNTH_DUCKDB_PATH", "data/market_data.duckdb")
        self._db_path = db_path

    @property
    def name(self) -> str:
        return "DuckDB"

    def fetch(
        self,
        asset: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        resolution: str = "5m",
    ) -> dict[str, float]:
        from synth.miner.data.storage.duckdb_store import DuckDBMarketStore

        if not os.path.exists(self._db_path):
            return {}
        store = DuckDBMarketStore(self._db_path)
        return store.read_prices(symbol=asset, start=start, end=end)

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("duckdb")

from synth.miner.data.storage.duckdb_store import DuckDBMarketStore
from synth.miner.data.storage.duckdb_sync import sync_price_dict_to_duckdb


def test_sync_price_dict_to_duckdb(tmp_path):
    db_path = tmp_path / "market_data.duckdb"
    base_ts = int(datetime.now(timezone.utc).timestamp())
    prices = {
        str(base_ts - 180): 10.0,
        str(base_ts - 120): 11.0,
        str(base_ts - 60): 12.0,
    }
    inserted = sync_price_dict_to_duckdb(
        symbol="BTC",
        prices=prices,
        source="unit_test",
        max_rows=2,
        db_path=str(db_path),
    )
    assert inserted == 2
    store = DuckDBMarketStore(str(db_path))
    out = store.read_prices("BTC")
    assert len(out) == 2
    assert list(out.values()) == [11.0, 12.0]

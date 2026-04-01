from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

duckdb = pytest.importorskip("duckdb")

from synth.miner.data.storage.duckdb_store import DuckDBMarketStore, RawPriceRow
from synth.miner.pipeline.providers.duckdb_provider import DuckDBProvider


def test_duckdb_store_read_write(tmp_path):
    db_path = tmp_path / "market_data.duckdb"
    store = DuckDBMarketStore(str(db_path))
    now = datetime.now(timezone.utc).replace(microsecond=0)
    rows = [
        RawPriceRow(symbol="BTC", timestamp=now - timedelta(minutes=2), price=100.0, source="test"),
        RawPriceRow(symbol="BTC", timestamp=now - timedelta(minutes=1), price=101.0, source="test"),
    ]
    store.write_batch(rows)
    data = store.read_prices("BTC")
    assert len(data) == 2
    assert list(data.values()) == [100.0, 101.0]


def test_duckdb_provider_fetch(tmp_path):
    db_path = tmp_path / "market_data.duckdb"
    store = DuckDBMarketStore(str(db_path))
    now = datetime.now(timezone.utc).replace(microsecond=0)
    store.write_batch(
        [
            RawPriceRow(symbol="ETH", timestamp=now - timedelta(minutes=2), price=200.0, source="test"),
            RawPriceRow(symbol="ETH", timestamp=now - timedelta(minutes=1), price=201.0, source="test"),
        ]
    )
    provider = DuckDBProvider(db_path=str(db_path))
    out = provider.fetch("ETH", start=now - timedelta(minutes=3), end=now, resolution="1m")
    assert len(out) == 2

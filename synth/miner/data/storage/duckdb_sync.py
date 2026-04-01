from __future__ import annotations

import os
from datetime import datetime, timezone

from synth.miner.data.storage.duckdb_store import DuckDBMarketStore, RawPriceRow


def _to_datetime(ts: str) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(tzinfo=None)


def sync_price_dict_to_duckdb(
    symbol: str,
    prices: dict[str, float],
    source: str = "mysql_sync",
    max_rows: int = 500,
    db_path: str | None = None,
) -> int:
    if not prices:
        return 0
    if db_path is None:
        db_path = os.getenv("SYNTH_DUCKDB_PATH", "data/market_data.duckdb")
    items = sorted(prices.items(), key=lambda x: int(x[0]))
    if max_rows and len(items) > max_rows:
        items = items[-max_rows:]
    rows = [
        RawPriceRow(
            symbol=symbol,
            timestamp=_to_datetime(ts),
            price=float(price),
            source=source,
        )
        for ts, price in items
    ]
    store = DuckDBMarketStore(db_path)
    store.write_batch(rows)
    return len(rows)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

try:
    import duckdb
except Exception:
    duckdb = None


@dataclass
class RawPriceRow:
    symbol: str
    timestamp: datetime
    price: float
    volume: float = 0.0
    source: str = "unknown"


class DuckDBMarketStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        if duckdb is None:
            raise RuntimeError("duckdb package is required for DuckDBMarketStore")

    def ensure_schema(self) -> None:
        with duckdb.connect(self.db_path, read_only=False) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_prices (
                    symbol VARCHAR,
                    timestamp TIMESTAMP,
                    price DOUBLE,
                    volume DOUBLE,
                    source VARCHAR
                )
                """
            )

    def write_batch(self, rows: list[RawPriceRow]) -> None:
        if not rows:
            return
        self.ensure_schema()
        payload = [
            (r.symbol, r.timestamp, float(r.price), float(r.volume), r.source)
            for r in rows
        ]
        with duckdb.connect(self.db_path, read_only=False) as con:
            con.executemany(
                "INSERT INTO raw_prices(symbol, timestamp, price, volume, source) VALUES (?, ?, ?, ?, ?)",
                payload,
            )

    def read_prices(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> dict[str, float]:
        self.ensure_schema()
        clauses = ["symbol = ?"]
        params: list = [symbol]
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end)
        where = " AND ".join(clauses)
        query = f"SELECT timestamp, price FROM raw_prices WHERE {where} ORDER BY timestamp ASC"
        out: dict[str, float] = {}
        with duckdb.connect(self.db_path, read_only=True) as con:
            rows = con.execute(query, params).fetchall()
        for ts, price in rows:
            out[str(int(ts.timestamp()))] = float(price)
        return out

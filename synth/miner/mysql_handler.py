"""
mysql_handler.py

Storage handler for price data. Uses SQLite for local/backtest mode.

Originally designed as MySQL replacement for sn50's MongoDBHandler.
Switched to SQLite because:
    - Python 3.14 has auth compatibility issues with pymysql/MariaDB
    - SQLite requires zero setup (no Docker needed)
    - Perfect for local backtest use case
    - Same interface: save_price_data(), load_price_data()

Schema:
    CREATE TABLE price_data (
        asset TEXT NOT NULL,
        time_frame TEXT NOT NULL, 
        prices TEXT NOT NULL,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(asset, time_frame)
    );

Data format matches sn50 MongoDB:
    {"5m": {"timestamp": "price", ...}}
"""

import os
import json
import sqlite3
from dotenv import load_dotenv

load_dotenv()

# Default SQLite database path
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "price_data.db"
)


class MySQLHandler:
    """
    SQLite-based price data handler (named MySQLHandler for compatibility).
    Replaces MongoDBHandler from sn50.

    Same public interface:
        - save_price_data(asset, time_frame, prices)
        - load_price_data(asset, time_frame) -> dict with "prices" key or {}
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH)
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_table()

    def _get_connection(self):
        """Get a new SQLite connection."""
        return sqlite3.connect(self.db_path)

    def _ensure_table(self):
        """Create the price_data table if it doesn't exist."""
        try:
            conn = self._get_connection()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS price_data (
                        asset TEXT NOT NULL,
                        time_frame TEXT NOT NULL,
                        prices TEXT NOT NULL,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(asset, time_frame)
                    )
                """)
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            print(f"[WARN] SQLite table creation failed: {e}")

    def save_price_data(self, asset: str, time_frame: str, prices: dict):
        """
        Save price data using upsert (INSERT OR REPLACE).

        Equivalent to sn50 MongoDBHandler.save_price_data() (L421-437).

        Args:
            asset: Asset name (e.g., "BTC", "ETH")
            time_frame: Time frame (e.g., "5m", "60")
            prices: Dict of prices {time_frame: {timestamp: price, ...}}
        """
        try:
            conn = self._get_connection()
            try:
                prices_json = json.dumps(prices, ensure_ascii=False)
                conn.execute("""
                    INSERT OR REPLACE INTO price_data (asset, time_frame, prices, updated_at)
                    VALUES (?, ?, ?, datetime('now'))
                """, (asset, time_frame, prices_json))
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            print(f"[ERROR] Failed to save price data to SQLite: {e}")

    def load_price_data(self, asset: str, time_frame: str) -> dict:
        """
        Load price data from SQLite.

        Equivalent to sn50 MongoDBHandler.load_price_data() (L439-444).

        Returns:
            dict: {"prices": {time_frame: {timestamp: price}}} or {} if not found.
        """
        try:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT prices FROM price_data WHERE asset = ? AND time_frame = ?",
                    (asset, time_frame)
                )
                row = cursor.fetchone()
                if row:
                    prices = json.loads(row[0])
                    return {"prices": prices}
                return {}
            finally:
                conn.close()
        except Exception as e:
            print(f"[ERROR] Failed to load price data from SQLite: {e}")
            return {}

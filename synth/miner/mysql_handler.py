"""
mysql_handler.py

Storage handler for price data using MySQL (MariaDB).

Uses pymysql to connect to MariaDB defined in docker-compose.yaml (service: mysql).

Connection config from .env:
    MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD

Schema:
    CREATE TABLE IF NOT EXISTS price_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        asset VARCHAR(20) NOT NULL,
        time_frame VARCHAR(20) NOT NULL,
        prices LONGTEXT NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uq_asset_tf (asset, time_frame)
    );

Data format matches sn50 MongoDB:
    {"5m": {"timestamp": "price", ...}}
"""

import os
import json
import pymysql
import re
from dotenv import load_dotenv

load_dotenv()


class MySQLHandler:
    """
    MySQL-based price data handler.
    Separates each asset into its own table dynamically.

    Public interface:
        - save_price_data(asset, time_frame, prices)
        - load_price_data(asset, time_frame) -> dict with "prices" key or {}
    """

    def __init__(self):
        self.config = {
            "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
            "port": int(os.getenv("MYSQL_PORT", 3306)),
            "database": os.getenv("MYSQL_DB", "synth_prices"),
            "user": os.getenv("MYSQL_USER", "synth"),
            "password": os.getenv("MYSQL_PASSWORD", "synth_pass"),
            "charset": "utf8mb4",
            "connect_timeout": 10,
        }
        # In case inside container
        if os.getenv("HOSTNAME") and not os.getenv("MYSQL_HOST"):
            self.config["host"] = "synth-mysql"

    def _get_connection(self):
        """Get a new MySQL connection."""
        return pymysql.connect(**self.config)

    def _get_table_name(self, asset: str) -> str:
        """Sanitize asset name to be safe for MySQL table names."""
        safe_asset = re.sub(r'[^a-zA-Z0-9]', '_', asset)
        return f"price_data_{safe_asset}"

    def _ensure_table(self, table_name: str):
        """Create the asset-specific table if it doesn't exist."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            time_frame VARCHAR(20) NOT NULL,
                            timestamp INT NOT NULL,
                            price DOUBLE NOT NULL,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                            UNIQUE KEY uq_tf_ts (time_frame, timestamp)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """)
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            print(f"[WARN] MySQL table creation failed for {table_name}: {e}")

    def save_price_data(self, asset: str, time_frame: str, prices: dict):
        """
        Save price data using INSERT ... ON DUPLICATE KEY UPDATE (upsert).
        Prices should be individual rows per timestamp.

        Args:
            asset: Asset name (e.g., "BTC", "ETH")
            time_frame: Time frame (e.g., "5m", "60")
            prices: Dict of prices {time_frame: {timestamp: price, ...}}
        """
        table_name = self._get_table_name(asset)
        self._ensure_table(table_name)
        
        insert_data = []
        for tf, price_dict in prices.items():
            for ts, price in price_dict.items():
                try:
                    ts_int = int(ts)
                    price_float = float(price)
                    insert_data.append((tf, ts_int, price_float))
                except ValueError:
                    pass
        
        if not insert_data:
            return

        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.executemany(f"""
                        INSERT INTO {table_name} (time_frame, timestamp, price, updated_at)
                        VALUES (%s, %s, %s, NOW())
                        ON DUPLICATE KEY UPDATE
                            price = VALUES(price),
                            updated_at = NOW()
                    """, insert_data)
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            print(f"[ERROR] Failed to save price data to MySQL for {table_name}: {e}")

    def load_price_data(self, asset: str, time_frame: str) -> dict:
        """
        Load price data from MySQL. Groups rows by timestamp.

        Returns:
            dict: {"prices": {time_frame: {str(timestamp): price}}} or {} if not found.
        """
        table_name = self._get_table_name(asset)
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        f"SELECT timestamp, price FROM {table_name} WHERE time_frame = %s ORDER BY timestamp ASC",
                        (time_frame,)
                    )
                    rows = cursor.fetchall()
                    if rows:
                        prices = {}
                        for ts, price in rows:
                            prices[str(ts)] = price
                        return {"prices": {time_frame: prices}}
                    return {}
            finally:
                conn.close()
        except pymysql.err.ProgrammingError as e:
            if e.args[0] == 1146:
                return {}
            print(f"[ERROR] Failed to load price data from MySQL {table_name}: {e}")
            return {}
        except Exception as e:
            print(f"[ERROR] Failed to load price data from MySQL {table_name}: {e}")
            return {}



"""
mysql_handler.py

Storage handler for price data, validation scores, and leaderboard data
using MySQL (MariaDB).

Connection config from .env:
    MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD
"""

import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional
try:
    import pymysql
    import pymysql.cursors
except ModuleNotFoundError:  # optional dependency for DB features
    pymysql = None  # type: ignore

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # optional dependency
    load_dotenv = None  # type: ignore

if load_dotenv is not None:
    load_dotenv()
logger = logging.getLogger(__name__)


def _clean_iso(s: str) -> str:
    """Normalize ISO 8601 string into MySQL DATETIME format."""
    if not s:
        return s
    return s.replace("T", " ").replace("Z", "").split("+")[0]


class MySQLHandler:
    """
    MySQL-based storage for price data, validation scores, and leaderboard.

    Tables (per asset):
        price_data_{ASSET}          — 1m/5m candle prices
        validation_scores_{ASSET}   — CRPS scores from validators

    Tables (global):
        leaderboard_v1              — metagraph: incentive, emission, stake, rank
        leaderboard_v2              — rewards per neuron
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
        if os.getenv("HOSTNAME") and not os.getenv("MYSQL_HOST"):
            self.config["host"] = "synth-mysql"

    def _get_connection(self):
        if pymysql is None:
            raise ModuleNotFoundError(
                "pymysql is required for MySQLHandler. Install with: pip install pymysql"
            )
        return pymysql.connect(**self.config)

    @staticmethod
    def _safe_table(asset: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]", "_", asset)

    # ══════════════════════════════════════════════════════════════════
    # Price Data
    # ══════════════════════════════════════════════════════════════════

    def _get_table_name(self, asset: str) -> str:
        return f"price_data_{self._safe_table(asset)}"

    def _ensure_table(self, table_name: str):
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"""
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
            logger.warning("Table creation failed for %s: %s", table_name, e)

    def save_price_data(self, asset: str, time_frame: str, prices: dict):
        table_name = self._get_table_name(asset)
        self._ensure_table(table_name)

        insert_data = []
        for tf, price_dict in prices.items():
            for ts, price in price_dict.items():
                try:
                    insert_data.append((tf, int(ts), float(price)))
                except ValueError:
                    pass

        if not insert_data:
            return

        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.executemany(f"""
                        INSERT INTO {table_name} (time_frame, timestamp, price, updated_at)
                        VALUES (%s, %s, %s, NOW())
                        ON DUPLICATE KEY UPDATE price = VALUES(price), updated_at = NOW()
                    """, insert_data)
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.error("Failed to save price data for %s: %s", table_name, e)

    def load_price_data(self, asset: str, time_frame: str) -> dict:
        table_name = self._get_table_name(asset)
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT timestamp, price FROM {table_name} "
                        f"WHERE time_frame = %s ORDER BY timestamp ASC",
                        (time_frame,),
                    )
                    rows = cur.fetchall()
                    if rows:
                        return {"prices": {time_frame: {str(ts): p for ts, p in rows}}}
                    return {}
            finally:
                conn.close()
        except pymysql.err.ProgrammingError as e:
            if e.args[0] == 1146:
                return {}
            logger.error("Load price data %s: %s", table_name, e)
            return {}
        except Exception as e:
            logger.error("Load price data %s: %s", table_name, e)
            return {}

    def get_latest_price_timestamp(self, asset: str, time_frame: str) -> Optional[int]:
        """Return the most recent timestamp in price_data for (asset, time_frame)."""
        table_name = self._get_table_name(asset)
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT MAX(timestamp) FROM {table_name} WHERE time_frame = %s",
                        (time_frame,),
                    )
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        return int(row[0])
                    return None
            finally:
                conn.close()
        except Exception as e:
            if pymysql and isinstance(e, pymysql.err.ProgrammingError) and getattr(e, "args", (None,))[0] == 1146:
                return None
            logger.error("get_latest_price_timestamp %s: %s", table_name, e)
            return None

    # ══════════════════════════════════════════════════════════════════
    # Validation Scores (per-asset tables)
    # ══════════════════════════════════════════════════════════════════

    def _val_table(self, asset: str) -> str:
        return f"validation_scores_{self._safe_table(asset)}"

    def _ensure_validation_table(self, table_name: str):
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            scored_time DATETIME NOT NULL,
                            time_length INT NOT NULL,
                            time_increment INT NOT NULL,
                            miner_uid INT NOT NULL,
                            crps DOUBLE NOT NULL,
                            prompt_score DOUBLE NOT NULL,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                            UNIQUE KEY uq_val (scored_time, time_length, time_increment, miner_uid),
                            KEY idx_scored_time (scored_time),
                            KEY idx_miner_uid (miner_uid)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """)
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Validation table creation failed for %s: %s", table_name, e)

    def save_validation_scores(self, asset: str, scores: list):
        """
        Upsert validation scores.

        Each score dict: {scored_time, time_length, time_increment, miner_uid, crps, prompt_score}
        """
        table_name = self._val_table(asset)
        self._ensure_validation_table(table_name)

        insert_data = []
        for s in scores:
            try:
                scored_time = _clean_iso(s.get("scored_time", ""))
                if not scored_time:
                    continue
                insert_data.append((
                    scored_time,
                    int(s.get("time_length", 86400)),
                    int(s.get("time_increment", 300)),
                    int(s.get("miner_uid", -1)),
                    float(s.get("crps", 0.0)),
                    float(s.get("prompt_score", 0.0)),
                ))
            except (ValueError, TypeError):
                pass

        if not insert_data:
            return

        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.executemany(f"""
                        INSERT INTO {table_name}
                        (scored_time, time_length, time_increment, miner_uid, crps, prompt_score, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW())
                        ON DUPLICATE KEY UPDATE
                            crps = VALUES(crps),
                            prompt_score = VALUES(prompt_score),
                            updated_at = NOW()
                    """, insert_data)
                conn.commit()
                logger.info("Saved %d validation scores → %s", len(insert_data), table_name)
            finally:
                conn.close()
        except Exception as e:
            logger.error("Failed to save validation scores to %s: %s", table_name, e)

    def get_validation_scores(
        self,
        asset: str,
        start_time: str,
        end_time: str,
        time_length: int = 86400,
        miner_uid: Optional[int] = None,
    ) -> list:
        """Load validation scores within a time range."""
        table_name = self._val_table(asset)
        st = _clean_iso(start_time)
        et = _clean_iso(end_time)

        try:
            conn = self._get_connection()
            try:
                with conn.cursor(pymysql.cursors.DictCursor) as cur:
                    sql = f"""
                        SELECT scored_time, time_length, time_increment,
                               miner_uid, crps, prompt_score
                        FROM {table_name}
                        WHERE time_length = %s
                          AND scored_time >= %s AND scored_time <= %s
                    """
                    params: list = [time_length, st, et]

                    if miner_uid is not None:
                        sql += " AND miner_uid = %s"
                        params.append(miner_uid)

                    sql += " ORDER BY scored_time ASC"
                    cur.execute(sql, params)
                    rows = cur.fetchall()

                    for r in rows:
                        if hasattr(r["scored_time"], "isoformat"):
                            r["scored_time"] = r["scored_time"].isoformat()
                    return rows
            finally:
                conn.close()
        except pymysql.err.ProgrammingError as e:
            if e.args[0] == 1146:
                return []
            logger.error("Load validation scores %s: %s", table_name, e)
            return []
        except Exception as e:
            logger.error("Load validation scores %s: %s", table_name, e)
            return []

    def get_validation_scores_for_slot(
        self,
        asset: str,
        scored_time: str,
        time_length: int,
        time_increment: int,
        min_crps: float = 0.0,
    ) -> list:
        """
        Load validation scores for an exact scored_time slot.

        Returns list of dicts:
            {scored_time, time_length, time_increment, miner_uid, crps, prompt_score}
        """
        table_name = self._val_table(asset)
        st = _clean_iso(scored_time)

        try:
            conn = self._get_connection()
            try:
                with conn.cursor(pymysql.cursors.DictCursor) as cur:
                    sql = f"""
                        SELECT scored_time, time_length, time_increment,
                               miner_uid, crps, prompt_score
                        FROM {table_name}
                        WHERE scored_time = %s
                          AND time_length = %s
                          AND time_increment = %s
                          AND crps > %s
                        ORDER BY crps ASC
                    """
                    cur.execute(sql, (st, int(time_length), int(time_increment), float(min_crps)))
                    rows = cur.fetchall()
                    for r in rows:
                        if hasattr(r["scored_time"], "isoformat"):
                            r["scored_time"] = r["scored_time"].isoformat()
                    return rows
            finally:
                conn.close()
        except pymysql.err.ProgrammingError as e:
            if e.args[0] == 1146:
                return []
            logger.error("Load validation slot scores %s: %s", table_name, e)
            return []
        except Exception as e:
            logger.error("Load validation slot scores %s: %s", table_name, e)
            return []

    def get_latest_scored_time(self, asset: str, time_length: int = 86400) -> Optional[str]:
        """Return the most recent scored_time in the validation table (for backfill gap detection)."""
        table_name = self._val_table(asset)
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT MAX(scored_time) FROM {table_name} WHERE time_length = %s",
                        (time_length,),
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        dt = row[0]
                        return dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
                    return None
            finally:
                conn.close()
        except Exception:
            return None

    def get_top_miners(
        self,
        asset: str,
        days_back: int = 14,
        time_length: int = 86400,
        top_n: int = 20,
    ) -> List[Dict]:
        """
        Rank miners by mean CRPS from local DB (lower = better).
        Returns [{miner_uid, mean_crps, count}, ...] sorted ascending.
        """
        table_name = self._val_table(asset)
        try:
            conn = self._get_connection()
            try:
                with conn.cursor(pymysql.cursors.DictCursor) as cur:
                    cur.execute(f"""
                        SELECT miner_uid,
                               AVG(crps)   AS mean_crps,
                               COUNT(*)    AS cnt,
                               STDDEV(crps) AS std_crps
                        FROM {table_name}
                        WHERE time_length = %s
                          AND scored_time >= DATE_SUB(NOW(), INTERVAL %s DAY)
                          AND crps > 0
                        GROUP BY miner_uid
                        HAVING cnt >= 5
                        ORDER BY mean_crps ASC
                        LIMIT %s
                    """, (time_length, days_back, top_n))
                    return cur.fetchall()
            finally:
                conn.close()
        except Exception as e:
            logger.error("get_top_miners %s: %s", table_name, e)
            return []

    # ══════════════════════════════════════════════════════════════════
    # Leaderboard v1 (full metagraph)
    # ══════════════════════════════════════════════════════════════════

    def _ensure_leaderboard_v1_table(self):
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS leaderboard_v1 (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            updated_at DATETIME NOT NULL,
                            neuron_uid INT NOT NULL,
                            incentive DOUBLE NOT NULL DEFAULT 0,
                            emission DOUBLE NOT NULL DEFAULT 0,
                            stake DOUBLE NOT NULL DEFAULT 0,
                            rank_val INT NOT NULL DEFAULT 0,
                            pruning_score DOUBLE NOT NULL DEFAULT 0,
                            coldkey VARCHAR(64) NOT NULL DEFAULT '',
                            ip_address VARCHAR(64) NOT NULL DEFAULT '',
                            inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE KEY uq_lb1 (updated_at, neuron_uid),
                            KEY idx_neuron (neuron_uid),
                            KEY idx_updated (updated_at)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """)
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("leaderboard_v1 table creation: %s", e)

    def save_leaderboard_v1(self, records: list):
        """Upsert v1 leaderboard records."""
        self._ensure_leaderboard_v1_table()

        insert_data = []
        for r in records:
            try:
                insert_data.append((
                    _clean_iso(r.get("updated_at", "")),
                    int(r.get("neuron_uid", -1)),
                    float(r.get("incentive", 0)),
                    float(r.get("emission", 0)),
                    float(r.get("stake", 0)),
                    int(r.get("rank", 0)),
                    float(r.get("pruning_score", 0)),
                    str(r.get("coldkey", "")),
                    str(r.get("ip_address", "")),
                ))
            except (ValueError, TypeError):
                pass

        if not insert_data:
            return

        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.executemany("""
                        INSERT INTO leaderboard_v1
                        (updated_at, neuron_uid, incentive, emission, stake,
                         rank_val, pruning_score, coldkey, ip_address)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            incentive = VALUES(incentive),
                            emission = VALUES(emission),
                            stake = VALUES(stake),
                            rank_val = VALUES(rank_val),
                            pruning_score = VALUES(pruning_score),
                            coldkey = VALUES(coldkey),
                            ip_address = VALUES(ip_address)
                    """, insert_data)
                conn.commit()
                logger.info("Saved %d leaderboard_v1 records", len(insert_data))
            finally:
                conn.close()
        except Exception as e:
            logger.error("Failed to save leaderboard_v1: %s", e)

    def get_leaderboard_v1(
        self,
        start_time: str,
        end_time: str,
        neuron_uid: Optional[int] = None,
    ) -> list:
        self._ensure_leaderboard_v1_table()
        st = _clean_iso(start_time)
        et = _clean_iso(end_time)
        try:
            conn = self._get_connection()
            try:
                with conn.cursor(pymysql.cursors.DictCursor) as cur:
                    sql = """
                        SELECT updated_at, neuron_uid, incentive, emission,
                               stake, rank_val, pruning_score, coldkey, ip_address
                        FROM leaderboard_v1
                        WHERE updated_at >= %s AND updated_at <= %s
                    """
                    params: list = [st, et]
                    if neuron_uid is not None:
                        sql += " AND neuron_uid = %s"
                        params.append(neuron_uid)
                    sql += " ORDER BY updated_at ASC"
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    for r in rows:
                        if hasattr(r["updated_at"], "isoformat"):
                            r["updated_at"] = r["updated_at"].isoformat()
                    return rows
            finally:
                conn.close()
        except Exception as e:
            logger.error("get_leaderboard_v1: %s", e)
            return []

    # ══════════════════════════════════════════════════════════════════
    # Leaderboard v2 (rewards)
    # ══════════════════════════════════════════════════════════════════

    def _ensure_leaderboard_v2_table(self):
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS leaderboard_v2 (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            updated_at DATETIME NOT NULL,
                            neuron_uid INT NOT NULL,
                            rewards DOUBLE NOT NULL DEFAULT 0,
                            coldkey VARCHAR(64) NOT NULL DEFAULT '',
                            ip_address VARCHAR(64) NOT NULL DEFAULT '',
                            inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE KEY uq_lb2 (updated_at, neuron_uid),
                            KEY idx_neuron (neuron_uid),
                            KEY idx_updated (updated_at)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """)
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning("leaderboard_v2 table creation: %s", e)

    def save_leaderboard_v2(self, records: list):
        """Upsert v2 leaderboard records (rewards)."""
        self._ensure_leaderboard_v2_table()

        insert_data = []
        for r in records:
            try:
                insert_data.append((
                    _clean_iso(r.get("updated_at", "")),
                    int(r.get("neuron_uid", -1)),
                    float(r.get("rewards", 0)),
                    str(r.get("coldkey", "")),
                    str(r.get("ip_address", "")),
                ))
            except (ValueError, TypeError):
                pass

        if not insert_data:
            return

        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.executemany("""
                        INSERT INTO leaderboard_v2
                        (updated_at, neuron_uid, rewards, coldkey, ip_address)
                        VALUES (%s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            rewards = VALUES(rewards),
                            coldkey = VALUES(coldkey),
                            ip_address = VALUES(ip_address)
                    """, insert_data)
                conn.commit()
                logger.info("Saved %d leaderboard_v2 records", len(insert_data))
            finally:
                conn.close()
        except Exception as e:
            logger.error("Failed to save leaderboard_v2: %s", e)

    def get_leaderboard_v2(
        self,
        start_time: str,
        end_time: str,
        neuron_uid: Optional[int] = None,
    ) -> list:
        self._ensure_leaderboard_v2_table()
        st = _clean_iso(start_time)
        et = _clean_iso(end_time)
        try:
            conn = self._get_connection()
            try:
                with conn.cursor(pymysql.cursors.DictCursor) as cur:
                    sql = """
                        SELECT updated_at, neuron_uid, rewards, coldkey, ip_address
                        FROM leaderboard_v2
                        WHERE updated_at >= %s AND updated_at <= %s
                    """
                    params: list = [st, et]
                    if neuron_uid is not None:
                        sql += " AND neuron_uid = %s"
                        params.append(neuron_uid)
                    sql += " ORDER BY updated_at ASC"
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    for r in rows:
                        if hasattr(r["updated_at"], "isoformat"):
                            r["updated_at"] = r["updated_at"].isoformat()
                    return rows
            finally:
                conn.close()
        except Exception as e:
            logger.error("get_leaderboard_v2: %s", e)
            return []

    def get_latest_leaderboard_time(self, version: int = 2) -> Optional[str]:
        """Return the most recent updated_at in the leaderboard table (for backfill gap detection)."""
        table = "leaderboard_v2" if version == 2 else "leaderboard_v1"
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT MAX(updated_at) FROM {table}")
                    row = cur.fetchone()
                    if row and row[0]:
                        dt = row[0]
                        return dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
                    return None
            finally:
                conn.close()
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════
    # Ensure all tables exist (leaderboard + validation_scores)
    # ══════════════════════════════════════════════════════════════════

    DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "XAU", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]

    def ensure_all_tables(self, assets: Optional[List[str]] = None) -> List[str]:
        """
        Tạo tất cả bảng leaderboard và validation_scores (history).
        Returns: danh sách tên bảng đã tạo/đảm bảo tồn tại.
        """
        created = []
        assets = assets or self.DEFAULT_ASSETS

        self._ensure_leaderboard_v1_table()
        created.append("leaderboard_v1")
        self._ensure_leaderboard_v2_table()
        created.append("leaderboard_v2")

        for asset in assets:
            table_name = self._val_table(asset)
            self._ensure_validation_table(table_name)
            created.append(table_name)

        return created

    # ══════════════════════════════════════════════════════════════════
    # Utility: DB summary
    # ══════════════════════════════════════════════════════════════════

    def get_db_summary(self) -> Dict:
        """Return a summary of all tables and row counts for diagnostics."""
        summary: Dict = {}
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("SHOW TABLES")
                    tables = [r[0] for r in cur.fetchall()]
                    for t in tables:
                        cur.execute(f"SELECT COUNT(*) FROM `{t}`")
                        summary[t] = cur.fetchone()[0]
            finally:
                conn.close()
        except Exception as e:
            logger.error("get_db_summary: %s", e)
        return summary

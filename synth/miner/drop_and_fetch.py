"""
drop_and_fetch.py

Drop tất cả bảng trong DB (validation_scores, leaderboard, price_data) rồi fetch lại.
Chạy: uv run python synth/miner/drop_and_fetch.py
      uv run python synth/miner/drop_and_fetch.py --skip-prices   # Chỉ scores + leaderboard
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from synth.miner.mysql_handler import MySQLHandler
from synth.miner.fetch_daemon import (
    backfill_scores,
    backfill_leaderboard,
    fetch_price_cycle,
)


def drop_all_tables(mysql: MySQLHandler) -> list[str]:
    """Drop tất cả bảng trong database. Trả về danh sách bảng đã drop."""
    dropped = []
    try:
        conn = mysql._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SHOW TABLES")
                tables = [r[0] for r in cur.fetchall()]
                if not tables:
                    print("No tables to drop.")
                    return []

                cur.execute("SET FOREIGN_KEY_CHECKS = 0")
                for t in tables:
                    cur.execute(f"DROP TABLE IF EXISTS `{t}`")
                    dropped.append(t)
                    print(f"  Dropped: {t}")
                cur.execute("SET FOREIGN_KEY_CHECKS = 1")
                conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"Error dropping tables: {e}")
        raise
    return dropped


def main():
    parser = argparse.ArgumentParser(description="Drop DB tables and refetch")
    parser.add_argument("--skip-prices", action="store_true",
                        help="Skip price fetch (chỉ scores + leaderboard)")
    args = parser.parse_args()

    print("=" * 60)
    print("DROP DB & FETCH LẠI")
    print("=" * 60)

    mysql = MySQLHandler()
    print("\n1. Dropping all tables...")
    dropped = drop_all_tables(mysql)
    print(f"   Dropped {len(dropped)} tables.\n")

    if not args.skip_prices:
        print("2. Fetching price data...")
        fetch_price_cycle()
        print()
    else:
        print("2. Skipping price data (--skip-prices)\n")

    print("3. Backfilling validation scores (90 days)...")
    backfill_scores(days=90)
    print()

    print("4. Backfilling leaderboard (30 days)...")
    backfill_leaderboard(days=30)
    print()

    print("=" * 60)
    print("DONE. DB đã được reset và fetch lại.")
    print("Khởi động lại fetch daemon: pm2 start fetch-daemon")
    print("=" * 60)


if __name__ == "__main__":
    main()

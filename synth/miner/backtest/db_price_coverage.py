"""
Report min/max timestamp span per asset and timeframe in MySQL price tables.

Uses the same env as ``MySQLHandler`` (MYSQL_HOST, MYSQL_PORT, MYSQL_DB, ...).
Tables: ``price_data_{ASSET}`` with columns ``time_frame``, ``timestamp`` (unix).

Example:
    uv run python -m synth.miner.backtest.db_price_coverage
    uv run python -m synth.miner.backtest.db_price_coverage --json-out result/db_price_coverage.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone

try:
    import pymysql
except ModuleNotFoundError:
    pymysql = None  # type: ignore

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None  # type: ignore

if load_dotenv is not None:
    load_dotenv()


def _connect():
    if pymysql is None:
        raise SystemExit("pymysql required: pip install pymysql")
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        database=os.getenv("MYSQL_DB", "synth_prices"),
        user=os.getenv("MYSQL_USER", "synth"),
        password=os.getenv("MYSQL_PASSWORD", "synth_pass"),
        charset="utf8mb4",
        connect_timeout=10,
    )


def _asset_from_table(name: str) -> str | None:
    m = re.match(r"^price_data_(.+)$", name, re.I)
    return m.group(1).replace("_", "") if m else None  # tables use safe names; best-effort


def main() -> None:
    p = argparse.ArgumentParser(description="DB price_data coverage per asset/time_frame")
    p.add_argument(
        "--json-out",
        default=None,
        help="Write full report as JSON to this path",
    )
    args = p.parse_args()

    conn = _connect()
    db = os.getenv("MYSQL_DB", "synth_prices")
    rows_out: list[dict] = []
    tables: list[str] = []

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT TABLE_NAME FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME LIKE 'price_data_%%'
                ORDER BY TABLE_NAME
                """,
                (db,),
            )
            tables = [r[0] for r in cur.fetchall()]

        for table in tables:
            asset_guess = _asset_from_table(table)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT time_frame, MIN(timestamp), MAX(timestamp), COUNT(*) "
                        f"FROM `{table}` GROUP BY time_frame ORDER BY time_frame"
                    )
                    for tf, t_min, t_max, cnt in cur.fetchall():
                        if t_min is None or t_max is None:
                            continue
                        span_s = int(t_max) - int(t_min)
                        span_days = span_s / 86400.0
                        rows_out.append(
                            {
                                "table": table,
                                "asset_guess": asset_guess,
                                "time_frame": str(tf),
                                "min_ts": int(t_min),
                                "max_ts": int(t_max),
                                "min_utc": datetime.fromtimestamp(int(t_min), tz=timezone.utc).isoformat(),
                                "max_utc": datetime.fromtimestamp(int(t_max), tz=timezone.utc).isoformat(),
                                "n_rows": int(cnt),
                                "span_seconds": span_s,
                                "span_days": round(span_days, 4),
                            }
                        )
            except Exception as e:
                rows_out.append({"table": table, "error": str(e)[:200]})
    finally:
        conn.close()

    payload = {
        "database": db,
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "rows": rows_out,
    }

    if args.json_out:
        d = os.path.dirname(os.path.abspath(args.json_out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.json_out}")

    # Human-readable table
    print(f"database={db}  price_data_tables={len(tables)}")
    for r in rows_out:
        if "error" in r:
            print(f"  {r['table']}: ERROR {r['error']}")
            continue
        print(
            f"  {r['table']} | {r['time_frame']:>3} | rows={r['n_rows']:,} | "
            f"span={r['span_days']} d | {r['min_utc'][:10]} .. {r['max_utc'][:10]}"
        )


if __name__ == "__main__":
    main()

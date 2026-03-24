"""
fetch_daemon.py

Standalone data fetch daemon for pm2.
Fetches: price data, validation scores (history + latest), leaderboard (history + latest).
- Price 1m: mỗi 60s, incremental từ last → now
- Validation scores latest: mỗi 30 phút
- Validation scores history: mỗi 4 giờ, incremental
- Leaderboard latest (v1+v2): mỗi 30 phút
- Leaderboard history (v1+v2): mỗi 1 giờ, incremental

Usage:
    # Via pm2 (recommended):
    pm2 start fetch.config.js

    # Direct:
    PYTHONPATH=. python synth/miner/fetch_daemon.py

    # Single iteration:
    PYTHONPATH=. python synth/miner/fetch_daemon.py --once

    # Backfill validation scores for 90 days:
    PYTHONPATH=. python synth/miner/fetch_daemon.py --backfill-scores --days 90

    # Backfill leaderboard for 30 days:
    PYTHONPATH=. python synth/miner/fetch_daemon.py --backfill-leaderboard --days 30
"""

import sys
import os
import time
import datetime
import traceback
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from synth.miner.my_simulation import fetch_price_data
from synth.miner.my_simulation import data_handler
from synth.miner.constants import ASSETS_PERIODIC_FETCH_PRICE_DATA
from synth.miner.mysql_handler import MySQLHandler
from synth.miner.synthdata_client import SynthDataClient, VALID_ASSETS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
CRYPTO_ASSETS = ["BTC", "ETH", "SOL"]
PERIODIC_ASSETS = ASSETS_PERIODIC_FETCH_PRICE_DATA
ALL_ASSETS = list(dict.fromkeys(CRYPTO_ASSETS + PERIODIC_ASSETS))

TIME_INCREMENTS = [60]
FETCH_INTERVAL = 60  # 1 minute — fetch từ last_time đến hiện tại mỗi phút

# How often to run each fetch type (seconds)
VALIDATION_SCORES_INTERVAL = 14400    # 4 hours — validation scores history
LEADERBOARD_INTERVAL = 3600           # 1 hour — leaderboard history (v1 + v2)
LATEST_SCORES_INTERVAL = 1800         # 30 minutes — validation scores latest
LEADERBOARD_LATEST_INTERVAL = 1800    # 30 minutes — leaderboard latest snapshot

# Default lookback for incremental fetches (when no prior data in DB)
SCORES_LOOKBACK_DAYS = 3
LEADERBOARD_LOOKBACK_DAYS = 7  # More days for backtest when DB empty

# Số luồng song song (tránh rate limit API, mặc định 4)
MAX_WORKERS = 4


def log(msg: str, level: str = "INFO"):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


class FetchState:
    """Track when each fetch type last ran."""
    def __init__(self):
        self.last_validation_scores = 0.0
        self.last_leaderboard = 0.0
        self.last_leaderboard_latest = 0.0
        self.last_latest_scores = 0.0

    def should_run(self, key: str, interval: float) -> bool:
        last = getattr(self, f"last_{key}", 0.0)
        return (time.time() - last) >= interval or last == 0.0

    def mark_done(self, key: str):
        setattr(self, f"last_{key}", time.time())


state = FetchState()


# ── Price data ──────────────────────────────────────────────────────

def _fetch_price_one(asset: str, time_increment: int) -> tuple[str, bool, str]:
    """
    Fetch 1 asset — incremental: từ last_time trong DB đến hiện tại.
    - Nếu DB trống: backfill đầy đủ qua fetch_price_data.
    - Nếu có data và stale (>= time_increment): fetch từ last_timestamp đến now.
    """
    try:
        time_frame = "1m" if time_increment == 60 else ("5m" if time_increment == 300 else str(time_increment))
        db = MySQLHandler()
        latest_ts = db.get_latest_price_timestamp(asset, time_frame)

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        if latest_ts is None:
            # DB trống — backfill đầy đủ
            fetch_price_data(asset, time_increment, only_load=False)
            return (asset, True, "OK (backfill)")
        else:
            staleness = now_utc.timestamp() - latest_ts
            if staleness >= time_increment:
                last_dt = datetime.datetime.fromtimestamp(latest_ts, datetime.timezone.utc)
                data_handler.fetch_multi_timeframes_price_data(
                    asset, last_dt, weeks=1, time_frame=time_frame
                )
                return (asset, True, "OK (incremental)")
            return (asset, True, "OK (up-to-date)")
    except Exception as e:
        return (asset, False, str(e))


def fetch_price_cycle(max_workers: int = MAX_WORKERS):
    """Fetch 1m candle prices for all assets (đa luồng)."""
    cycle_start = time.time()
    tasks = [(a, ti) for a in ALL_ASSETS for ti in TIME_INCREMENTS]
    results = {"success": 0, "error": 0, "total": len(tasks)}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_price_one, a, ti): a for a, ti in tasks}
        for fut in as_completed(futures):
            asset = futures[fut]
            try:
                a, ok, msg = fut.result()
                if ok:
                    results["success"] += 1
                    log(f"  {a} @ 60s — OK")
                else:
                    results["error"] += 1
                    log(f"  {a} @ 60s — {msg}", "ERROR")
            except Exception as e:
                results["error"] += 1
                log(f"  {asset} — {e}", "ERROR")

    elapsed = time.time() - cycle_start
    log(f"Price cycle: {results['success']}/{results['total']} OK, "
        f"{results['error']} errors, {elapsed:.1f}s")
    return results


# ── Validation scores (CRPS) ───────────────────────────────────────

def _fetch_scores_one(
    asset: str, tl: int, ti: int,
    end_dt: datetime.datetime, force: bool, lookback_days: int,
) -> tuple[str, int, bool, str]:
    """Fetch scores cho 1 (asset, tl). Returns (asset, tl, ok, msg)."""
    from synth.miner.synthdata_client import _parse_date
    try:
        db = MySQLHandler()
        client = SynthDataClient()
        latest = db.get_latest_scored_time(asset, tl)
        if latest and not force:
            last_dt = _parse_date(latest)
            start_dt = last_dt - datetime.timedelta(hours=1)
        else:
            start_dt = end_dt - datetime.timedelta(days=lookback_days)

        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        scores = client.get_historical_scores(
            asset=asset, start_date=start_str, end_date=end_str,
            time_length=tl, time_increment=ti,
        )
        if scores:
            db.save_validation_scores(asset, scores)
            return (asset, tl, True, f"{len(scores)} scores saved")
        return (asset, tl, True, "no scores returned")
    except Exception as e:
        return (asset, tl, False, str(e))


def fetch_validation_scores_cycle(
    force: bool = False,
    lookback_days: int = SCORES_LOOKBACK_DAYS,
    max_workers: int = MAX_WORKERS,
):
    """
    Fetch validation scores (CRPS) for backtest (đa luồng).
    - Periodic: runs every VALIDATION_SCORES_INTERVAL (12h).
    - Incremental: when DB has data, fetches from MAX(scored_time) to now.
    - Initial: when DB empty, fetches lookback_days.
    """
    if not force and not state.should_run("validation_scores", VALIDATION_SCORES_INTERVAL):
        return

    log(f"Fetching validation scores (lookback={lookback_days}d, workers={max_workers})...")
    end_dt = datetime.datetime.now(datetime.timezone.utc)
    tasks = [(a, tl, ti) for a in ALL_ASSETS for tl, ti in [(86400, 300), (3600, 60)]]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_fetch_scores_one, a, tl, ti, end_dt, force, lookback_days): (a, tl)
            for a, tl, ti in tasks
        }
        for fut in as_completed(futures):
            a, tl = futures[fut]
            try:
                _, _, ok, msg = fut.result()
                if ok:
                    log(f"  {a} tl={tl} — {msg}")
                else:
                    log(f"  {a} tl={tl} — ERROR: {msg}", "ERROR")
            except Exception as e:
                log(f"  {a} tl={tl} — ERROR: {e}", "ERROR")
                traceback.print_exc()

    state.mark_done("validation_scores")
    log("Validation scores cycle complete.")


# ── Latest scores (incremental update) ─────────────────────────────

def _fetch_latest_scores_one(asset: str, tl: int, ti: int) -> tuple[str, int, bool, str]:
    """Fetch latest scores cho 1 (asset, tl). Returns (asset, tl, ok, msg)."""
    try:
        client = SynthDataClient()
        db = MySQLHandler()
        scores = client.get_latest_scores(asset=asset, time_length=tl, time_increment=ti)
        if scores:
            db.save_validation_scores(asset, scores)
            return (asset, tl, True, f"{len(scores)} scores")
        return (asset, tl, True, "no scores")
    except Exception as e:
        return (asset, tl, False, str(e))


def fetch_latest_scores_cycle(force: bool = False, max_workers: int = MAX_WORKERS):
    """Fetch latest validation scores snapshot every 30 minutes (đa luồng)."""
    if not force and not state.should_run("latest_scores", LATEST_SCORES_INTERVAL):
        return

    log("Fetching latest validation scores...")
    tasks = [(a, tl, ti) for a in ALL_ASSETS for tl, ti in [(86400, 300), (3600, 60)]]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_latest_scores_one, a, tl, ti): (a, tl) for a, tl, ti in tasks}
        for fut in as_completed(futures):
            a, tl = futures[fut]
            try:
                _, _, ok, msg = fut.result()
                if ok:
                    log(f"  {a} tl={tl} latest — {msg}")
                else:
                    log(f"  {a} tl={tl} latest — ERROR: {msg}", "ERROR")
            except Exception as e:
                log(f"  {a} tl={tl} — ERROR: {e}", "ERROR")

    state.mark_done("latest_scores")
    log("Latest scores cycle complete.")


# ── Leaderboard ────────────────────────────────────────────────────

def _fetch_leaderboard_one(
    version: int,
    end_dt: datetime.datetime,
    force: bool,
    lookback_days: int,
) -> tuple[int, bool, str]:
    """Fetch leaderboard v1 hoặc v2. Returns (version, ok, msg)."""
    from synth.miner.synthdata_client import _parse_date
    try:
        db = MySQLHandler()
        client = SynthDataClient()
        latest = db.get_latest_leaderboard_time(version=version)
        if latest and not force:
            last_dt = _parse_date(latest)
            start_dt = last_dt - datetime.timedelta(hours=1)
        else:
            start_dt = end_dt - datetime.timedelta(days=lookback_days)

        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        if version == 2:
            records = client.get_leaderboard_v2_historical(
                start_date=start_str, end_date=end_str,
            )
            if records:
                db.save_leaderboard_v2(records)
                return (version, True, f"{len(records)} records saved")
        else:
            records = client.get_leaderboard_v1_historical(
                start_date=start_str, end_date=end_str,
            )
            if records:
                db.save_leaderboard_v1(records)
                return (version, True, f"{len(records)} records saved")
        return (version, True, "no records")
    except Exception as e:
        return (version, False, str(e))


def fetch_leaderboard_cycle(
    force: bool = False,
    lookback_days: int = LEADERBOARD_LOOKBACK_DAYS,
    max_workers: int = 2,
):
    """
    Fetch leaderboard data (v1 + v2) for backtest (đa luồng).
    - Periodic: runs every LEADERBOARD_INTERVAL (6h).
    - Incremental: when DB has data, fetches from MAX(updated_at) to now.
    - Initial: when DB empty, fetches lookback_days.
    """
    if not force and not state.should_run("leaderboard", LEADERBOARD_INTERVAL):
        return

    log(f"Fetching leaderboard data (lookback={lookback_days}d)...")
    end_dt = datetime.datetime.now(datetime.timezone.utc)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_fetch_leaderboard_one, v, end_dt, force, lookback_days): v
            for v in [1, 2]
        }
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                _, ok, msg = fut.result()
                if ok:
                    log(f"  v{v} — {msg}")
                else:
                    log(f"  v{v} — ERROR: {msg}", "ERROR")
                    traceback.print_exc()
            except Exception as e:
                log(f"  v{v} — ERROR: {e}", "ERROR")
                traceback.print_exc()

    state.mark_done("leaderboard")
    log("Leaderboard cycle complete.")


# ── Leaderboard latest (snapshot) ───────────────────────────────────

def fetch_leaderboard_latest_cycle(force: bool = False):
    """
    Fetch leaderboard v1 + v2 latest snapshot mỗi LEADERBOARD_LATEST_INTERVAL.
    Đảm bảo luôn có dữ liệu leaderboard mới nhất.
    """
    if not force and not state.should_run("leaderboard_latest", LEADERBOARD_LATEST_INTERVAL):
        return

    log("Fetching leaderboard latest (v1 + v2)...")
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        client = SynthDataClient()
        db = MySQLHandler()

        for version, get_fn, save_fn in [
            (1, client.get_leaderboard_v1_latest, db.save_leaderboard_v1),
            (2, client.get_leaderboard_v2_latest, db.save_leaderboard_v2),
        ]:
            records = get_fn()
            if records:
                for r in records:
                    if not r.get("updated_at"):
                        r["updated_at"] = now_iso
                save_fn(records)
                log(f"  v{version} latest — {len(records)} records saved")
            else:
                log(f"  v{version} latest — no records", "WARNING")
    except Exception as e:
        log(f"Leaderboard latest ERROR: {e}", "ERROR")
        traceback.print_exc()

    state.mark_done("leaderboard_latest")
    log("Leaderboard latest cycle complete.")


# ── Backfill commands ──────────────────────────────────────────────

def backfill_scores(days: int = 90, max_workers: int = MAX_WORKERS):
    """Full backfill of validation scores for N days."""
    log(f"=== BACKFILL: Validation scores for {days} days (workers={max_workers}) ===")
    fetch_validation_scores_cycle(force=True, lookback_days=days, max_workers=max_workers)
    log("=== BACKFILL complete ===")


def backfill_leaderboard(days: int = 30, max_workers: int = 2):
    """Full backfill of leaderboard data for N days."""
    log(f"=== BACKFILL: Leaderboard for {days} days ===")
    fetch_leaderboard_cycle(force=True, lookback_days=days, max_workers=max_workers)
    log("=== BACKFILL complete ===")


# ── Main loop ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Synth data fetch daemon")
    parser.add_argument("--once", action="store_true",
                        help="Run a single cycle and exit")
    parser.add_argument("--interval", type=int, default=FETCH_INTERVAL,
                        help=f"Seconds between cycles (default: {FETCH_INTERVAL})")
    parser.add_argument("--backfill-scores", action="store_true",
                        help="Backfill validation scores and exit")
    parser.add_argument("--backfill-leaderboard", action="store_true",
                        help="Backfill leaderboard data and exit")
    parser.add_argument("--ensure-tables", action="store_true",
                        help="Tạo bảng leaderboard và validation_scores rồi thoát")
    parser.add_argument("--days", type=int, default=90,
                        help="Days to backfill (default: 90)")
    parser.add_argument("--skip-prices", action="store_true",
                        help="Skip price data fetching")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Số luồng song song (default: {MAX_WORKERS})")
    args = parser.parse_args()

    log("=" * 60)
    log("SYNTH FETCH DAEMON")
    log(f"Assets: {ALL_ASSETS}")
    log(f"Time increments: {TIME_INCREMENTS}")
    log(f"Interval: {args.interval}s")
    log(f"Workers: {args.workers}")
    log("=" * 60)

    workers = args.workers

    # ── Ensure tables (run and exit) ──
    if args.ensure_tables:
        db = MySQLHandler()
        tables = db.ensure_all_tables(assets=ALL_ASSETS)
        log(f"Đã tạo/đảm bảo {len(tables)} bảng: {', '.join(tables)}")
        return

    # ── Backfill modes (run and exit) ──
    if args.backfill_scores:
        backfill_scores(args.days, max_workers=workers)
        return

    if args.backfill_leaderboard:
        backfill_leaderboard(args.days, max_workers=min(2, workers))
        return

    # ── Single-shot mode ──
    if args.once:
        if not args.skip_prices:
            fetch_price_cycle(max_workers=workers)
        fetch_validation_scores_cycle(force=True, max_workers=workers)
        fetch_latest_scores_cycle(force=True, max_workers=workers)
        fetch_leaderboard_cycle(force=True, max_workers=min(2, workers))
        fetch_leaderboard_latest_cycle(force=True)
        log("Single-shot complete. Exiting.")
        return

    # ── Continuous mode ──
    cycle_num = 0
    while True:
        cycle_num += 1
        log(f"━━━ Cycle #{cycle_num} ━━━")
        try:
            if not args.skip_prices:
                fetch_price_cycle(max_workers=workers)

            fetch_latest_scores_cycle(max_workers=workers)
            fetch_leaderboard_latest_cycle()
            fetch_validation_scores_cycle(max_workers=workers)
            fetch_leaderboard_cycle(max_workers=min(2, workers))

        except Exception as e:
            log(f"Cycle #{cycle_num} crashed: {e}", "CRITICAL")
            traceback.print_exc()

        log(f"Sleeping {args.interval}s until next cycle...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

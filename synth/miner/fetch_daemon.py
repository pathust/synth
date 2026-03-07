"""
fetch_daemon.py

Standalone data fetch daemon for pm2.
Runs forever, fetching price data for all assets every 5 minutes and saving to MySQL.

Based on sn50's start_periodic_fetch_price_data() pattern but as a standalone script
(no threading — pm2 manages the process lifecycle).

Usage:
    # Via pm2 (recommended):
    pm2 start fetch.config.js

    # Direct:
    cd /Users/taiphan/Documents/synth
    PYTHONPATH=. python synth/miner/fetch_daemon.py

    # Single iteration (for testing):
    PYTHONPATH=. python synth/miner/fetch_daemon.py --once
"""

import sys
import os
import time
import datetime
import traceback
import argparse

# ── Auto-resolve `synth` module path ─────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from synth.miner.my_simulation import fetch_price_data
from synth.miner.constants import ASSETS_PERIODIC_FETCH_PRICE_DATA
from synth.miner.mysql_handler import MySQLHandler
from synth.miner.synthdata_client import SynthDataClient

# ── Config ──────────────────────────────────────────────────────────
# Core crypto assets (always fetched)
CRYPTO_ASSETS = ["BTC", "ETH", "SOL"]

# Additional periodic assets (XAU, stocks — from constants.py)
PERIODIC_ASSETS = ASSETS_PERIODIC_FETCH_PRICE_DATA

# All assets to fetch
ALL_ASSETS = list(dict.fromkeys(CRYPTO_ASSETS + PERIODIC_ASSETS))

# Time increments to fetch (in seconds)
TIME_INCREMENTS = [60]  # 1m candles

# Interval between fetch cycles (seconds)
FETCH_INTERVAL = 300  # 5 minutes


def log(msg: str, level: str = "INFO"):
    """Structured log with timestamp."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

# State for validation scores fetch
last_validation_fetch_time = 0

def fetch_validation_scores_cycle(force=False):
    """Fetch validation scores for the last 3 days every 12 hours."""
    global last_validation_fetch_time
    # 12 hours = 43200 seconds
    if not force and (time.time() - last_validation_fetch_time < 43200) and last_validation_fetch_time != 0:
        return

    log("Fetching validation scores from SynthData API (last 3 days)...")
    client = SynthDataClient()
    db = MySQLHandler()
    
    end_dt = datetime.datetime.now(datetime.timezone.utc)
    start_dt = end_dt - datetime.timedelta(days=3)
    
    start_str = start_dt.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    end_str = end_dt.replace(microsecond=0).isoformat().replace('+00:00', 'Z')

    for asset in ALL_ASSETS:
        try:
            # Low_frequency (24h/5m) and High_frequency (1h/1m) 
            for tl, ti in [(86400, 300), (3600, 60)]:
                # API expects ISO strings
                scores = client.get_historical_validation_scores(
                    asset=asset, start_date=start_str, end_date=end_str,
                    time_length=tl, time_increment=ti
                )
                if scores:
                    db.save_validation_scores(asset, scores)
                    log(f"✅ Config(len={tl}) {asset} — {len(scores)} validation scores saved")
        except Exception as e:
            log(f"❌ Config {asset} — {e}", "ERROR")
            traceback.print_exc()

    last_validation_fetch_time = time.time()
    log("Validation scores fetch cycle complete.")


def fetch_cycle():
    """Run one complete fetch cycle for all assets × time increments."""
    cycle_start = time.time()
    results = {"success": 0, "error": 0, "total": 0}

    for asset in ALL_ASSETS:
        for time_increment in TIME_INCREMENTS:
            results["total"] += 1
            try:
                log(f"Fetching {asset} @ {time_increment}s...")
                fetch_price_data(asset, time_increment, only_load=False)
                results["success"] += 1
                log(f"✅ {asset} @ {time_increment}s — OK")
            except Exception as e:
                results["error"] += 1
                log(f"❌ {asset} @ {time_increment}s — {e}", "ERROR")
                traceback.print_exc()
                # Wait briefly before next asset to avoid cascading failures
                time.sleep(5)

    elapsed = time.time() - cycle_start
    log(
        f"Cycle complete: {results['success']}/{results['total']} OK, "
        f"{results['error']} errors, {elapsed:.1f}s elapsed"
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Price data fetch daemon")
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single fetch cycle and exit (for testing)"
    )
    parser.add_argument(
        "--interval", type=int, default=FETCH_INTERVAL,
        help=f"Seconds between fetch cycles (default: {FETCH_INTERVAL})"
    )
    args = parser.parse_args()

    log("=" * 60)
    log("SYNTH PRICE FETCH DAEMON")
    log(f"Assets: {ALL_ASSETS}")
    log(f"Time increments: {TIME_INCREMENTS}")
    log(f"Interval: {args.interval}s")
    log(f"Mode: {'single-shot' if args.once else 'continuous'}")
    log("=" * 60)

    if args.once:
        fetch_cycle()
        fetch_validation_scores_cycle(force=True)
        log("Single-shot complete. Exiting.")
        return

    # Continuous mode — run forever
    cycle_num = 0
    while True:
        cycle_num += 1
        log(f"━━━ Cycle #{cycle_num} ━━━")
        try:
            fetch_cycle()
            # This handles its own 12h interval checks
            fetch_validation_scores_cycle()
        except Exception as e:
            log(f"Cycle #{cycle_num} crashed: {e}", "CRITICAL")
            traceback.print_exc()

        log(f"Sleeping {args.interval}s until next cycle...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

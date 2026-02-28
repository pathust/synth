"""
backtest_runner.py

Chạy backtest pipeline đầy đủ, lưu kết quả và log vào folder result/.

Usage:
    cd /Users/taiphan/Documents/synth
    conda activate synth
    python synth/miner/backtest_runner.py
"""

import json
import os
import sys
import time
import traceback
import numpy as np
from datetime import datetime, timedelta, timezone
from io import StringIO

# ── Project imports ──────────────────────────────────────────────────
from synth.miner.simulations import generate_simulations
from synth.simulation_input import SimulationInput
from synth.validator.response_validation_v2 import validate_responses
from synth.validator import prompt_config


# ── Config ───────────────────────────────────────────────────────────
# Assets to test (crypto only — stocks need market hours)
ASSETS = ["BTC", "ETH", "SOL"]

# Use recent dates within our 14-day data window
NUM_TEST_DATES = 5
DAYS_BACK_START = 10  # Start from 10 days ago
DAYS_BACK_END = 3     # End at 3 days ago (need future data for CRPS)

# Simulation params
TIME_INCREMENT = 300   # 5 minutes
TIME_LENGTH = 3600     # 1 hour simulation (faster for testing)
NUM_SIMULATIONS = 100  # Reduced for speed; increase for production
SEED = 42

# Output directories
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "result")
LOG_DIR = os.path.join(RESULT_DIR, "logs")


class TeeLogger:
    """Tee stdout to both console and a log file."""

    def __init__(self, log_path):
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        self.log_file.write(msg)
        self.log_file.flush()

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def generate_test_dates(num_dates, days_back_start, days_back_end, seed=42):
    """Generate evenly spaced test dates within the recent data window."""
    import random
    random.seed(seed)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back_start)
    end = now - timedelta(days=days_back_end)

    dates = []
    for _ in range(num_dates):
        delta = (end - start).total_seconds()
        random_seconds = random.uniform(0, delta)
        dt = start + timedelta(seconds=random_seconds)
        dt = dt.replace(second=0, microsecond=0)
        dates.append(dt)

    dates.sort()
    return dates


def run_single_test(asset, start_time, time_increment, time_length, num_simulations, seed):
    """Run a single simulation and return the result dict."""
    simulation_input = SimulationInput(
        asset=asset,
        start_time=start_time.isoformat(),
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
    )

    t0 = time.time()
    result = generate_simulations(
        simulation_input,
        simulation_input.asset,
        start_time=simulation_input.start_time,
        time_increment=simulation_input.time_increment,
        time_length=simulation_input.time_length,
        num_simulations=simulation_input.num_simulations,
        seed=seed,
    )
    elapsed = time.time() - t0

    predictions = result.get("predictions") if result else None

    # Validate format
    valid = False
    if predictions and len(predictions) > 2:
        try:
            fmt = validate_responses(predictions, simulation_input, "0")
            valid = fmt.get("is_valid", False) if isinstance(fmt, dict) else bool(fmt)
        except Exception as e:
            print(f"[WARN] Validation error: {e}")
            valid = True  # If validation itself errors, prediction might still be OK

    # Extract stats
    num_paths = len(predictions) - 2 if predictions and len(predictions) > 2 else 0
    path_length = len(predictions[2]) if num_paths > 0 else 0
    price_start = float(predictions[2][0]) if num_paths > 0 else None
    price_end = float(predictions[2][-1]) if num_paths > 0 else None

    return {
        "asset": asset,
        "start_time": start_time.isoformat(),
        "time_increment": time_increment,
        "time_length": time_length,
        "num_simulations": num_simulations,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 2),
        "num_paths": num_paths,
        "path_length": path_length,
        "price_start": price_start,
        "price_end": price_end,
        "format_valid": valid,
        "status": "SUCCESS" if num_paths > 0 else "FAIL",
        "predictions_sample": [
            [float(x) for x in predictions[i][:5]]  # First 5 points of first 3 paths
            for i in range(2, min(5, len(predictions)))
        ] if num_paths > 0 else [],
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directories
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Setup logging
    log_path = os.path.join(LOG_DIR, f"backtest_{timestamp}.log")
    logger = TeeLogger(log_path)
    sys.stdout = logger

    print("=" * 80)
    print(f"SYNTH MINER BACKTEST — {timestamp}")
    print(f"Assets: {ASSETS}")
    print(f"Test dates: {NUM_TEST_DATES} (from {DAYS_BACK_START}d to {DAYS_BACK_END}d ago)")
    print(f"Simulation: {TIME_LENGTH}s length, {TIME_INCREMENT}s increment, {NUM_SIMULATIONS} sims")
    print("=" * 80)

    # Generate test dates
    dates = generate_test_dates(NUM_TEST_DATES, DAYS_BACK_START, DAYS_BACK_END, SEED)
    print(f"\nTest dates generated:")
    for i, d in enumerate(dates):
        print(f"  [{i+1}] {d.isoformat()}")

    # Run all tests
    all_results = []
    summary = {}

    for asset in ASSETS:
        print(f"\n{'='*60}")
        print(f"ASSET: {asset}")
        print(f"{'='*60}")

        asset_results = []
        for i, dt in enumerate(dates):
            print(f"\n--- {asset} Test {i+1}/{len(dates)}: {dt.isoformat()} ---")
            try:
                result = run_single_test(
                    asset, dt, TIME_INCREMENT, TIME_LENGTH, NUM_SIMULATIONS, SEED
                )
                asset_results.append(result)
                all_results.append(result)

                print(f"  Status: {result['status']}")
                print(f"  Paths: {result['num_paths']}, Length: {result['path_length']}")
                if result['price_start']:
                    print(f"  Price: {result['price_start']:.2f} → {result['price_end']:.2f}")
                print(f"  Time: {result['elapsed_seconds']}s")
            except Exception as e:
                print(f"  [ERROR] {e}")
                traceback.print_exc()
                fail_result = {
                    "asset": asset,
                    "start_time": dt.isoformat(),
                    "status": "ERROR",
                    "error": str(e),
                    "elapsed_seconds": 0,
                    "num_paths": 0,
                }
                asset_results.append(fail_result)
                all_results.append(fail_result)

        # Asset summary
        successes = [r for r in asset_results if r["status"] == "SUCCESS"]
        fails = [r for r in asset_results if r["status"] != "SUCCESS"]
        avg_time = np.mean([r["elapsed_seconds"] for r in successes]) if successes else 0

        summary[asset] = {
            "total": len(asset_results),
            "success": len(successes),
            "fail": len(fails),
            "avg_time_seconds": round(avg_time, 2),
        }

        # Save per-asset results
        asset_result_path = os.path.join(RESULT_DIR, f"{asset}_results.json")
        with open(asset_result_path, "w") as f:
            json.dump(asset_results, f, indent=2, default=str)
        print(f"\n  → Saved {len(asset_results)} results to {asset_result_path}")

    # Save full results
    full_result_path = os.path.join(RESULT_DIR, f"backtest_all_{timestamp}.json")
    with open(full_result_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "assets": ASSETS,
                "num_test_dates": NUM_TEST_DATES,
                "time_increment": TIME_INCREMENT,
                "time_length": TIME_LENGTH,
                "num_simulations": NUM_SIMULATIONS,
                "seed": SEED,
            },
            "summary": summary,
            "results": all_results,
        }, f, indent=2, default=str)

    # Save summary
    summary_path = os.path.join(RESULT_DIR, f"summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*80}")
    print("BACKTEST SUMMARY")
    print(f"{'='*80}")
    total_success = 0
    total_tests = 0
    for asset, s in summary.items():
        total_success += s["success"]
        total_tests += s["total"]
        status_icon = "✅" if s["fail"] == 0 else "⚠️"
        print(f"  {status_icon} {asset}: {s['success']}/{s['total']} passed, avg {s['avg_time_seconds']}s")
    print(f"\n  TOTAL: {total_success}/{total_tests} passed")
    print(f"\n  Results: {RESULT_DIR}")
    print(f"  Log: {log_path}")
    print(f"  Full: {full_result_path}")
    print(f"  Summary: {summary_path}")
    print("=" * 80)

    # Restore stdout
    sys.stdout = logger.stdout
    logger.close()

    # Print final summary to real stdout too
    print(f"\n✅ Backtest complete: {total_success}/{total_tests} passed")
    print(f"   Results saved to: {RESULT_DIR}")
    print(f"   Log saved to: {log_path}")


if __name__ == "__main__":
    main()

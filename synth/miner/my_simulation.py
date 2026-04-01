"""
my_simulation.py

Ported from sn50/synth/miner/my_simulation.py (145 lines).

Changes from sn50:
    1. Removed start_periodic_fetch_price_data() (L43-82) — not needed for backtest
    2. Removed __main__ block (L127-145) — periodic fetch startup
    3. Kept: iso_to_timestamp(), fetch_price_data(), simulate_crypto_price_paths()
    
Flow:
    simulate_crypto_price_paths()
      → fetch_price_data(asset, time_increment)
        → data_handler.load_price_data() (MySQL)
        → data_handler.fetch_multi_timeframes_price_data() (if not in MySQL)
      → filter prices < start_time
      → simulate_fn(filter_prices_dict, ...) → np.ndarray
"""

import numpy as np
import datetime
import traceback
import json
import os
from typing import Any, Optional, Callable

from synth.miner.data_handler import DataHandler
from synth.miner.price_aggregation import aggregate_1m_to_5m
from synth.miner.constants import ASSETS_PERIODIC_FETCH_PRICE_DATA

# Global DataHandler instance (shared across simulation calls)
# sn50 L8
data_handler = DataHandler()
OVERFLOW_LOG_DIR = "synth/miner/logs/overflow_requests"
os.makedirs(OVERFLOW_LOG_DIR, exist_ok=True)


def iso_to_timestamp(iso_string: str) -> int:
    """
    Convert ISO format datetime string to Unix timestamp (seconds).
    sn50 L11-22
    
    Args:
        iso_string: ISO format string, e.g., "2025-11-26T16:19:00+00:00"
    
    Returns:
        Unix timestamp as integer (seconds since epoch)
    """
    dt = datetime.datetime.fromisoformat(iso_string)
    return int(dt.timestamp())


def fetch_price_data(asset: str, time_increment: int, only_load: bool = False):
    """
    Load historical price data from MySQL. If not available, fetch from API.
    Khi time_increment=300 (5m) và only_load=True: nếu không có 5m trong DB thì
    load 1m và aggregate sang 5m (giống backtest_data_loader) để tránh fetch API chậm.
    
    Args:
        asset: Asset name (e.g., "BTC", "ETH")
        time_increment: Time increment in seconds (e.g., 300 for 5m)
        only_load: If True, only load from DB without fetching new data
    
    Returns:
        dict: {time_frame: {timestamp: price, ...}}
    """
    time_frame = "1m" if time_increment == 60 else ("5m" if time_increment == 300 else str(time_increment))
    hist_price_data = data_handler.load_price_data(asset, time_frame)

    # Khi cần 5m và only_load: nếu không có 5m thì aggregate từ 1m (tránh fetch API)
    if time_increment == 300 and only_load:
        has_5m = hist_price_data and time_frame in hist_price_data and hist_price_data[time_frame]
        if not has_5m:
            loaded_1m = data_handler.load_price_data(asset, "1m")
            if loaded_1m and "1m" in loaded_1m and loaded_1m["1m"]:
                prices_5m = aggregate_1m_to_5m(loaded_1m["1m"])
                if prices_5m:
                    hist_price_data = {"5m": prices_5m}
                    total = len(prices_5m)
                    print(f"[INFO] Loaded 1m, aggregated to 5m: {asset}, {total} points (only_load=True)")
                    return hist_price_data

    current_time = datetime.datetime.now(datetime.timezone.utc).replace(second=0, microsecond=0)
    target_start = current_time - datetime.timedelta(days=45)

    if not hist_price_data or time_frame not in hist_price_data or not hist_price_data[time_frame]:
        print(f"[WARN] No historical data found for {asset}/{time_frame} in DB, fetching from API backwards...")
        data_handler.fetch_historical_data_backwards(
            asset, current_time, days_back=45, time_frame=time_frame, batch_minutes=60
        )
        hist_price_data = data_handler.load_price_data(asset, time_frame)
    else:
        if only_load:
            total = sum(len(v) if isinstance(v, dict) else 0 for v in hist_price_data.values())
            print(f"[INFO] Loaded {total} data points for {asset}/{time_frame} (only_load=True)")
            return hist_price_data

        # 1. Backfill missing past data if oldest timestamp is too recent
        try:
            first_timestamp = int(min(hist_price_data[time_frame].keys()))
            if first_timestamp > target_start.timestamp() + 86400: # if gap is more than 1 day
                oldest_dt = datetime.datetime.fromtimestamp(first_timestamp, datetime.timezone.utc)
                days_to_backfill = (oldest_dt - target_start).days
                if days_to_backfill > 0:
                    print(f"[WARN] Data for {asset}/{time_frame} only goes back to {oldest_dt.strftime('%Y-%m-%d')}. Backfilling {days_to_backfill} days backwards...")
                    data_handler.fetch_historical_data_backwards(
                        asset, oldest_dt, days_back=days_to_backfill, time_frame=time_frame, batch_minutes=60
                    )
        except Exception as e:
            print(f"[ERROR] Checking oldest timestamp for {asset}: {e}")

        # 2. Forward fill if data is stale
        try:
            last_timestamp = int(max(hist_price_data[time_frame].keys()))
            last_datetime = datetime.datetime.fromtimestamp(
                last_timestamp, datetime.timezone.utc
            )
            current_timestamp = current_time.timestamp()
            staleness = current_timestamp - last_timestamp
            if staleness >= time_increment * 1: # if more than 1 increment missing
                print(f"[INFO] Data stale by {staleness:.0f}s (>{time_increment}s), fetching update forward...")
                data_handler.fetch_multi_timeframes_price_data(
                    asset, last_datetime, weeks=1, time_frame=time_frame
                )
        except Exception as e:
            print(f"[ERROR] Checking latest timestamp for {asset}: {e}")

        # Reload after potential fetches
        hist_price_data = data_handler.load_price_data(asset, time_frame)

    total = sum(len(v) if isinstance(v, dict) else 0 for v in hist_price_data.values()) if hist_price_data else 0
    print(f"[INFO] fetch_price_data complete: {asset}/{time_frame}, {total} total points, only_load={only_load}")
    return hist_price_data


from synth.miner.data.dataloader import UnifiedDataLoader

# Instantiate single dataloader for all simulations
_unified_loader = UnifiedDataLoader()

def simulate_crypto_price_paths(
    current_price: float,
    asset: str,
    start_time: str,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    simulate_fn: Callable,
    max_data_points: Optional[int] = 100000,
    seed: Optional[int] = 42,
    **kwargs
) -> np.ndarray:
    """
    Load historical price data robustly using UnifiedDataLoader, 
    and dispatch to the specified strategy simulation function.
    """
    # Force 30 days of window data by default to give enough context for models like GARCH
    window_days = 30
    
    # Determine frequency based on time_increment to pass to UnifiedDataLoader
    frequency = "high" if time_increment == 60 else ("low" if time_increment == 300 else None)
    
    # 1. Fetch strictly OOS dictionary
    filter_prices_dict = _unified_loader.get_historical_dict(asset, start_time, window_days, frequency=frequency)

    if not filter_prices_dict:
        print(f"[ERROR] No historical prices before start_time={start_time} for {asset} within {window_days} days.")
        return None

    # Limit to max_data_points if requested (ascending order guaranteed by dataloader)
    if max_data_points is not None:
        items = list(filter_prices_dict.items())
        if len(items) > max_data_points:
            filter_prices_dict = dict(items[-max_data_points:])

    print(
        f"[INFO] Simulating {asset} with {simulate_fn.__name__}: "
        f"filtered={len(filter_prices_dict)}, "
        f"n_sims={num_simulations}, seed={seed}"
    )

    def _run_fn(fn: Callable):
        return fn(
            filter_prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=num_simulations,
            seed=seed,
            **kwargs,
        )

    result = _run_fn(simulate_fn)

    # Detect numeric explosion/underflow and log request context for debugging.
    try:
        arr = np.asarray(result, dtype=float)
        if arr.ndim == 2 and arr.size > 0:
            finite_mask = np.isfinite(arr)
            has_nonfinite = not np.all(finite_mask)
            finite_vals = arr[finite_mask]
            max_abs = float(np.max(np.abs(finite_vals))) if finite_vals.size > 0 else float("inf")
            min_val = float(np.min(finite_vals)) if finite_vals.size > 0 else float("nan")
            # Heuristic overflow guard:
            # - inf/nan present
            # - any non-positive price
            # - or absurd magnitude
            exploded = has_nonfinite or min_val <= 0.0 or max_abs > 1e12
            if exploded:
                first_ts = next(iter(filter_prices_dict.keys()), None)
                last_ts = next(reversed(filter_prices_dict.keys()), None) if filter_prices_dict else None
                payload = {
                    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "asset": asset,
                    "start_time": start_time,
                    "time_increment": time_increment,
                    "time_length": time_length,
                    "num_simulations": num_simulations,
                    "seed": seed,
                    "simulate_fn": getattr(simulate_fn, "__name__", str(simulate_fn)),
                    "history_points": len(filter_prices_dict),
                    "history_first_ts": first_ts,
                    "history_last_ts": last_ts,
                    "has_nonfinite": has_nonfinite,
                    "min_finite_price": min_val,
                    "max_abs_finite_price": max_abs,
                    "sample_first_path_head": arr[0, : min(10, arr.shape[1])].tolist(),
                }
                day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
                log_path = os.path.join(OVERFLOW_LOG_DIR, f"overflow_{day}.jsonl")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                print(
                    f"[OVERFLOW] {asset} {simulate_fn.__name__} start={start_time} "
                    f"nonfinite={has_nonfinite} min={min_val:.6g} max_abs={max_abs:.6g} -> {log_path}"
                )
                # Auto-fallback: regenerate using garch_v2_1 for stability
                if getattr(simulate_fn, "__name__", "") != "simulate_single_price_path_with_garch":
                    try:
                        from synth.miner.core.grach_simulator_v2_1 import (
                            simulate_single_price_path_with_garch as _fallback_garch_v2_1,
                        )
                        print(
                            f"[OVERFLOW] Retrying with fallback garch_v2_1 for "
                            f"{asset} start={start_time}"
                        )
                        fb_result = _fallback_garch_v2_1(
                            filter_prices_dict,
                            asset=asset,
                            time_increment=time_increment,
                            time_length=time_length,
                            n_sims=num_simulations,
                            seed=seed,
                        )
                        fb_arr = np.asarray(fb_result, dtype=float)
                        fb_ok = (
                            fb_arr.ndim == 2
                            and fb_arr.size > 0
                            and np.all(np.isfinite(fb_arr))
                            and float(np.min(fb_arr)) > 0.0
                            and float(np.max(np.abs(fb_arr))) <= 1e12
                        )
                        if fb_ok:
                            print("[OVERFLOW] Fallback garch_v2_1 succeeded.")
                            result = fb_result
                        else:
                            print("[OVERFLOW] Fallback garch_v2_1 still unstable; keeping original output.")
                    except Exception as fb_e:
                        print(f"[OVERFLOW] Fallback garch_v2_1 failed: {fb_e}")
    except Exception as e:
        print(f"[OVERFLOW-LOG-WARN] Failed to inspect simulation output: {e}")

    return result

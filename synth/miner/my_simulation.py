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
from typing import Any, Optional, Callable

from synth.miner.data_handler import DataHandler
from synth.miner.constants import ASSETS_PERIODIC_FETCH_PRICE_DATA

# Global DataHandler instance (shared across simulation calls)
# sn50 L8
data_handler = DataHandler()


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
                from synth.miner.backtest_data_loader import aggregate_1m_to_5m
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
    Load historical price data, filter by start_time, and run simulation.
    sn50 L84-125
    
    This is the main dispatcher: it loads the data and delegates to
    the specific simulation function (GARCH, HAR-RV, APARCH, etc.).
    
    Args:
        current_price: Current asset price (may not be used by all simulate_fn)
        asset: Asset name
        start_time: ISO format start time (e.g., "2025-11-26T16:19:00+00:00")
        time_increment: Time increment in seconds
        time_length: Total simulation time in seconds
        num_simulations: Number of simulation paths
        simulate_fn: Callable that performs the actual simulation
        max_data_points: Maximum historical data points to use
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Simulated price paths, shape (n_sims, steps+1)
    """
    time_frame = "1m" if time_increment == 60 else ("5m" if time_increment == 300 else str(time_increment))

    only_load = True  # Force only load for backtesting
    hist_price_data = fetch_price_data(asset, time_increment, only_load=only_load)

    if not hist_price_data or time_frame not in hist_price_data:
        print(f"[ERROR] No historical data available for {asset}/{time_frame}. Cannot simulate.")
        return None

    prices_dict = hist_price_data[time_frame]

    # Filter: only use prices BEFORE start_time
    # This ensures the simulation doesn't "cheat" by using future data
    start_timestamp = iso_to_timestamp(start_time)
    filter_prices_dict = {
        k: v for k, v in prices_dict.items() if int(k) < start_timestamp
    }

    # Sort ascending and limit to max_data_points
    sorted_items = sorted(filter_prices_dict.items(), key=lambda x: int(x[0]))
    if max_data_points is not None:
        filter_prices_dict = dict[Any, Any](sorted_items[-max_data_points:])
    else:
        filter_prices_dict = dict(sorted_items)

    if not filter_prices_dict:
        print(f"[ERROR] No historical prices before start_time={start_time} for {asset}. Total={len(prices_dict)}, filtered=0")
        return None

    print(
        f"[INFO] Simulating {asset} with {simulate_fn.__name__}: "
        f"total_hist={len(prices_dict)}, filtered={len(filter_prices_dict)}, "
        f"n_sims={num_simulations}, seed={seed}")

    result = simulate_fn(
        filter_prices_dict,
        asset=asset,
        time_increment=time_increment,
        time_length=time_length,
        n_sims=num_simulations,
        seed=seed,
        **kwargs
    )
    return result

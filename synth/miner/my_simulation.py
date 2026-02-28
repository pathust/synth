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
    sn50 L24-41
    
    Args:
        asset: Asset name (e.g., "BTC", "ETH")
        time_increment: Time increment in seconds (e.g., 300 for 5m)
        only_load: If True, only load from DB without fetching new data
    
    Returns:
        dict: {time_frame: {timestamp: price, ...}}
    """
    time_frame = "5m" if time_increment == 300 else str(time_increment)
    hist_price_data = data_handler.load_price_data(asset, time_frame)

    if not hist_price_data:
        print(f"[WARN] No historical data found for {asset}/{time_frame} in DB, fetching from API...")
        # No data in DB → fetch historical data from API (reduced to 2 weeks to avoid API limits)
        start_time_crawl = (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(second=0, microsecond=0)
            - datetime.timedelta(days=14)
        )
        hist_price_data = data_handler.fetch_multi_timeframes_price_data(
            asset, start_time_crawl, weeks=2, time_frame=time_frame
        )
    else:
        if only_load:
            total = sum(len(v) if isinstance(v, dict) else 0 for v in hist_price_data.values())
            print(f"[INFO] Loaded {total} data points for {asset}/{time_frame} (only_load=True)")
            return hist_price_data

        # Check if data is stale → fetch additional data
        last_timestamp = int(max(hist_price_data[time_frame].keys()))
        last_datetime = datetime.datetime.fromtimestamp(
            last_timestamp, datetime.timezone.utc
        )
        current_timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()
        staleness = current_timestamp - last_timestamp
        if staleness >= time_increment * 1:
            print(f"[INFO] Data stale by {staleness:.0f}s (>{time_increment}s), fetching update...")
            data_handler.fetch_multi_timeframes_price_data(
                asset, last_datetime, weeks=1, time_frame=time_frame
            )
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
    seed: Optional[int] = 42
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
    time_frame = "5m" if time_increment == 300 else str(time_increment)

    only_load = True if asset in ASSETS_PERIODIC_FETCH_PRICE_DATA else False
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
        seed=seed
    )
    return result

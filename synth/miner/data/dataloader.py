import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta

from synth.miner.data_handler import DataHandler
from synth.miner.price_aggregation import aggregate_1m_to_5m
from synth.validator.prompt_config import get_prompt_labels_for_asset, HIGH_FREQUENCY, LOW_FREQUENCY


class UnifiedDataLoader:
    """
    A unified wrapper around the legacy DataHandler.
    It enforces strict Out-Of-Sample boundaries preventing data leakage and provides 
    clean numpy arrays for Strategies and Backtest Engines.
    
    It abstracts away Pyth API execution vs DB fetching.
    """
    def __init__(self):
        self._handler = DataHandler()

    def _get_time_config(self, asset: str, frequency: Optional[str] = None):
        if frequency == "high":
            return HIGH_FREQUENCY
        if frequency == "low":
            return LOW_FREQUENCY
            
        labels = get_prompt_labels_for_asset(asset) or []
        if "high" in labels:
            return HIGH_FREQUENCY
        return LOW_FREQUENCY

    def get_historical_data(self, asset: str, end_time: str, window_days: int, frequency: Optional[str] = None) -> np.ndarray:
        """
        Retrieves exactly `window_days` worth of data strictly BEFORE `end_time`.
        Uses nearest past interpolation if some points are missing.
        
        Args:
            asset: The target asset ticker.
            end_time: The cutoff timestamp (exclusive). No data at or after this time.
            window_days: Amount of historical context required.
            frequency: Optional frequency override ("high" or "low").
            
        Returns:
            np.ndarray: 1D Historical price array.
        """
        end_dt = self._parse_iso(end_time)
        end_ts = int(end_dt.timestamp())
        start_ts = int((end_dt - timedelta(days=window_days)).timestamp())
        
        cfg = self._get_time_config(asset, frequency)
        tf = "1m" if cfg.time_increment == 60 else "5m"
        
        prices_dict = self._load_data(asset, tf)
        if not prices_dict:
            return np.array([])
            
        sorted_ts = sorted(int(k) for k in prices_dict.keys())
        if not sorted_ts:
            return np.array([])
            
        # We want to build an array matching the exact increments within [start_ts, end_ts)
        inc = cfg.time_increment
        required_ts = list(range(start_ts, end_ts, inc))
        
        path = []
        for ts in required_ts:
            if str(ts) in prices_dict:
                path.append(float(prices_dict[str(ts)]))
                continue
            # Nearest past: largest sorted_ts <= ts
            idx = np.searchsorted(sorted_ts, ts, side="right") - 1
            if idx < 0:
                path.append(float(prices_dict[str(sorted_ts[0])]))
            else:
                path.append(float(prices_dict[str(sorted_ts[idx])]))
                
        # Strict cutoff is guaranteed by required_ts stopping before end_ts
        return np.array(path, dtype=float)

    def get_historical_dict(self, asset: str, end_time: str, window_days: int, frequency: Optional[str] = None) -> Dict[str, float]:
        """
        Retrieves exactly `window_days` worth of data strictly BEFORE `end_time`
        in the legacy dictionary format needed by existing math models.
        
        Args:
            asset: The target asset ticker.
            end_time: The cutoff timestamp (exclusive).
            window_days: Amount of historical context required.
            frequency: Optional frequency override ("high" or "low").
            
        Returns:
            Dict[str, float]: {timestamp_str: price_float}
        """
        end_dt = self._parse_iso(end_time)
        end_ts = int(end_dt.timestamp())
        start_ts = int((end_dt - timedelta(days=window_days)).timestamp())
        
        cfg = self._get_time_config(asset, frequency)
        tf = "1m" if cfg.time_increment == 60 else "5m"
        
        prices_dict = self._load_data(asset, tf)
        if not prices_dict:
            return {}
            
        # Filter strictly out-of-sample
        filter_prices_dict = {
            k: float(v) for k, v in prices_dict.items() if start_ts <= int(k) < end_ts
        }
        
        # Sort ascending
        sorted_items = sorted(filter_prices_dict.items(), key=lambda x: int(x[0]))
        return dict(sorted_items)

    def get_future_data(self, asset: str, start_time: str, time_length: int) -> np.ndarray:
        """
        Retrieves the exact target period prices used for evaluating predictions.
        Should only be called by the BacktestEngine, NEVER by a Strategy.
        """
        start_dt = self._parse_iso(start_time)
        start_ts = int(start_dt.timestamp())
        cfg = self._get_time_config(asset)
        inc = cfg.time_increment
        
        num_steps = time_length // inc + 1
        required_ts = [start_ts + k * inc for k in range(num_steps)]
        
        tf = "1m" if inc == 60 else "5m"
        prices_dict = self._load_data(asset, tf)
        if not prices_dict:
            return np.array([])
            
        sorted_ts = sorted(int(k) for k in prices_dict.keys())
        if not sorted_ts:
            return np.array([])
            
        path = []
        for ts in required_ts:
            if str(ts) in prices_dict:
                path.append(float(prices_dict[str(ts)]))
                continue
            idx = np.searchsorted(sorted_ts, ts, side="right") - 1
            if idx < 0:
                path.append(float(prices_dict[str(sorted_ts[0])]))
            else:
                path.append(float(prices_dict[str(sorted_ts[idx])]))
                
        return np.array(path, dtype=float)
        
    def _load_data(self, asset: str, tf: str) -> Dict[str, float]:
        loaded = self._handler.load_price_data(asset, tf, load_from_file=False)
        if loaded and tf in loaded and loaded[tf]:
            return loaded[tf]
        
        # If 5m is missing, try 1m and aggregate
        if tf == "5m":
            loaded_1m = self._handler.load_price_data(asset, "1m", load_from_file=False)
            if loaded_1m and "1m" in loaded_1m and loaded_1m["1m"]:
                aggregated = aggregate_1m_to_5m(loaded_1m["1m"])
                if aggregated:
                    return aggregated
                    
        return {}

    @staticmethod
    def _parse_iso(iso_str: str) -> datetime:
        d = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d

"""
data_handler.py

Ported from sn50/synth/miner/data_handler.py (604 lines).

Changes from sn50:
    1. MongoDBHandler → MySQLHandler (L18, L348, L379-389)
    2. Removed MongoDBHandler class entirely (sn50 L403-533)
    3. Kept all DataHandler methods intact:
       - fetch_cm_data()
       - fetch_synth_data() 
       - get_real_prices()
       - fetch_multi_timeframes_price_data()
       - load_price_data()
       - save_price_data()
       - _transform_data()
    4. Removed validation score methods (not needed for backtest):
       - get_validation_scores_historical()
       - _postprocess_validation_log()
       - get_latest_validation_scores()
       - get_miner_reward_scores()
"""

import requests
import datetime
import os
import json
import time
import traceback
import numpy as np
from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from synth.db.models import ValidatorRequest
from synth.validator.price_data_provider import PriceDataProvider
from synth.miner.coinmetric_client import CMData
from synth.miner.constants import ASSETS_PERIODIC_FETCH_PRICE_DATA
from synth.miner.mysql_handler import MySQLHandler


class DataHandler:
    """
    Central data handler for the miner.
    
    Ported from sn50/synth/miner/data_handler.py (class DataHandler, L15-401).
    
    Key change: self.mongo_handler → self.mysql_handler
    """
    
    def __init__(self):
        self.sn50_price_data_provider = PriceDataProvider()
        self.mysql_handler = MySQLHandler()  # Was: MongoDBHandler()
        self.cm_data = CMData()

        # Possible resolutions for Pyth TradingView API
        # sn50 L21-22
        self.sn50_supported_resolutions = [
            1, 2, 5, 15, 30, 60, 120, 240, 360, 720,
            "D", "1D", "W", "1W", "M", "1M"
        ]
        self.DATA_DIR = "synth/miner/data"
        self.MAX_DATA_POINTS_TO_KEEP = 150000

        # Map timeframe name → time_increment (in seconds)
        # sn50 L28-31
        self.TIMEFRAME_TO_TIME_INCREMENT = {
            "5m": 5 * 60,
        }
        self.TIME_INCREMENT_TO_TIMEFRAME = {
            v: k for k, v in self.TIMEFRAME_TO_TIME_INCREMENT.items()
        }

    def fetch_cm_data(
        self,
        asset: str,
        start_time: Optional[Union[datetime.datetime, datetime.date, str]],
        end_time: Optional[Union[datetime.datetime, datetime.date, str]],
        time_increment: int,
    ):
        """
        Fetch price data from CoinMetrics API.
        sn50 L33-43
        """
        print(f"[CM] Fetching {asset} from {start_time} to {end_time}, increment={time_increment}")
        try:
            data = self.cm_data.get_CM_ReferenceRate(
                assets=[asset], start=start_time, end=end_time,
                frequency="1m", use_cache=False
            )
            if data is None or data.empty:
                print(f"[CM] No data returned for {asset}")
                return {}
            prices_dict = self.cm_data._pd_to_dict(data)
            if not prices_dict:
                print(f"[CM] Empty prices_dict after conversion for {asset}")
                return {}
            for asset_, prices in prices_dict.items():
                transformed = self.cm_data._transform_prices_to_time_frames(
                    prices, start_time, end_time, time_frames=[time_increment]
                )
                result = transformed.get(str(time_increment), {})
                print(f"[CM] Got {len(result)} data points for {asset}")
                return result
            return {}
        except Exception as e:
            print(f"[CM] Error fetching {asset}: {e}")
            traceback.print_exc()
            return {}

    def fetch_synth_data(
        self, asset: str, start_time: datetime.datetime,
        time_length: int, time_increment: int
    ):
        """
        Fetch price data from Pyth TradingView API (Synth's price data provider).
        sn50 L45-63
        """
        print(
            f"[Synth] Fetching {asset} time_increment={time_increment} "
            f"from {start_time} to {start_time + datetime.timedelta(seconds=time_length)}"
        )
        try:
            req = ValidatorRequest(
                asset=asset,
                start_time=start_time,
                time_length=time_length,
                time_increment=time_increment,
            )

            q_inc = 1  # Always use resolution=1 for raw data
            q = {
                "validator_request": req,
                "resolution": q_inc,
                "return_raw_data": True,
            }
            data = self.get_real_prices(**q)
            if data is None:
                print(f"[Synth] No data returned for {asset}")
                return {}
            transformed_data = self._transform_data(
                data, int(start_time.timestamp()), time_increment, time_length
            )
            print(f"[Synth] Got {len(transformed_data)} data points for {asset}")
            return transformed_data
        except Exception as e:
            print(f"[Synth] Error fetching {asset}: {e}")
            traceback.print_exc()
            return {}

    def get_real_prices(self, **kwargs):
        """
        Fetch real prices from PriceDataProvider (Pyth API).
        sn50 L64-65
        """
        return self.sn50_price_data_provider.fetch_data(**kwargs)

    def fetch_multi_timeframes_price_data(
        self,
        asset: str,
        start_time: datetime.datetime,
        weeks: int = 1,
        time_frame: str = "5m"
    ):
        """
        Fetch historical price data for a specific timeframe.
        Races CoinMetrics vs Pyth API, takes whichever returns first.
        Saves results to MySQL.
        
        sn50 L221-301
        
        Returns dict:
        {
            "5m": {"timestamp": "price", ...},
        }
        """
        import math

        MAX_DATA_POINTS_PER_REQUEST = 1000
        time_length = (
            datetime.datetime.now(datetime.timezone.utc) - start_time
            + datetime.timedelta(days=1)
        ).total_seconds()

        results = {}

        if time_frame in self.TIMEFRAME_TO_TIME_INCREMENT:
            tf_name = time_frame
            inc = self.TIMEFRAME_TO_TIME_INCREMENT[time_frame]
        else:
            tf_name = time_frame
            inc = int(time_frame)

        num_data_points = time_length / inc
        num_requests = math.ceil(num_data_points / MAX_DATA_POINTS_PER_REQUEST)
        results[tf_name] = {}

        for i in range(num_requests):
            start_time_i = start_time + datetime.timedelta(
                seconds=i * MAX_DATA_POINTS_PER_REQUEST * inc
            )
            if start_time_i > datetime.datetime.now(datetime.timezone.utc):
                break

            time_length_i = int(min(
                MAX_DATA_POINTS_PER_REQUEST * inc,
                (start_time + datetime.timedelta(seconds=time_length) - start_time_i).total_seconds()
            ))

            st_time = time.time()
            transformed_data = {}
            executor = ThreadPoolExecutor(max_workers=2)
            try:
                # Submit both tasks concurrently (sn50 L264-268)
                future_cm = (
                    executor.submit(
                        self.fetch_cm_data, asset, start_time_i,
                        start_time_i + datetime.timedelta(seconds=time_length_i), inc
                    )
                    if asset not in ASSETS_PERIODIC_FETCH_PRICE_DATA
                    else None
                )
                future_synth = executor.submit(
                    self.fetch_synth_data, asset, start_time_i, time_length_i, inc
                )

                # Take result from whichever completes first (sn50 L270-292)
                futures_list = [future_synth]
                futures_dict = {future_synth: "Synth"}
                if future_cm is not None:
                    futures_list.append(future_cm)
                    futures_dict[future_cm] = "CM"

                for future in as_completed(futures_list):
                    try:
                        result = future.result()
                        source = futures_dict[future]
                        if result and len(result) > 0:
                            transformed_data = result
                            print(
                                f"[{source}] Fetch completed, "
                                f"got {len(transformed_data)} data points for {asset} {time_frame}"
                            )
                            # Cancel the other future
                            if future == future_cm:
                                future_synth.cancel()
                            elif future_cm is not None:
                                future_cm.cancel()
                            break
                        else:
                            print(f"[{source}] Returned empty data for {asset} {time_frame}")
                    except Exception as e:
                        source = futures_dict[future]
                        print(f"[{source}] Error fetching data: {e}")
                        traceback.print_exc()
                        continue
            finally:
                executor.shutdown(wait=False)

            if transformed_data:
                results[tf_name].update(transformed_data)
            elapsed = time.time() - st_time
            print(
                f"[DataHandler] Batch {i+1}/{num_requests}: "
                f"{len(transformed_data)} new points, "
                f"{len(results[tf_name])} total, "
                f"{elapsed:.1f}s elapsed"
            )

        total_points = sum(len(v) for v in results.values())
        print(f"[DataHandler] Fetch complete for {asset}/{time_frame}: {total_points} total data points")
        if total_points > 0:
            self.save_price_data(asset, time_frame, results)
        else:
            print(f"[WARN] No data fetched for {asset}/{time_frame}, skipping save")
        return results

    @staticmethod
    def _transform_data(
        data, start_time_int: int, time_increment: int, time_length: int
    ) -> dict:
        """
        Transform raw Pyth API response into {timestamp: price} dict.
        sn50 L303-335
        """
        if data is None or len(data) == 0 or len(data["t"]) == 0:
            return {}

        time_end_int = start_time_int + time_length
        timestamps = [
            t
            for t in range(
                start_time_int, time_end_int + time_increment, time_increment
            )
        ]

        if len(timestamps) != int(time_length / time_increment) + 1:
            if len(timestamps) == int(time_length / time_increment) + 2:
                if data["t"][-1] < timestamps[1]:
                    timestamps = timestamps[:-1]
                elif data["t"][0] > timestamps[0]:
                    timestamps = timestamps[1:]
            else:
                return {}

        close_prices_dict = {t: c for t, c in zip(data["t"], data["c"])}
        transformed_data = {}

        for idx, t in enumerate(timestamps):
            if t in close_prices_dict:
                transformed_data[str(t)] = close_prices_dict[t]

        return transformed_data

    def load_price_data(
        self, asset: str, time_frame: str, load_from_file: bool = False
    ):
        """
        Load price data from MySQL or file.
        sn50 L337-352. Changed: mongo_handler → mysql_handler
        """
        if load_from_file:
            load_path = os.path.join(
                self.DATA_DIR, f"{asset}_{time_frame}_prices.json"
            )
            try:
                with open(load_path, "r") as f:
                    data = json.load(f)
                return data
            except Exception as e:
                print(f"[ERROR] Failed to load price data from file: {load_path}: {e}")
                return {}
        else:
            dt = self.mysql_handler.load_price_data(asset, time_frame)
            if isinstance(dt, dict) and "prices" in dt:
                prices = dt.get("prices", {})
                total = sum(len(v) if isinstance(v, dict) else 0 for v in prices.values())
                print(f"[DataHandler] Loaded {total} data points for {asset}/{time_frame} from DB")
                return prices
            else:
                print(f"[DataHandler] No data in DB for {asset}/{time_frame}")
                return {}

    def save_price_data(
        self, asset: str, time_frame: str, prices: dict,
        load_from_file: bool = False
    ):
        """
        Save price data to MySQL or file. Merges with existing data.
        sn50 L354-391. Changed: mongo_handler → mysql_handler
        """

        def keep_last_k_from_dict(data: dict, k: int):
            """Keep only the last k entries sorted by timestamp."""
            sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
            kept_items = dict(sorted_items[-k:])
            return kept_items

        if load_from_file:
            save_path = os.path.join(
                self.DATA_DIR, f"{asset}_{time_frame}_prices.json"
            )
            if os.path.exists(save_path):
                latest_prices = self.load_price_data(asset, time_frame, load_from_file=True)
                for tf in prices.keys():
                    latest_prices.get(tf, {}).update(prices[tf])
                    latest_prices[tf] = keep_last_k_from_dict(
                        latest_prices[tf], k=self.MAX_DATA_POINTS_TO_KEEP
                    )
            else:
                latest_prices = prices

            with open(save_path, "w") as f:
                json.dump(latest_prices, f, ensure_ascii=False)
        else:
            # Changed from sn50: mongo_handler → mysql_handler
            latest_prices_info = self.mysql_handler.load_price_data(asset, time_frame)
            if latest_prices_info:
                latest_prices = latest_prices_info.get("prices", {})
                for tf in prices.keys():
                    if tf not in latest_prices:
                        latest_prices[tf] = {}
                    latest_prices[tf].update(prices[tf])
                    latest_prices[tf] = keep_last_k_from_dict(
                        latest_prices[tf], k=self.MAX_DATA_POINTS_TO_KEEP
                    )
            else:
                latest_prices = prices
            self.mysql_handler.save_price_data(asset, time_frame, latest_prices)

        return latest_prices

    def compare_data(
        self, asset: str, start_time: datetime.datetime,
        time_length: int, time_increment: int
    ):
        """
        Debug utility: compare data from both sources.
        sn50 L393-401
        """
        print(
            f"Fetching data for {asset} time_increment: {time_increment} "
            f"from {start_time} to {start_time + datetime.timedelta(seconds=time_length)}"
        )
        synth_data = self.fetch_synth_data(asset, start_time, time_length, time_increment)
        if asset == 'XAU':
            cm_data = None
        else:
            cm_data = self.fetch_cm_data(
                asset, start_time,
                start_time + datetime.timedelta(seconds=time_length), time_increment
            )
        print(f"Synth data: {synth_data}")
        print(f"CM data: {cm_data}")

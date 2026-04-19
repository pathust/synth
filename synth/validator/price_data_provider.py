import logging
import time
import requests


from tenacity import (
    before_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import numpy as np
try:
    import bittensor as bt
    _retry_before = before_log(bt.logging._logger, logging.DEBUG)
except ImportError:
    bt = None
    _retry_before = None


from synth.db.models import ValidatorRequest
from synth.utils.helpers import from_iso_to_unix_time
from synth.utils.logging import print_execution_time

# Pyth API benchmarks doc: https://benchmarks.pyth.network/docs
# get the list of stocks supported by pyth: https://benchmarks.pyth.network/v1/shims/tradingview/symbol_info?group=pyth_stock
# get the list of crypto supported by pyth: https://benchmarks.pyth.network/v1/shims/tradingview/symbol_info?group=pyth_crypto
# get the ticket: https://benchmarks.pyth.network/v1/shims/tradingview/symbols?symbol=Metal.XAU/USD


class PriceDataProvider:
    PYTH_BASE_URL = "https://benchmarks.pyth.network/v1/shims/tradingview/history"
    HYPERLIQUID_BASE_URL = "https://api.hyperliquid.xyz/info"

    # Assets fetched from Pyth
    PYTH_SYMBOL_MAP = {
        "BTC": "Crypto.BTC/USD",
        "ETH": "Crypto.ETH/USD",
        "XAU": "Crypto.XAUT/USD",
        "SOL": "Crypto.SOL/USD",
        "SPYX": "Crypto.SPYX/USD",
        "NVDAX": "Crypto.NVDAX/USD",
        "TSLAX": "Crypto.TSLAX/USD",
        "AAPLX": "Crypto.AAPLX/USD",
        "GOOGLX": "Crypto.GOOGLX/USD",
        "XRP": "Crypto.XRP/USD",
        "HYPE": "Crypto.HYPE/USD",
    }

    # Assets fetched from Hyperliquid (overrides Pyth for these assets)
    HYPERLIQUID_SYMBOL_MAP = {
        "WTIOIL": "xyz:CL",
    }

    @staticmethod
    def assert_assets_supported(asset_list: list[str]):
        supported = (
            PriceDataProvider.PYTH_SYMBOL_MAP.keys()
            | PriceDataProvider.HYPERLIQUID_SYMBOL_MAP.keys()
        )
        for asset in asset_list:
            assert asset in supported

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=7),
        reraise=True,
        **(dict(before=_retry_before) if _retry_before else {}),
    )
    @print_execution_time
    def fetch_data(
        self,
        validator_request: ValidatorRequest,
        resolution: int = 1,
        return_raw_data: bool = False,
    ) -> list:
        """
        Fetch real prices data from an external REST service.
        Returns an array of time points with prices.

        Args:
            validator_request: ValidatorRequest with asset, start_time, etc.
            resolution: Resolution for Pyth API (1=1min, default)
            return_raw_data: If True, return raw API JSON instead of transformed data

        :return: List of dictionaries with 'time' and 'price' keys.
        """

        start_time_int = from_iso_to_unix_time(
            validator_request.start_time.isoformat()
        )
        end_time_int = start_time_int + validator_request.time_length

        asset = str(validator_request.asset)

        # ── Hyperliquid override (WTIOIL) ────────────────────────────────
        if asset in self.HYPERLIQUID_SYMBOL_MAP:
            prices = self._fetch_data_hyperliquid(validator_request)
            if not prices or np.isnan(prices[-1]):
                raise ValueError(
                    f"missing price data for the last timestamp for asset {asset} in request {validator_request.id}"
                )
            return prices

        # ── Default: Pyth ───────────────────────────────────────────────
        params = {
            "symbol": self.PYTH_SYMBOL_MAP[asset],
            "resolution": resolution,
            "from": start_time_int,
            "to": end_time_int,
        }

        response = requests.get(self.PYTH_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if return_raw_data:
            return data

        transformed_data = self._transform_data(
            data,
            start_time_int,
            int(validator_request.time_increment),
            int(validator_request.time_length),
        )

        # Important: retry if realized last price is missing.
        if not transformed_data or np.isnan(transformed_data[-1]):
            raise ValueError(
                f"missing price data for the last timestamp for asset {asset} in request {validator_request.id}"
            )

        return transformed_data

    def _fetch_data_hyperliquid(self, validator_request: ValidatorRequest) -> list:
        """
        Fetch 1m candles from Hyperliquid, then align to (time_increment, time_length)
        grid by transforming to the same list format as Pyth.
        """
        start_time_int = from_iso_to_unix_time(
            validator_request.start_time.isoformat()
        )
        raw = self._download_hyperliquid_candles(
            beginning=start_time_int,
            end=start_time_int + int(validator_request.time_length),
            symbol=str(validator_request.asset),
        )
        # raw candles are 1m; convert to Pyth-like dict {t:[], c:[]}
        if not raw:
            return []
        # Ensure sorted by timestamp
        raw_sorted = sorted(raw, key=lambda x: int(x.get("t", 0)))
        data = {
            "t": [int(x["t"]) for x in raw_sorted if "t" in x and "c" in x],
            "c": [float(x["c"]) for x in raw_sorted if "t" in x and "c" in x],
        }
        return self._transform_data(
            data,
            start_time_int,
            int(validator_request.time_increment),
            int(validator_request.time_length),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=2),
        reraise=True,
        **(dict(before=_retry_before) if _retry_before else {}),
    )
    def _download_hyperliquid_candles(
        self,
        *,
        beginning: int,  # seconds
        end: int,        # seconds
        symbol: str = "WTIOIL",
        loop_wait_time_seconds: float = 0.1,
    ) -> list[dict]:
        """
        Download 1m candle snapshots from Hyperliquid, chunked.
        Returns list of {t: unix_seconds, c: close_price}.
        """
        max_candles = 5000
        interval_ms = 60 * 1000
        chunk_ms = max_candles * interval_ms

        begin_ms = int(beginning) * 1000
        end_ms = int(end) * 1000
        out: list[dict] = []

        coin = self.HYPERLIQUID_SYMBOL_MAP.get(symbol)
        if not coin:
            return []

        with requests.Session() as session:
            cur = begin_ms
            while cur < end_ms:
                cur_end = min(cur + chunk_ms, end_ms)
                payload = {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": coin,
                        "interval": "1m",
                        "startTime": cur,
                        "endTime": cur_end,
                    },
                }
                resp = session.post(
                    self.HYPERLIQUID_BASE_URL, json=payload, timeout=100
                )
                resp.raise_for_status()
                data = resp.json() or []
                for c in data:
                    try:
                        ts = int(c.get("t")) // 1000
                        out.append({"t": ts, "c": float(c.get("c"))})
                    except Exception:
                        continue
                cur = cur_end
                time.sleep(loop_wait_time_seconds)

        return out

    @staticmethod
    def _transform_data(
        data, start_time_int: int, time_increment: int, time_length: int
    ) -> list:
        if data is None or len(data) == 0 or len(data["t"]) == 0:
            return []

        time_end_int = start_time_int + time_length
        timestamps = [
            t
            for t in range(
                start_time_int, time_end_int + time_increment, time_increment
            )
        ]

        if len(timestamps) != int(time_length / time_increment) + 1:
            # Note: this part of code should never be activated; just included for precaution
            if len(timestamps) == int(time_length / time_increment) + 2:
                if data["t"][-1] < timestamps[1]:
                    timestamps = timestamps[:-1]
                elif data["t"][0] > timestamps[0]:
                    timestamps = timestamps[1:]
            else:
                return []

        close_prices_dict = {t: c for t, c in zip(data["t"], data["c"])}
        transformed_data = [np.nan for _ in range(len(timestamps))]

        for idx, t in enumerate(timestamps):
            if t in close_prices_dict:
                transformed_data[idx] = close_prices_dict[t]

        return transformed_data

    @staticmethod
    def _get_token_mapping(token: str) -> str:
        """
        Retrieve the mapped value for a given token.
        If the token is not in the map, raise an exception or return None.
        """
        if token in PriceDataProvider.PYTH_SYMBOL_MAP:
            return PriceDataProvider.PYTH_SYMBOL_MAP[token]
        raise ValueError(f"Token '{token}' is not supported.")

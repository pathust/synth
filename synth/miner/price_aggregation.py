"""
price_aggregation.py

Aggregate 1m OHLC/close price data to 5m. Used when 5m data is not in DB
but 1m is available (e.g. run.py, my_simulation fetch_price_data).
"""

from collections import defaultdict
from typing import Dict, Union


def aggregate_1m_to_5m(prices_1m: Dict[Union[str, int], float]) -> Dict[str, float]:
    """
    Aggregate 1-minute candle prices to 5-minute candles.

    Each 5m bucket is (timestamp // 300) * 300. The close of the 5m candle
    is the price at the last 1m in that bucket (standard OHLC aggregation).

    Args:
        prices_1m: Dict mapping 1m timestamp (str or int) -> price (close of that minute).

    Returns:
        Dict mapping 5m bucket timestamp (str) -> close price for that 5m candle.
    """
    if not prices_1m:
        return {}
    buckets: Dict[int, Dict[int, float]] = defaultdict(dict)
    for ts, price in prices_1m.items():
        t = int(ts)
        bucket = (t // 300) * 300
        buckets[bucket][t] = float(price)
    out: Dict[str, float] = {}
    for bucket_ts in sorted(buckets.keys()):
        last_ts = max(buckets[bucket_ts].keys())
        out[str(bucket_ts)] = buckets[bucket_ts][last_ts]
    return out

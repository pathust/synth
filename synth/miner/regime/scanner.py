"""Helpers to sample and classify backtest dates by regime bias."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from synth.miner.regime.pattern import detect_pattern_v2


def get_random_dates(
    start_date: datetime,
    end_date: datetime,
    num_dates: int,
    seed: int = 42,
) -> list[datetime]:
    """Generate random timestamps between start/end, rounded to 5m."""
    random.seed(seed)
    total_seconds = int((end_date - start_date).total_seconds())
    if total_seconds <= 0:
        return [start_date]

    dates: list[datetime] = []
    for _ in range(num_dates):
        random_seconds = random.randint(0, total_seconds)
        date = start_date + timedelta(seconds=random_seconds)
        rounded = date.replace(
            minute=date.minute - (date.minute % 5),
            second=0,
            microsecond=0,
        )
        dates.append(rounded)
    return dates


def classify_regime_bias(prices_dict: dict[str, float]) -> str:
    """Classify one window into bullish/bearish/neutral."""
    pattern_data = detect_pattern_v2(prices_dict)
    bias = str(pattern_data.get("bias", "neutral")).lower()
    if bias not in {"bullish", "bearish", "neutral"}:
        return "neutral"
    return bias


def scan_regime_dates(
    prices_dict: dict[str, float],
    start_date: datetime,
    end_date: datetime,
    num_per_regime: int = 5,
    pool_size: int = 150,
    seed: int = 42,
    window_points: int = 181,
) -> dict[str, list[datetime]]:
    """
    Scan random dates and group them into bullish/bearish/neutral buckets.
    """
    random.seed(seed)
    regimes: dict[str, list[datetime]] = {
        "bullish": [],
        "bearish": [],
        "neutral": [],
    }

    if not prices_dict:
        return regimes

    candidate_dates = get_random_dates(start_date, end_date, pool_size, seed)
    print(
        f"[RegimeScanner] Scanning {pool_size} dates "
        f"to find regimes..."
    )

    window_seconds = window_points * 60
    for date in candidate_dates:
        if all(len(dates) >= num_per_regime for dates in regimes.values()):
            break

        ts_end = int(date.timestamp())
        window_start = ts_end - window_seconds
        filtered_window = {
            ts: price
            for ts, price in prices_dict.items()
            if window_start <= int(ts) < ts_end
        }
        if len(filtered_window) < window_points:
            continue

        bias = classify_regime_bias(filtered_window)
        if len(regimes[bias]) < num_per_regime:
            regimes[bias].append(date)

    for bias, dates in regimes.items():
        print(f"  -> Found {len(dates)} {bias.upper()} dates")
    return regimes

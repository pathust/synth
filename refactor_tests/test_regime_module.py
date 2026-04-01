from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from synth.miner.core.regime_detection import (
    REGIME_TYPE,
    detect_market_regime_with_er,
)
from synth.miner.regime import (
    detect_market_regime_with_bbw,
    get_random_dates,
    scan_regime_dates,
)


def test_er_detector_shim_back_compat():
    prices = pd.Series([100.0, 100.2, 100.1, 100.4, 100.3])
    result = detect_market_regime_with_er(prices, lookback=20)

    assert result["type"] in {REGIME_TYPE.TRENDING, REGIME_TYPE.SIDEWAYS}
    assert isinstance(result["strength"], float)


def test_bbw_detector_handles_short_input():
    prices = pd.Series([10.0, 10.1, 10.2])
    result = detect_market_regime_with_bbw(prices)

    assert set(result.keys()) == {"is_squeeze", "is_trending", "bbw_ratio"}
    assert isinstance(result["is_squeeze"], bool)
    assert isinstance(result["is_trending"], bool)
    assert isinstance(result["bbw_ratio"], float)


def test_get_random_dates_rounds_to_five_minutes():
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    end = datetime(2026, 3, 5, tzinfo=timezone.utc)
    dates = get_random_dates(start, end, num_dates=20, seed=123)

    assert len(dates) == 20
    assert all(d.minute % 5 == 0 for d in dates)
    assert all(d.second == 0 and d.microsecond == 0 for d in dates)


def test_scan_regime_dates_with_detector_hook(monkeypatch):
    def fake_detect_pattern(prices_dict):
        last_ts = int(max(prices_dict.keys(), key=int))
        mod = last_ts % 3
        if mod == 0:
            bias = "bullish"
        elif mod == 1:
            bias = "bearish"
        else:
            bias = "neutral"
        return {"bias": bias}

    monkeypatch.setattr(
        "synth.miner.regime.scanner.detect_pattern_v2",
        fake_detect_pattern,
    )

    data_start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    start = data_start + timedelta(hours=6)
    end = data_start + timedelta(days=2)

    prices_dict = {}
    for idx in range(0, 60 * 72):
        ts = int((data_start + timedelta(minutes=idx)).timestamp())
        prices_dict[str(ts)] = 100.0 + (idx * 0.01)

    regimes = scan_regime_dates(
        prices_dict=prices_dict,
        start_date=start,
        end_date=end,
        num_per_regime=2,
        pool_size=120,
        seed=7,
    )

    assert set(regimes.keys()) == {"bullish", "bearish", "neutral"}
    assert all(len(dates) <= 2 for dates in regimes.values())
    assert sum(len(dates) for dates in regimes.values()) > 0

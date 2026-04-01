"""
Regime detection for Strategy Selector (docs/ARCHITECTURE.md).

Maps to REGIME_TYPES in strategies.base:
- crypto: bull / high_vol / ranging
- gold: trending / mean_reverting
- equity: market_open / overnight / earnings
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone

import pandas as pd

from synth.miner.strategies.base import get_asset_type
from synth.miner.regime.detectors import (
    detect_market_regime_with_bbw,
    detect_market_regime_with_er,
)
from synth.miner.regime.pattern import detect_pattern_v2
from synth.miner.regime.types import RegimeType


@dataclass
class RegimeResult:
    asset_type: str
    regime: str
    confidence: float = 0.0
    meta: dict | None = None


def _detect_equity_regime(start_time: str) -> RegimeResult:
    try:
        dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
    except Exception:
        return RegimeResult("equity", "market_open", 0.0, None)

    t = dt.time()
    if dt.weekday() < 5 and dtime(14, 30) <= t < dtime(15, 0):
        return RegimeResult("equity", "earnings", 0.6, None)
    if dt.weekday() < 5 and dtime(14, 30) <= t < dtime(21, 0):
        return RegimeResult("equity", "market_open", 0.6, None)
    return RegimeResult("equity", "overnight", 0.6, None)


def _detect_gold_regime(history_dict: dict[str, float]) -> RegimeResult:
    try:
        keys = sorted(history_dict.keys(), key=lambda x: int(x))
        if len(keys) < 50:
            return RegimeResult("gold", "mean_reverting", 0.0, None)
        tail = keys[-800:]
        s = pd.Series([float(history_dict[k]) for k in tail])
        info = detect_market_regime_with_bbw(s)
        regime = "trending" if bool(info.get("is_trending")) else "mean_reverting"
        conf = float(min(1.0, abs(float(info.get("bbw_ratio", 1.0)) - 1.0)))
        return RegimeResult("gold", regime, conf, {"bbw": info})
    except Exception:
        return RegimeResult("gold", "mean_reverting", 0.0, None)


def _detect_crypto_high(history_dict: dict[str, float]) -> RegimeResult:
    keys = sorted(history_dict.keys(), key=lambda x: int(x))
    tail_keys = keys[-181:]
    if len(tail_keys) < 181:
        return RegimeResult("crypto", "ranging", 0.0, None)
    window = {k: history_dict[k] for k in tail_keys}
    out = detect_pattern_v2(window)
    bias = str(out.get("bias", "neutral")).lower()
    score = abs(float(out.get("bias_score", 0.0)))
    if bias == "bullish":
        return RegimeResult("crypto", "bull", score, {"pattern": out})
    if bias == "bearish":
        return RegimeResult("crypto", "high_vol", score, {"pattern": out})
    return RegimeResult("crypto", "ranging", 0.3, {"pattern": out})


def _detect_crypto_low(history_dict: dict[str, float]) -> RegimeResult:
    try:
        keys = sorted(history_dict.keys(), key=lambda x: int(x))
        if len(keys) < 30:
            return RegimeResult("crypto", "ranging", 0.0, None)
        tail = keys[-2000:]
        s = pd.Series([float(history_dict[k]) for k in tail])
        info = detect_market_regime_with_er(s, lookback=20)
        if info.get("type") == RegimeType.TRENDING:
            return RegimeResult(
                "crypto",
                "bull",
                float(info.get("strength", 0.0)),
                {"er": info},
            )
        return RegimeResult(
            "crypto",
            "ranging",
            1.0 - float(info.get("strength", 0.0)),
            {"er": info},
        )
    except Exception:
        return RegimeResult("crypto", "ranging", 0.0, None)


def detect_regime(
    asset: str,
    start_time: str,
    time_increment: int,
    time_length: int,
    history_dict: dict[str, float],
) -> RegimeResult:
    """
    Classify market regime for strategy routing.

    HFT vs LFT is implied by time_increment/time_length (validator prompt);
    this function only outputs the regime label for StrategyStore lookup.
    """
    asset_type = get_asset_type(asset)
    if asset_type == "equity":
        return _detect_equity_regime(start_time)
    if asset_type == "gold":
        return _detect_gold_regime(history_dict)

    is_high = time_length == 3600 or time_increment <= 60
    if is_high:
        return _detect_crypto_high(history_dict)
    return _detect_crypto_low(history_dict)

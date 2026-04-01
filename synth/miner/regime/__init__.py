"""Regime module for detectors and regime-aware sampling helpers."""

from synth.miner.regime.detectors import (
    detect_market_regime_with_bbw,
    detect_market_regime_with_er,
)
from synth.miner.regime.pattern import detect_pattern_v1, detect_pattern_v2
from synth.miner.regime.scanner import (
    classify_regime_bias,
    get_random_dates,
    scan_regime_dates,
)
from synth.miner.regime.types import RegimeType


class REGIME_TYPE:
    """Backward-compatible constant holder used by legacy imports."""

    TRENDING = RegimeType.TRENDING
    SIDEWAYS = RegimeType.SIDEWAYS


__all__ = [
    "REGIME_TYPE",
    "RegimeType",
    "classify_regime_bias",
    "detect_market_regime_with_bbw",
    "detect_market_regime_with_er",
    "detect_pattern_v1",
    "detect_pattern_v2",
    "get_random_dates",
    "scan_regime_dates",
]

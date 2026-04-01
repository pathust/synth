"""Pattern detector facade for regime-aware routing/scanning."""

from __future__ import annotations

from typing import Dict

from synth.miner.strategies.pattern_detector import (
    detect_pattern as _detect_pattern_v1,
)
from synth.miner.strategies.pattern_detector_v2 import (
    detect_pattern as _detect_pattern_v2,
)


def detect_pattern_v1(prices_dict: Dict[str, float]) -> dict:
    return _detect_pattern_v1(prices_dict)


def detect_pattern_v2(prices_dict: Dict[str, float]) -> dict:
    return _detect_pattern_v2(prices_dict)


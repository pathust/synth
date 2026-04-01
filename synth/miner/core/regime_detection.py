"""Compatibility shim for old imports.

Prefer using ``synth.miner.regime`` directly for new code.
"""

from synth.miner.regime import (
    REGIME_TYPE,
    detect_market_regime_with_bbw,
    detect_market_regime_with_er,
)

__all__ = [
    "REGIME_TYPE",
    "detect_market_regime_with_bbw",
    "detect_market_regime_with_er",
]


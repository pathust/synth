"""
pattern_detector_v2.py

Price Action heuristic (1H context from 61×1m bars).

Fixes vs pattern_detector.py:
- expected_range uses sqrt-time scaling (not linear ×60)
- reversal rules filter flash wicks via wick/range and body/range
- strength uses body/range and wick/range (stable on doji)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


class Candle:
    def __init__(self, prices: np.ndarray):
        self.prices = prices
        self.open = float(prices[0])
        self.close = float(prices[-1])
        self.high = float(np.max(prices))
        self.low = float(np.min(prices))

        self.body = abs(self.close - self.open)
        self.range = max(self.high - self.low, 1e-5)
        self.upper_wick = self.high - max(self.open, self.close)
        self.lower_wick = min(self.open, self.close) - self.low

        if self.close > self.open:
            self.direction = "bull"
        elif self.close < self.open:
            self.direction = "bear"
        else:
            self.direction = "neutral"

        self.rejection = self.lower_wick - self.upper_wick
        self.momentum = self.close - self.open


def _candle_summary(c: Candle) -> dict:
    return {
        "open": float(c.open),
        "high": float(c.high),
        "low": float(c.low),
        "close": float(c.close),
        "direction": c.direction,
        "body": float(c.body),
        "range": float(c.range),
        "upper_wick": float(c.upper_wick),
        "lower_wick": float(c.lower_wick),
        "rejection": float(c.rejection),
        "momentum": float(c.momentum),
    }


def evaluate_context(a2: Candle, a1: Candle) -> Tuple[str, float, float]:
    # 5-state context (continuation / reversal / indecision)
    # Returns (pattern, strength_clamped, strength_raw) so callers can log min(1,·) clipping.
    if a2.direction == "bull" and a1.direction == "bull" and a1.body > 1.5 * a1.upper_wick:
        strength_raw = float(a1.body / a1.range)
        strength = min(1.0, strength_raw)
        return "bull_continuation", float(strength), strength_raw

    if a2.direction == "bear" and a1.direction == "bear" and a1.body > 1.5 * a1.lower_wick:
        strength_raw = float(a1.body / a1.range)
        strength = min(1.0, strength_raw)
        return "bear_continuation", float(strength), strength_raw

    # Reversal: loosened a bit vs previous v2 (0.4/0.4) per spec
    if a2.direction == "bear" and a1.lower_wick > 0.4 * a1.range and a1.body < 0.4 * a1.range:
        strength_raw = float(a1.lower_wick / a1.range)
        strength = min(1.0, strength_raw)
        return "bull_reversal", float(strength), strength_raw

    if a2.direction == "bull" and a1.upper_wick > 0.4 * a1.range and a1.body < 0.4 * a1.range:
        strength_raw = float(a1.upper_wick / a1.range)
        strength = min(1.0, strength_raw)
        return "bear_reversal", float(strength), strength_raw

    return "indecision", 0.0, 0.0


def final_bias(context_bias: str, context_strength: float, candle_t: Candle) -> Tuple[str, float]:
    c_score = 0.0
    if context_bias == "bullish":
        c_score = 0.6 * context_strength
    elif context_bias == "bearish":
        c_score = -0.6 * context_strength

    rej_ratio = candle_t.rejection / candle_t.range
    rej_score = max(-0.3, min(0.3, rej_ratio))

    mom_score = 0.1 if candle_t.momentum > 0 else (-0.1 if candle_t.momentum < 0 else 0.0)

    bias_score = c_score + rej_score + mom_score
    bias_score = max(-1.0, min(1.0, bias_score))

    if bias_score > 0.20:
        return "bullish", float(bias_score)
    if bias_score < -0.20:
        return "bearish", float(bias_score)
    return "neutral", float(bias_score)


def detect_pattern(prices_dict: Dict[str, float]) -> dict:
    sorted_ts = sorted(prices_dict.keys(), key=lambda x: int(x))
    prices = np.array([prices_dict[ts] for ts in sorted_ts], dtype=float)

    if len(prices) < 181:
        return {
            "pattern": "indecision",
            "strength": 0.0,
            "strength_raw": 0.0,
            "expected_range": 0.0,
            "last_price": float(prices[-1]) if len(prices) > 0 else 0.0,
            # Back-compat fields (router v1/v2)
            "bias": "neutral",
            "bias_score": 0.0,
            "candles": None,
        }

    t2_prices = prices[-181:-120]
    t1_prices = prices[-121:-60]
    t_prices = prices[-61:]

    c2 = Candle(t2_prices)
    c1 = Candle(t1_prices)
    c0 = Candle(t_prices)

    pattern_type, strength, strength_raw = evaluate_context(c2, c1)

    # Derive 3-state bias for backwards compatibility (and downstream scaling)
    if pattern_type.startswith("bull_"):
        bias = "bullish"
    elif pattern_type.startswith("bear_"):
        bias = "bearish"
    else:
        bias = "neutral"
    bias_score = 0.6 * float(strength) if bias == "bullish" else (-0.6 * float(strength) if bias == "bearish" else 0.0)

    diffs = np.abs(np.diff(c0.prices))
    avg_range_1m = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    expected_range = avg_range_1m * np.sqrt(60.0)

    return {
        # New primary output (5-state)
        "pattern": pattern_type,
        "strength": float(strength),
        "strength_raw": float(strength_raw),
        "expected_range": float(expected_range),
        "last_price": float(c0.close),
        "candles": {
            "c2": _candle_summary(c2),
            "c1": _candle_summary(c1),
            "c0": _candle_summary(c0),
        },

        # Back-compat output (3-state)
        "bias": bias,
        "bias_score": float(bias_score),
        "expected_range": float(expected_range),
    }


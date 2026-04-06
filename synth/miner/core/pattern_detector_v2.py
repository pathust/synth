"""
pattern_detector_v2.py

Price Action heuristic (1H context from 60×1m bars).

Fixes vs pattern_detector.py & original v2:
- Fixed array slicing to exactly 60m per candle (no overlap).
- Evaluates the most recent context (c1 -> c0) instead of old history.
- Reversal rules require liquidity sweeps (lower lows / higher highs) + flash wick filters.
- Combines sqrt-time scaling with actual recent range for expected_range.
- Fully utilizes final_bias for robust 3-state output scoring.
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

def evaluate_context(prev_candle: Candle, curr_candle: Candle) -> Tuple[str, float, float]:
    """
    Evaluates the 5-state context (continuation / reversal / indecision)
    Returns (pattern, strength_clamped, strength_raw)
    """
    # Bull Continuation: Hai nến tăng, nến hiện tại thân lớn, râu trên ngắn
    if prev_candle.direction == "bull" and curr_candle.direction == "bull" and curr_candle.body > 1.5 * curr_candle.upper_wick:
        strength_raw = float(curr_candle.body / curr_candle.range)
        strength = min(1.0, strength_raw)
        return "bull_continuation", float(strength), strength_raw

    # Bear Continuation: Hai nến giảm, nến hiện tại thân lớn, râu dưới ngắn
    if prev_candle.direction == "bear" and curr_candle.direction == "bear" and curr_candle.body > 1.5 * curr_candle.lower_wick:
        strength_raw = float(curr_candle.body / curr_candle.range)
        strength = min(1.0, strength_raw)
        return "bear_continuation", float(strength), strength_raw

    # Bull Reversal (Liquidity Sweep): Đang giảm, nhưng nến hiện tại rút râu dưới sâu QUA ĐÁY cũ
    if prev_candle.direction == "bear" and curr_candle.direction != "bear" and curr_candle.lower_wick > 0.4 * curr_candle.range and curr_candle.body < 0.4 * curr_candle.range and curr_candle.low < prev_candle.low:
        strength_raw = float(curr_candle.lower_wick / curr_candle.range)
        strength = min(1.0, strength_raw)
        return "bull_reversal", float(strength), strength_raw

    # Bear Reversal (Liquidity Sweep): Đang tăng, nhưng nến hiện tại rút râu trên QUA ĐỈNH cũ
    if prev_candle.direction == "bull" and curr_candle.direction != "bull" and curr_candle.upper_wick > 0.4 * curr_candle.range and curr_candle.body < 0.4 * curr_candle.range and curr_candle.high > prev_candle.high:
        strength_raw = float(curr_candle.upper_wick / curr_candle.range)
        strength = min(1.0, strength_raw)
        return "bear_reversal", float(strength), strength_raw

    # Default
    return "indecision", 0.0, 0.0

def final_bias(context_bias: str, context_strength: float, candle_t: Candle) -> Tuple[str, float]:
    """
    Calculates a final -1.0 to 1.0 bias score integrating macro context + micro candle momentum/rejection.
    """
    c_score = 0.0
    if context_bias == "bullish":
        c_score = 0.6 * context_strength
    elif context_bias == "bearish":
        c_score = -0.6 * context_strength

    # Rejection score (dương nếu râu dưới dài, âm nếu râu trên dài)
    rej_ratio = candle_t.rejection / candle_t.range
    rej_score = max(-0.3, min(0.3, rej_ratio))

    # Momentum score
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

    # Fallback nếu thiếu data (cần đúng 180 phút để chẻ 3 nến 1H)
    if len(prices) < 180:
        return {
            "pattern": "indecision",
            "strength": 0.0,
            "strength_raw": 0.0,
            "expected_range": 0.0,
            "last_price": float(prices[-1]) if len(prices) > 0 else 0.0,
            "bias": "neutral",
            "bias_score": 0.0,
            "candles": None,
        }

    # Cắt mảng chuẩn xác: 60 điểm cho mỗi nến 1H
    t2_prices = prices[-180:-120]
    t1_prices = prices[-120:-60]
    t_prices  = prices[-60:]

    c2 = Candle(t2_prices)
    c1 = Candle(t1_prices)
    c0 = Candle(t_prices)  # Nến sát thời điểm hiện tại nhất

    # Đánh giá Context dựa trên nến sát nhất thay vì nhìn xa về quá khứ
    pattern_type, strength, strength_raw = evaluate_context(c1, c0)

    # Quy đổi base bias
    if pattern_type.startswith("bull_"):
        context_bias = "bullish"
    elif pattern_type.startswith("bear_"):
        context_bias = "bearish"
    else:
        context_bias = "neutral"

    # Kết hợp sức mạnh của nến c0 để ra Final Score (tránh bỏ lãng phí hành vi giá gần nhất)
    final_bias_str, final_bias_score = final_bias(context_bias, strength, c0)

    # Tính toán Expected Range: Kết hợp Volatility theo lý thuyết thời gian & Range thực tế
    diffs = np.abs(np.diff(c0.prices))
    avg_range_1m = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    vol_sqrt_time = avg_range_1m * np.sqrt(60.0)
    expected_range = float((vol_sqrt_time + c0.range) / 2.0)

    return {
        # 5-state Router Output
        "pattern": pattern_type,
        "strength": float(strength),
        "strength_raw": float(strength_raw),
        "expected_range": expected_range,
        "last_price": float(c0.close),
        "candles": {
            "c2": _candle_summary(c2),
            "c1": _candle_summary(c1),
            "c0": _candle_summary(c0),
        },

        # Backwards compatible (3-state) cho các components khác
        "bias": final_bias_str,
        "bias_score": float(final_bias_score),
    }
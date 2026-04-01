import numpy as np
from typing import Dict, Tuple

class Candle:
    def __init__(self, prices: np.ndarray):
        self.prices = prices
        self.open = prices[0]
        self.close = prices[-1]
        self.high = np.max(prices)
        self.low = np.min(prices)
        
        self.body = abs(self.close - self.open)
        self.upper_wick = self.high - max(self.open, self.close)
        self.lower_wick = min(self.open, self.close) - self.low
        self.direction = "bull" if self.close > self.open else "bear"
        self.rejection = self.lower_wick - self.upper_wick
        self.momentum = self.close - self.open

def evaluate_context(a2: Candle, a1: Candle) -> Tuple[str, float]:
    eps = 1e-5
    
    # 1. Bull continuation
    if a2.direction == "bull" and a1.direction == "bull" and a1.body > 1.5 * a1.upper_wick:
        strength = min(1.0, a1.body / ((a1.upper_wick + eps) * 3))
        return "bullish", float(strength)
        
    # 2. Bear continuation
    if a2.direction == "bear" and a1.direction == "bear" and a1.body > 1.5 * a1.lower_wick:
        strength = min(1.0, a1.body / ((a1.lower_wick + eps) * 3))
        return "bearish", float(strength)
        
    # 3. Bullish reversal
    if a2.direction == "bear" and a1.rejection > a1.body:
        strength = a1.rejection / (a1.body + eps)
        return "bullish", float(min(1.0, strength))
        
    # 4. Bearish reversal
    if a2.direction == "bull" and a1.rejection < -a1.body:
        strength = abs(a1.rejection) / (a1.body + eps)
        return "bearish", float(min(1.0, strength))
        
    # 5. Indecision
    if abs(a1.rejection) < 0.3 * a1.body:
        return "neutral", 0.0
        
    # Default
    return "neutral", 0.0

def final_bias(context_bias: str, context_strength: float, candle_t: Candle) -> Tuple[str, float]:
    eps = 1e-5
    c_score = 0.0
    if context_bias == "bullish":
        c_score = 0.6 * context_strength
    elif context_bias == "bearish":
        c_score = -0.6 * context_strength
        
    rej_ratio = candle_t.rejection / (candle_t.body + eps)
    rej_score = max(-0.3, min(0.3, rej_ratio))
    
    mom_score = 0.1 if candle_t.momentum > 0 else (-0.1 if candle_t.momentum < 0 else 0)
    
    bias_score = c_score + rej_score + mom_score
    bias_score = max(-1.0, min(1.0, bias_score))
    
    if bias_score > 0.15:
        return "bullish", float(bias_score)
    elif bias_score < -0.15:
        return "bearish", float(bias_score)
    else:
        return "neutral", float(bias_score)

def detect_pattern(prices_dict: Dict[str, float]) -> dict:
    """
    Analyzes the 1-minute historical prices to detect 1h biases based on the Precog pattern document.
    """
    # Requires 1-minute data for analysis. The dictionary keys must be string timestamps.
    sorted_ts = sorted(prices_dict.keys(), key=lambda x: int(x))
    prices = np.array([prices_dict[ts] for ts in sorted_ts], dtype=float)
    
    if len(prices) < 181:
        # Not enough data for three 61-point 1h candles
        return {"bias": "neutral", "bias_score": 0.0, "expected_range": 0.0, "last_price": float(prices[-1]) if len(prices) > 0 else 0.0}
        
    # Extract the last 3 hours (61 items each, where overlap is normal for open/close continuity)
    t2_prices = prices[-181:-120]
    t1_prices = prices[-121:-60]
    t_prices = prices[-61:]
    
    c2 = Candle(t2_prices)
    c1 = Candle(t1_prices)
    c0 = Candle(t_prices)
    
    c_bias, c_str = evaluate_context(c2, c1)
    bias, b_score = final_bias(c_bias, c_str, c0)
    
    # expected_range = avg_range_1m * 60
    diffs = np.abs(np.diff(c0.prices))
    avg_range_1m = np.mean(diffs) if len(diffs) > 0 else 0.0
    expected_range = avg_range_1m * 60.0
    
    return {
        "bias": bias,
        "bias_score": b_score,
        "expected_range": float(expected_range),
        "last_price": float(c0.close),
        "context_bias": c_bias,
        "context_strength": c_str
    }

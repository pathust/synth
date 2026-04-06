"""Market regime detector utilities."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from synth.miner.regime.types import RegimeType


def _to_price_series(prices: pd.Series) -> pd.Series:
    if prices is None:
        return pd.Series(dtype=float)
    cleaned = pd.Series(prices, dtype=float).dropna()
    return cleaned.sort_index()


def detect_market_regime_with_bbw(
    prices: pd.Series,
    window: int = 72,      # Mặc định nhìn 6 giờ (72 nến) để đo Volatility (BBW)
    long_window: int = 288 # Mặc định nhìn 24 giờ (288 nến) để đo Trend (EMA)
) -> dict:
    """
    Detect squeeze/trend state using Bollinger Band Width + Normalized EMA spread.
    Provides direction (+1/-1) and normalized trend strength (Z-Trend).
    """
    series = _to_price_series(prices)
    
    # An toàn dữ liệu
    if len(series) < long_window:
        long_window = max(10, len(series) // 2)
        window = max(5, long_window // 4)

    if len(series) < 4:
        return {
            "is_squeeze": False,
            "is_trending": False,
            "bbw_ratio": 1.0,
            "direction": 1,
            "z_trend": 0.0
        }

    # --- 1. BOLLINGER BAND WIDTH (Đo độ nén/nở biến động) ---
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    
    # Chống chia cho 0
    sma_safe = sma.replace(0.0, pd.NA).bfill().fillna(1.0)
    bbw = (4.0 * std) / sma_safe  # Upper - Lower = (SMA+2std) - (SMA-2std) = 4std

    current_bbw = float(bbw.iloc[-1]) if pd.notna(bbw.iloc[-1]) else 0.0
    
    # Trung bình BBW của 24h qua để xem hiện tại đang Nở (Breakout) hay Nén (Squeeze)
    avg_bbw_series = bbw.rolling(long_window).mean()
    avg_bbw = float(avg_bbw_series.iloc[-1]) if pd.notna(avg_bbw_series.iloc[-1]) else max(current_bbw, 1e-12)

    bbw_ratio = float(current_bbw / avg_bbw) if avg_bbw > 0 else 1.0

    # --- 2. XÁC ĐỊNH XU HƯỚNG BẰNG ĐỘ MỞ EMA (Z-TREND) ---
    ema_short = series.ewm(span=window).mean()        # EMA 6h
    ema_long = series.ewm(span=long_window).mean()    # EMA 24h

    raw_trend_diff = float(ema_short.iloc[-1]) - float(ema_long.iloc[-1])
    current_std = max(float(std.iloc[-1]) if pd.notna(std.iloc[-1]) else 1.0, 1e-12)
    
    # Chuẩn hóa Trend: Chênh lệch EMA bằng bao nhiêu lần độ lệch chuẩn?
    # Không dùng 0.2% cứng nhắc nữa, mà dùng Z-Score của Trend
    z_trend = abs(raw_trend_diff) / current_std
    
    direction = 1 if raw_trend_diff >= 0 else -1

    return {
        "is_squeeze": bbw_ratio < 0.85,       # Squeeze khi Vol hiện tại co lại < 85% trung bình ngày
        "is_trending": z_trend > 1.2,         # Trending khi EMA tách nhau xa hơn 1.2 độ lệch chuẩn
        "bbw_ratio": bbw_ratio,
        "direction": direction,
        "z_trend": float(z_trend)
    }


def detect_market_regime_with_er(
    prices: pd.Series,
    lookback: Optional[int] = 72,  # Nâng mặc định lên 6 giờ (72 nến 5m) để chống nhiễu
) -> dict:
    """
    Detect trending/sideways using Efficiency Ratio.
    Returns direction to differentiate Uptrend (+1) vs Downtrend (-1).
    """
    series = _to_price_series(prices)
    
    if len(series) <= lookback:
        return {"type": RegimeType.SIDEWAYS, "strength": 0.0, "direction": 1}

    lookback = max(1, min(int(lookback), len(series) - 1))

    # Tính biến thiên thuần (có âm/dương) để lấy hướng
    raw_change = series.diff(lookback)
    
    # Tính trị tuyệt đối cho công thức ER
    change = raw_change.abs()
    volatility = series.diff().abs().rolling(window=lookback).sum()
    
    er = change / volatility.replace(0.0, pd.NA)
    current_er = float(er.iloc[-1]) if pd.notna(er.iloc[-1]) else 0.0

    # Bắt HƯỚNG của xu hướng
    current_direction = 1 if float(raw_change.iloc[-1]) >= 0 else -1

    if current_er > 0.30:
        regime = RegimeType.TRENDING
    else:
        regime = RegimeType.SIDEWAYS
        
    return {
        "type": regime, 
        "strength": current_er,
        "direction": current_direction  # <-- THÊM CÁI NÀY
    }


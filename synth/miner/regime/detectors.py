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


def detect_market_regime_with_bbw(prices: pd.Series) -> dict:
    """
    Detect squeeze/trend state using Bollinger Band Width + EMA spread.
    """
    series = _to_price_series(prices)
    if len(series) < 2:
        return {
            "is_squeeze": False,
            "is_trending": False,
            "bbw_ratio": 1.0,
        }

    window = 20
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + 2.0 * std
    lower = sma - 2.0 * std
    bbw = (upper - lower) / sma

    current_bbw = float(bbw.iloc[-1]) if pd.notna(bbw.iloc[-1]) else 0.0
    avg_bbw_series = bbw.rolling(50).mean()
    avg_bbw = (
        float(avg_bbw_series.iloc[-1])
        if pd.notna(avg_bbw_series.iloc[-1])
        else 0.0
    )
    if avg_bbw <= 0.0:
        avg_bbw = max(abs(current_bbw), 1e-12)

    ema_short = series.ewm(span=10).mean()
    ema_long = series.ewm(span=30).mean()
    last_price = max(abs(float(series.iloc[-1])), 1e-12)
    trend_diff = abs(float(ema_short.iloc[-1]) - float(ema_long.iloc[-1]))
    trend_diff = trend_diff / last_price

    return {
        "is_squeeze": current_bbw < (avg_bbw * 0.9),
        "is_trending": trend_diff > 0.002,
        "bbw_ratio": float(current_bbw / avg_bbw),
    }


def detect_market_regime_with_er(
    prices: pd.Series,
    lookback: Optional[int] = 20,
) -> dict:
    """
    Detect trending/sideways using Efficiency Ratio.
    """
    series = _to_price_series(prices)
    if len(series) < 2:
        return {"type": RegimeType.SIDEWAYS, "strength": 0.0}

    if not lookback or lookback <= 0:
        lookback = min(20, len(series) - 1)
    lookback = max(1, min(int(lookback), len(series) - 1))

    change = series.diff(lookback).abs()
    volatility = series.diff().abs().rolling(window=lookback).sum()
    er = change / volatility.replace(0.0, pd.NA)
    current_er = float(er.iloc[-1]) if pd.notna(er.iloc[-1]) else 0.0

    if current_er > 0.30:
        regime = RegimeType.TRENDING
    else:
        regime = RegimeType.SIDEWAYS
    return {"type": regime, "strength": current_er}


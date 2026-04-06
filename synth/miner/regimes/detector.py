"""
Regime detection for Strategy Selector (docs/ARCHITECTURE.md).

Maps to REGIME_TYPES in strategies.base:
- crypto: bull / high_vol / ranging  (same *names* for high and low; detection logic differs)
- gold: trending / mean_reverting
- equity: market_open / overnight / earnings / trending (trending from price action + BBW; others from session)

Crypto **high** uses `detect_pattern_v2` (bullish/bearish/neutral → bull/high_vol/ranging).
Crypto **low** uses ER + short-window vol/drawdown on the same label set so `strategies.yaml`
can use one schema under `high` and `low`.
"""

from __future__ import annotations

import math
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


def _detect_equity_regime(start_time: str, history_dict: dict[str, float]) -> RegimeResult:
    """
    Blend session calendar (earnings / RTH / overnight) with price action on 5m history.

    When history has at least 288 bars (~24h of 5m), use BBW + EMA trend (same helper as gold).
    Strong expansion + trending → ``trending`` so YAML can route momentum-aware models.
    """
    try:
        dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
    except Exception:
        return RegimeResult("equity", "market_open", 0.0, None)

    t = dt.time()
    is_weekday = dt.weekday() < 5
    is_earnings_window = is_weekday and dtime(14, 30) <= t < dtime(15, 0)
    is_market_open = is_weekday and dtime(14, 30) <= t < dtime(21, 0)

    try:
        keys = sorted(history_dict.keys(), key=lambda x: int(x))
        if len(keys) > 288:
            tail = keys[-2000:]
            s = pd.Series([float(history_dict[k]) for k in tail], dtype=float)
            info = detect_market_regime_with_bbw(s)
            is_trending = bool(info.get("is_trending"))
            bbw_ratio = float(info.get("bbw_ratio", 1.0))
            z_trend = float(info.get("z_trend", 0.0))
            # Vol expansion (breakout) or very strong EMA separation (e.g. crash / rip)
            strong_price = is_trending and (bbw_ratio > 1.2 or z_trend > 2.2)
            if strong_price:
                conf = min(1.0, max(0.65, 0.55 + 0.1 * min(z_trend, 3.0)))
                return RegimeResult(
                    "equity",
                    "trending",
                    conf,
                    {"bbw": info, "source": "price_action"},
                )
    except Exception:
        pass

    if is_earnings_window:
        return RegimeResult("equity", "earnings", 0.6, None)
    if is_market_open:
        return RegimeResult("equity", "market_open", 0.6, None)
    return RegimeResult("equity", "overnight", 0.6, None)


def _detect_gold_regime(history_dict: dict[str, float]) -> RegimeResult:
    try:
        keys = sorted(history_dict.keys(), key=lambda x: int(x))
        if len(keys) < 50:
            return RegimeResult("gold", "mean_reverting", 0.0, None)
            
        # Kéo 2000 nến (~7 ngày) để lấy bao quát 1 tuần của Vàng thay vì 800 nến
        tail = keys[-2000:]
        s = pd.Series([float(history_dict[k]) for k in tail])
        
        # Đẩy vào hàm BBW mới (Tự động nhận diện 6h và 24h)
        info = detect_market_regime_with_bbw(s)
        
        is_trending = bool(info.get("is_trending"))
        bbw_ratio = float(info.get("bbw_ratio", 1.0))

        regime = "trending" if is_trending else "mean_reverting"
        
        # TÍNH CONFIDENCE CHUẨN XÁC DỰA VÀO HÀNH VI DẢI BOLLINGER
        if regime == "trending":
            # Nếu đang có Trend: Bollinger Band CÀNG NỞ TO (Breakout) -> Càng tự tin
            # bbw_ratio = 1.5 (nở 150%) -> conf = 1.0
            conf = min(1.0, max(0.0, (bbw_ratio - 1.0) / 0.5))
        else:
            # Nếu đang Mean Reverting (Sideway): Bollinger Band CÀNG NÉN CHẶT -> Càng tự tin là Sideway
            # bbw_ratio = 0.5 (nén một nửa) -> conf = 1.0
            conf = min(1.0, max(0.0, (1.0 - bbw_ratio) / 0.5))

        return RegimeResult("gold", regime, float(conf), {"bbw": info})
        
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
    """
    Same regime *labels* as _detect_crypto_high (bull | high_vol | ranging) for YAML alignment;
    logic differs: ER (Direction-Aware) + realized-vol stress, not pattern_v2.
    """
    try:
        keys = sorted(history_dict.keys(), key=lambda x: int(x))
        if len(keys) < 30:
            return RegimeResult("crypto", "ranging", 0.0, None)
            
        tail = keys[-2000:]
        s = pd.Series([float(history_dict[k]) for k in tail], dtype=float).clip(lower=1e-12)
        
        # Ép lookback = 72 (6h) cho Low frequency (predict 24h)
        info = detect_market_regime_with_er(s, lookback=72)
        
        # 1. XỬ LÝ TREND (Bây giờ đã biết là trend Tăng hay Giảm)
        if info.get("type") == RegimeType.TRENDING:
            direction = info.get("direction", 1)
            strength = float(info.get("strength", 0.0))
            
            if direction == 1:
                # Uptrend chuẩn -> Label "bull" -> YAML thường gọi ARIMA hoặc Garch trơn
                return RegimeResult(
                    "crypto",
                    "bull",
                    strength,
                    {"er": info},
                )
            else:
                # Downtrend hoảng loạn -> Label "high_vol" -> YAML sẽ gọi EGARCH hoặc Jump Diffusion
                return RegimeResult(
                    "crypto",
                    "high_vol",
                    strength,
                    {"er": info},
                )

        # 2. NẾU SIDEWAY -> QUÉT CÚ SỐC BIẾN ĐỘNG (VOLATILITY SHOCK)
        logp = pd.Series([math.log(max(float(x), 1e-12)) for x in s.values])
        dlog = logp.diff().dropna()
        if len(dlog) < 80:
            return RegimeResult(
                "crypto",
                "ranging",
                0.3,
                {"er": info},
            )

        long_std = float(dlog.std())
        # Cắt đuôi 120 nến (10 giờ) để tính vol gần nhất
        tail = dlog.iloc[-min(120, len(dlog)) :]
        recent_std = float(tail.std())
        vol_ratio = recent_std / long_std if long_std > 1e-12 else 1.0
        
        # Xét chênh lệch 48 nến (4 giờ)
        short_ret = float(dlog.iloc[-min(48, len(dlog)) :].sum())

        if vol_ratio > 1.35 or short_ret < -0.008:
            conf = min(1.0, max(vol_ratio - 1.0, abs(short_ret) * 50.0))
            return RegimeResult(
                "crypto",
                "high_vol",  # Biến động nổ tung -> vào nhãn high_vol
                conf,
                {"er": info, "vol_ratio": vol_ratio, "short_ret": short_ret},
            )

        # 3. KỊCH BẢN CUỐI: ĐI NGANG NHIỄU (CHOP)
        return RegimeResult(
            "crypto",
            "ranging",
            max(0.0, 1.0 - float(info.get("strength", 0.0))),
            {"er": info, "vol_ratio": vol_ratio},
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
        return _detect_equity_regime(start_time, history_dict)
    if asset_type == "gold":
        return _detect_gold_regime(history_dict)

    is_high = time_length == 3600 or time_increment <= 60
    if is_high:
        return _detect_crypto_high(history_dict)
    return _detect_crypto_low(history_dict)

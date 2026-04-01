"""
dynamic_router_v2 — routes simulation using pattern_detector_v2 (sqrt-time range,
stricter reversal rules). Same routing table API as dynamic_router.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Optional

import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.strategies.pattern_detector_v2 import detect_pattern
from synth.miner.entry import _get_simulate_fn

ROUTER_LOG_DIR = "synth/miner/logs/dynamic_router_v2"
os.makedirs(ROUTER_LOG_DIR, exist_ok=True)

# Same magnitude guard as my_simulation overflow detection
_MAX_ABS_PRICE = 1.0e12
_STRENGTH_CLIP_EPS = 1e-12


def _history_ts_bounds(prices_dict: dict) -> tuple[Optional[str], Optional[str]]:
    if not prices_dict:
        return None, None
    keys = sorted(prices_dict.keys(), key=lambda x: int(x))
    return str(keys[0]), str(keys[-1])


def _router_log(event: str, detail: dict[str, Any]) -> None:
    """Append one JSON line for PM2 / file inspection."""
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(ROUTER_LOG_DIR, f"dynamic_router_v2_{day}.jsonl")
    row = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event": event,
        **detail,
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError as e:
        print(f"[DynamicRouterV2] log write failed: {e}")
    # Mirror critical issues to stdout for pm2 logs
    if event in ("strategy_fallback", "strength_clipped", "output_bad"):
        print(f"[DynamicRouterV2:{event}] {json.dumps(detail, ensure_ascii=False, default=str)[:2000]}")


def _request_snapshot(
    asset: str,
    time_increment: int,
    time_length: int,
    n_sims: int,
    seed: Optional[int],
    prices_dict: dict,
    kwargs: dict,
) -> dict[str, Any]:
    first_ts, last_ts = _history_ts_bounds(prices_dict)
    out: dict[str, Any] = {
        "asset": asset,
        "time_increment": time_increment,
        "time_length": time_length,
        "n_sims": n_sims,
        "seed": seed,
        "history_points": len(prices_dict),
        "history_first_ts": first_ts,
        "history_last_ts": last_ts,
        "miner_start_time": kwargs.get("miner_start_time"),
    }
    return out


def _inspect_paths(out: Any) -> tuple[bool, dict[str, Any]]:
    """Return (ok, stats) using same rules as my_simulation overflow check."""
    try:
        arr = np.asarray(out, dtype=float)
    except Exception as e:
        return False, {"error": f"asarray_failed: {e}"}
    if arr.ndim != 2 or arr.size == 0:
        return False, {"error": "bad_shape", "ndim": getattr(arr, "ndim", None), "size": int(arr.size)}
    finite_mask = np.isfinite(arr)
    has_nonfinite = not np.all(finite_mask)
    finite_vals = arr[finite_mask]
    max_abs = float(np.max(np.abs(finite_vals))) if finite_vals.size > 0 else float("inf")
    min_val = float(np.min(finite_vals)) if finite_vals.size > 0 else float("nan")
    bad = has_nonfinite or min_val <= 0.0 or max_abs > _MAX_ABS_PRICE
    stats = {
        "has_nonfinite": has_nonfinite,
        "min_finite_price": min_val,
        "max_abs_finite_price": max_abs,
        "shape": list(arr.shape),
        "sample_first_path_head": arr[0, : min(10, arr.shape[1])].tolist(),
    }
    return (not bad), stats


class DynamicRouterV2Strategy(BaseStrategy):
    """
    Smart router: 1H price-action bias from 61×1m bars per hour (v2 detector),
    then dispatch to an empirical simulator by asset + bias.
    """

    name = "dynamic_router_v2"
    description = (
        "Routes dynamically using v2 1h pattern detection "
        "(sqrt-time expected range, flash-wick filters)"
    )
    supported_assets = []
    supported_frequencies = ["high", "low"]

    # Routing table for 5 Precog patterns (continuation/reversal/indecision)
    default_routing = {
        "BTC": {
            "bull_continuation": "ensemble_garch_v2_v4", # Cân bằng giữa đẩy mượt và FOMO
            "bear_continuation": "garch_v4",             # Vua quản lý rủi ro xả hàng
            "bull_reversal": "egarch",                   # Bậc thầy xử lý bất đối xứng, quét râu kéo lên
            "bear_reversal": "mean_reversion",           # Vua bắt bài Fakeout, fomo lên rồi xả ngược
            "indecision": "ensemble_weighted"            # Lấy trung bình ý kiến trong vùng nhiễu loạn
        },
        "ETH": {
            "bull_continuation": "garch_v2",
            "bear_continuation": "jump_diffusion",
            "bull_reversal": "regime_switching",
            "bear_reversal": "garch_v4",
            "indecision": "ensemble_garch_v2_v4",
        },
        "SOL": {
            "bull_continuation": "ensemble_garch_v2_v4",
            "bear_continuation": "garch_v4",
            "bull_reversal": "egarch",
            "bear_reversal": "mean_reversion",
            "indecision": "ensemble_weighted"
        },
        "DEFAULT": {
            "bull_continuation": "garch_v4",
            "bear_continuation": "garch_v4",
            "bull_reversal": "garch_v2",
            "bear_reversal": "garch_v2",
            "indecision": "regime_switching",
        },
    }

    def simulate(
        self,
        prices_dict: dict,
        asset: str,
        time_increment: int,
        time_length: int,
        n_sims: int,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> np.ndarray:
        pattern_data = detect_pattern(prices_dict)
        pattern = pattern_data.get("pattern", "indecision")
        strength = float(pattern_data.get("strength", 0.0))
        strength_raw = float(pattern_data.get("strength_raw", strength))
        exp_rng = float(pattern_data.get("expected_range", 0.0))
        candles = pattern_data.get("candles")

        # Back-compat signals (3-state)
        bias = pattern_data.get("bias", "neutral")
        score = float(pattern_data.get("bias_score", 0.0))

        base_req = _request_snapshot(
            asset, time_increment, time_length, n_sims, seed, prices_dict, kwargs
        )

        if strength_raw > 1.0 + _STRENGTH_CLIP_EPS:
            _router_log(
                "strength_clipped",
                {
                    **base_req,
                    "pattern": pattern,
                    "strength_clamped": strength,
                    "strength_raw": strength_raw,
                },
            )

        routing = self.default_routing.get(asset, self.default_routing["DEFAULT"])
        selected_strategy_name = routing.get(pattern, routing.get("indecision"))

        print(
            f"[DynamicRouterV2] Asset: {asset} | Pattern: {str(pattern).upper()} "
            f"(strength={strength:.3f}, bias={str(bias).upper()}, score={score:.3f}, expected_range≈{exp_rng:.6f}) "
            f"-> {selected_strategy_name}"
        )
        if candles:
            c2 = candles.get("c2", {})
            c1 = candles.get("c1", {})
            c0 = candles.get("c0", {})
            def _fmt(c: dict) -> str:
                return (
                    f"O={c.get('open'):.2f} H={c.get('high'):.2f} "
                    f"L={c.get('low'):.2f} C={c.get('close'):.2f} "
                    f"dir={c.get('direction')} body={c.get('body'):.2f} "
                    f"uw={c.get('upper_wick'):.2f} lw={c.get('lower_wick'):.2f}"
                )
            print(f"  candles: c2[{_fmt(c2)}] c1[{_fmt(c1)}] c0[{_fmt(c0)}]")

        sim_fn = _get_simulate_fn(selected_strategy_name)
        resolved_strategy = selected_strategy_name
        if sim_fn is None:
            print(
                "[DynamicRouterV2 ERROR] Strategy not found; falling back to garch_v2."
            )
            _router_log(
                "strategy_fallback",
                {
                    **base_req,
                    "pattern": pattern,
                    "requested_strategy": selected_strategy_name,
                    "fallback_strategy": "garch_v2",
                    "reason": "simulate_fn_not_found",
                },
            )
            sim_fn = _get_simulate_fn("garch_v2")
            resolved_strategy = "garch_v2"

        # Pass pattern context downstream (safe: simulate_fns accept **kwargs)
        kwargs.setdefault("pattern", pattern)
        kwargs.setdefault("pattern_strength", float(strength))
        kwargs.setdefault("pattern_expected_range", float(exp_rng))
        kwargs.setdefault("pattern_strength_raw", float(strength_raw))
        # Back-compat kwargs
        kwargs.setdefault("pattern_bias", bias)
        kwargs.setdefault("pattern_bias_score", float(score))

        out = sim_fn(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **kwargs,
        )

        ok, stats = _inspect_paths(out)
        if not ok:
            _router_log(
                "output_bad",
                {
                    **base_req,
                    "pattern": pattern,
                    "routed_strategy": selected_strategy_name,
                    "resolved_sub_strategy": resolved_strategy,
                    **stats,
                },
            )

        return out


strategy = DynamicRouterV2Strategy()

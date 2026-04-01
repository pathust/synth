"""
simulations_new_v5.py

Dynamic regime-aware simulations module.

- For HIGH (1h horizon): detect regime from recent prices and choose strategy mix.
- For LOW (24h horizon): use static all-weather ensemble.
"""

from __future__ import annotations

import zlib
from typing import Callable, Optional

import numpy as np

from synth.simulation_input import SimulationInput
from synth.miner.my_simulation import (
    fetch_price_data,
    iso_to_timestamp,
    simulate_crypto_price_paths,
)
from synth.validator.response_validation_v2 import validate_responses
from synth.utils.helpers import convert_prices_to_time_format

# Core simulators
from synth.miner.core.garch_simulator import simulate_single_price_path_with_garch as sim_garch_v1
from synth.miner.core.grach_simulator_v2 import simulate_single_price_path_with_garch as sim_garch_v2
from synth.miner.core.grach_simulator_v2_1 import simulate_single_price_path_with_garch as sim_garch_v2_1
from synth.miner.core.garch_simulator_v2_2 import simulate_single_price_path_with_garch as sim_garch_v2_2
from synth.miner.core.HAR_RV_simulatior import simulate_single_price_path_with_har_garch as sim_har_rv
from synth.miner.core.stock_simulator import simulate_seasonal_stock as sim_seasonal_stock
from synth.miner.core.stock_simulator_v2 import simulate_weekly_seasonal_optimized as sim_weekly_stock

# Strategies
from synth.miner.strategies.grach_simulator_v4 import simulate_single_price_path_with_garch as sim_garch_v4
from synth.miner.strategies.grach_simulator_v4_1 import simulate_single_price_path_with_garch as sim_garch_v4_1
from synth.miner.strategies.grach_simulator_v4_2 import simulate_single_price_path_with_garch as sim_garch_v4_2


def _get_strategy_simulators() -> dict[str, Callable]:
    from synth.miner.strategies.registry import StrategyRegistry

    registry = StrategyRegistry()
    registry.auto_discover()

    out = {}
    for name, strategy in registry.get_all().items():
        def _runner(s):
            def fn(prices_dict, asset, time_increment, time_length, n_sims, seed=42, **kwargs):
                return s.simulate(
                    prices_dict, asset, time_increment, time_length, n_sims, seed=seed, **kwargs
                )
            return fn

        out[name] = _runner(strategy)
    return out


SIMULATOR_FUNCTIONS: dict[str, Callable] = {
    "garch_v1": sim_garch_v1,
    "garch_v2": sim_garch_v2,
    "garch_v2_1": sim_garch_v2_1,
    "garch_v2_2": sim_garch_v2_2,
    "garch_v4": sim_garch_v4,
    "garch_v4_1": sim_garch_v4_1,
    "garch_v4_2": sim_garch_v4_2,
    "har_rv": sim_har_rv,
    "seasonal_stock": sim_seasonal_stock,
    "weekly_stock": sim_weekly_stock,
}

_simulator_functions_cache: Optional[dict[str, Callable]] = None
_ENSEMBLE_TOP_N = 3


def _build_simulator_functions() -> dict[str, Callable]:
    global _simulator_functions_cache
    if _simulator_functions_cache is not None:
        return _simulator_functions_cache
    out = dict(SIMULATOR_FUNCTIONS)
    for name, fn in _get_strategy_simulators().items():
        if name not in out:
            out[name] = fn
        if f"{name}_strat" not in out:
            out[f"{name}_strat"] = fn
    _simulator_functions_cache = out
    return out


def _get_simulate_fn(simulator_name: str) -> Optional[Callable]:
    funcs = _build_simulator_functions()
    if simulator_name in funcs:
        return funcs[simulator_name]
    base = simulator_name.replace("_strat", "")
    return funcs.get(base)


def detect_market_regime(past_prices: list[float]) -> str:
    """
    Analyze recent prices (typically 60-120 last 1m bars) and detect market regime.
    Returns: "shock" | "uptrend" | "downtrend" | "sideways"
    """
    if not past_prices or len(past_prices) < 2:
        return "sideways"

    prices = np.array(past_prices, dtype=float)
    returns = np.diff(prices) / prices[:-1]

    # Keep sign to distinguish uptrend vs downtrend.
    total_return = (prices[-1] - prices[0]) / prices[0]
    volatility = np.std(returns)

    # Tuned thresholds for 1m bars (high horizon).
    vol_shock_threshold = 0.0035
    trend_threshold = 0.0025

    if volatility > vol_shock_threshold:
        return "shock"
    if total_return > trend_threshold:
        return "uptrend"
    if total_return < -trend_threshold:
        return "downtrend"
    return "sideways"


def get_dynamic_strategy_list(asset: str, time_length: int, past_prices: list[float]) -> list[tuple[str, float]]:
    """
    Regime-driven strategy allocation.
    """
    asset_upper = (asset or "").upper()

    if time_length == 3600:
        regime = detect_market_regime(past_prices)
        print(f"[REGIME DETECTOR] Tin hieu {asset} (HIGH): {regime.upper()}")

        # BTC HIGH
        if asset_upper == "BTC":
            if regime == "uptrend":
                return [("garch_v4_1", 1.0)]
            if regime == "downtrend":
                return [("garch_v4_2", 1.0)]
            if regime == "shock":
                return [("jump_diffusion", 1.0)]
            return [("arima_equity", 0.5), ("regime_switching", 0.5)]

        # ETH HIGH
        if asset_upper == "ETH":
            if regime == "uptrend":
                return [("garch_v4", 1.0)]
            if regime == "downtrend":
                return [("garch_v4_2", 0.7), ("garch_v4", 0.3)]
            if regime == "shock":
                return [("jump_diffusion", 0.6), ("garch_v4", 0.4)]
            return [("ensemble_weighted", 0.6), ("garch_v2", 0.4)]

        # SOL HIGH
        if asset_upper == "SOL":
            if regime == "uptrend":
                return [("ensemble_weighted", 0.7), ("garch_v4_1", 0.3)]
            if regime == "downtrend":
                return [("ensemble_weighted", 0.7), ("garch_v4_2", 0.3)]
            if regime == "shock":
                return [("jump_diffusion", 0.6), ("ensemble_weighted", 0.4)]
            return [("arima_equity", 0.6), ("ensemble_weighted", 0.4)]

        # XAU HIGH
        if asset_upper == "XAU":
            if regime in ("uptrend", "downtrend"):
                return [("jump_diffusion", 0.6), ("garch_v4_1", 0.4)]
            if regime == "shock":
                return [("jump_diffusion", 1.0)]
            return [("regime_switching", 0.6), ("jump_diffusion", 0.4)]

        # default HIGH
        return [("ensemble_weighted", 1.0)]

    # LOW uses static allocations from simulations_new_v3 (no regime detection).
    if asset_upper == "BTC":
        return [("garch_v4", 0.4), ("garch_v2_2", 0.3), ("gjr_garch", 0.3)]
    if asset_upper == "ETH":
        return [("garch_v4", 0.4), ("garch_v2_2", 0.3), ("regime_switching", 0.3)]
    if asset_upper == "XAU":
        return [("jump_diffusion", 0.4), ("garch_v4_1", 0.3), ("weekly_regime_switching", 0.3)]
    if asset_upper == "SOL":
        return [("garch_v2_2", 0.4), ("garch_v4", 0.3), ("regime_switching", 0.3)]
    if asset_upper == "NVDAX":
        return [("arima_equity", 0.4), ("garch_v4_1", 0.3), ("markov_garch_jump", 0.3)]
    if asset_upper == "TSLAX":
        return [("weekly_garch_v4", 0.4), ("garch_v4", 0.3), ("regime_switching", 0.3)]
    if asset_upper == "AAPLX":
        return [("markov_garch_jump", 0.4), ("regime_switching", 0.3), ("garch_v4_1", 0.3)]
    if asset_upper == "GOOGLX":
        return [("gjr_garch", 0.4), ("regime_switching", 0.3), ("garch_v4_1", 0.3)]
    if asset_upper == "SPYX":
        return [("weekly_regime_switching", 0.4), ("garch_v4", 0.3), ("arima_equity", 0.3)]
    return [("garch_v2_2", 0.4), ("garch_v4", 0.3), ("regime_switching", 0.3)]


def _make_sub_seed(base_seed: int, strategy_name: str) -> int:
    h = zlib.crc32(strategy_name.encode("utf-8")) & 0xFFFFFFFF
    return (base_seed + (h % 100_000)) & 0x7FFF_FFFF


def _run_ensemble(
    strategy_list: list[tuple[str, float]],
    asset: str,
    start_time: str,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    seed: int,
) -> Optional[np.ndarray]:
    n_strategies = min(len(strategy_list), _ENSEMBLE_TOP_N)
    active_strategies = strategy_list[:n_strategies]
    if not active_strategies:
        return None

    total_weight = sum(w for _, w in active_strategies) or 1.0
    normalized_strats = [(name, w / total_weight) for name, w in active_strategies]

    all_paths: list[np.ndarray] = []
    used_strategies: list[str] = []

    target_total_sims = int(num_simulations * 1.10)
    sims_collected = 0

    for i, (sim_name, weight) in enumerate(normalized_strats):
        fn = _get_simulate_fn(sim_name)
        if fn is None:
            continue

        if i == len(normalized_strats) - 1:
            n_sub = target_total_sims - sims_collected
        else:
            n_sub = int(target_total_sims * weight)
        if n_sub <= 0:
            continue

        sub_seed = _make_sub_seed(seed, sim_name)
        try:
            paths = simulate_crypto_price_paths(
                current_price=None,
                asset=asset,
                start_time=start_time,
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=n_sub,
                simulate_fn=fn,
                max_data_points=None,
                seed=sub_seed,
            )
            if paths is not None and isinstance(paths, np.ndarray) and paths.ndim == 2 and paths.shape[0] > 0:
                all_paths.append(paths)
                sims_collected += paths.shape[0]
                used_strategies.append(f"{sim_name}({paths.shape[0]})")
        except Exception as e:
            print(f"[ENSEMBLE] {sim_name} failed: {e}")

    if not all_paths:
        return None

    combined = np.vstack(all_paths)
    start_prices = combined[:, 0]
    final_prices = combined[:, -1]
    safe_start_prices = np.where(start_prices == 0, 1e-8, start_prices)
    returns = (final_prices - safe_start_prices) / safe_start_prices
    lower_bound = np.percentile(returns, 1.0)
    upper_bound = np.percentile(returns, 99.0)
    valid_indices = np.where((returns >= lower_bound) & (returns <= upper_bound))[0]
    trimmed_combined = combined[valid_indices]

    rng = np.random.RandomState(seed)
    if trimmed_combined.shape[0] >= num_simulations:
        selected_indices = rng.choice(trimmed_combined.shape[0], size=num_simulations, replace=False)
        final_ensemble = trimmed_combined[selected_indices]
    elif combined.shape[0] >= num_simulations:
        selected_indices = rng.choice(combined.shape[0], size=num_simulations, replace=False)
        final_ensemble = combined[selected_indices]
    else:
        selected_indices = rng.choice(combined.shape[0], size=num_simulations, replace=True)
        final_ensemble = combined[selected_indices]

    print(f"[ENSEMBLE] {asset}: {' + '.join(used_strategies)} => Selected {final_ensemble.shape[0]} paths")
    return final_ensemble


def _get_recent_prices(asset: str, time_increment: int, start_time: str, n_points: int = 120) -> list[float]:
    """
    Load up to n_points prices before start_time from local DB cache.
    """
    hist = fetch_price_data(asset, time_increment, only_load=True)
    tf = "1m" if time_increment == 60 else ("5m" if time_increment == 300 else str(time_increment))
    if not hist or tf not in hist or not hist[tf]:
        return []

    start_ts = iso_to_timestamp(start_time)
    filtered = [(int(k), float(v)) for k, v in hist[tf].items() if int(k) < start_ts]
    if not filtered:
        return []
    filtered.sort(key=lambda x: x[0])
    return [p for _, p in filtered[-n_points:]]


DEFAULT_FALLBACK_CHAIN = [
    "garch_v2_2",
    "garch_v2",
    "garch_v4",
    "garch_v4_1",
    "garch_v4_2",
    "jump_diffusion",
    "regime_switching",
    "garch_v2_1",
    "garch_v1",
    "har_rv",
]


def generate_simulations(
    simulation_input: SimulationInput,
    asset: str = "BTC",
    start_time: str = "",
    time_increment: int = 300,
    time_length: int = 86400,
    num_simulations: int = 1,
    seed: int = 42,
    version: str | None = None,
) -> dict:
    if start_time == "":
        raise ValueError("Start time must be provided.")

    past_prices = _get_recent_prices(asset, time_increment, start_time, n_points=120)
    strategy_list = get_dynamic_strategy_list(asset, time_length, past_prices)
    generated_paths = _run_ensemble(
        strategy_list, asset, start_time, time_increment,
        time_length, num_simulations, seed,
    )

    if generated_paths is not None and isinstance(generated_paths, np.ndarray) and generated_paths.shape[0] > 0:
        predictions = convert_prices_to_time_format(
            generated_paths.tolist(), start_time, time_increment
        )
        fmt = validate_responses(predictions, simulation_input, "0")
        if fmt == "CORRECT":
            print("[INFO] simulations_new_v5: generation SUCCESS")
            return {"predictions": predictions}
        print(f"[WARN] simulations_new_v5: generation failed validation ({fmt})")

    print("[WARN] simulations_new_v5: falling back to sequential mode")
    try_names = [name for name, _ in strategy_list]
    for fb in DEFAULT_FALLBACK_CHAIN:
        if fb not in try_names:
            try_names.append(fb)

    for sim_name in try_names:
        fn = _get_simulate_fn(sim_name)
        if fn is None:
            continue
        try:
            simulations = simulate_crypto_price_paths(
                current_price=None,
                asset=asset,
                start_time=start_time,
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=num_simulations,
                simulate_fn=fn,
                max_data_points=None,
                seed=seed,
            )
            if simulations is None:
                continue
            predictions = convert_prices_to_time_format(
                simulations.tolist(), start_time, time_increment
            )
            format_validation = validate_responses(predictions, simulation_input, "0")
            if format_validation == "CORRECT":
                print(f"[INFO] simulations_new_v5: fallback used simulator={sim_name}")
                return {"predictions": predictions}
        except Exception as e:
            print(f"[WARN] simulations_new_v5: fallback {sim_name} failed: {e}")

    return {"predictions": None}


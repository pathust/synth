"""
entry_simulations_new.py

Entry-point thay thế cho `synth/miner/entry.py` nhưng dùng logic kiểu
`simulations_new_v3.py` (ensemble + sequential fallback).

Legacy note: regime + strategies.yaml chọn model; mapping dưới đây chỉ dùng khi gọi legacy path.

Để dùng trong miner:
    from synth.miner.entry_simulations_new import generate_simulations
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from synth.simulation_input import SimulationInput
from synth.miner.my_simulation import simulate_crypto_price_paths
from synth.utils.helpers import convert_prices_to_time_format
from synth.validator.response_validation_v2 import validate_responses


# ---------------------------------------------------------------------------
# Strategy list per (asset, prompt_type) with WEIGHTS
# NOTE: Fallback list when not using entry.generate_simulations + strategies.yaml.
# ---------------------------------------------------------------------------
STRATEGY_LIST_FOR_ASSET: dict[tuple[str, str], list[tuple[str, float]]] = {
    # HIGH
    ("BTC", "high"): [("weekly_garch_v4", 1.0)],
    ("ETH", "high"): [("gjr_garch", 1.0)],
    ("XAU", "high"): [("garch_v4_2", 1.0)],
    ("SOL", "high"): [("gjr_garch", 1.0)],

    # LOW
    ("BTC", "low"): [("garch_v4", 0.4), ("garch_v2_2", 0.3), ("gjr_garch", 0.3)],
    ("ETH", "low"): [("garch_v4", 0.4), ("garch_v2_2", 0.3), ("regime_switching", 0.3)],
    ("XAU", "low"): [("jump_diffusion", 0.4), ("garch_v4_1", 0.3), ("weekly_regime_switching", 0.3)],
    ("SOL", "low"): [("garch_v4", 0.4), ("arima_equity", 0.3), ("mean_reversion", 0.3)],

    # Stocks (low only)
    ("NVDAX", "low"): [("arima_equity", 0.4), ("garch_v4_1", 0.3), ("markov_garch_jump", 0.3)],
    ("TSLAX", "low"): [("weekly_garch_v4", 0.4), ("garch_v4", 0.3), ("regime_switching", 0.3)],
    ("AAPLX", "low"): [("markov_garch_jump", 0.4), ("regime_switching", 0.3), ("garch_v4_1", 0.3)],
    ("GOOGLX", "low"): [("gjr_garch", 0.4), ("regime_switching", 0.3), ("garch_v4_1", 0.3)],
    ("SPYX", "low"): [("weekly_regime_switching", 0.4), ("garch_v4", 0.3), ("arima_equity", 0.3)],
}


DEFAULT_FALLBACK_CHAIN: list[str] = [
    "garch_v2",
    "garch_v4",
    "garch_v4_1",
    "garch_v4_2",
    "egarch",
    "garch_v2_1",
    "seasonal_stock",
    "garch_v1",
    "har_rv",
]

_ENSEMBLE_TOP_N = 3


def _get_prompt_type(time_length: int) -> str:
    return "high" if time_length == 3600 else "low"


def _get_strategy_list_for_asset(asset: str, time_length: int) -> list[tuple[str, float]]:
    prompt = _get_prompt_type(time_length)
    key = (asset, prompt)
    if key in STRATEGY_LIST_FOR_ASSET:
        return list(STRATEGY_LIST_FOR_ASSET[key])
    key_low = (asset, "low")
    if key_low in STRATEGY_LIST_FOR_ASSET:
        return list(STRATEGY_LIST_FOR_ASSET[key_low])
    return [("garch_v2", 0.5), ("garch_v4", 0.3), ("garch_v4_1", 0.2)]


def _get_simulate_fn(simulator_name: str) -> Optional[Callable]:
    # Reuse entry.py's simulator registry (core + strategy registry)
    from synth.miner.entry import _get_simulate_fn as _entry_get_simulate_fn

    return _entry_get_simulate_fn(simulator_name)


def _make_sub_seed(base_seed: int, strategy_name: str) -> int:
    return (base_seed + hash(strategy_name) % 100_000) & 0x7FFF_FFFF


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
    active = strategy_list[:n_strategies]

    total_weight = sum(w for _, w in active) or 1.0
    normalized = [(name, w / total_weight) for name, w in active]

    all_paths: list[np.ndarray] = []
    sims_collected = 0
    target_total_sims = int(num_simulations * 1.10)  # over-request 10%

    for i, (sim_name, weight) in enumerate(normalized):
        fn = _get_simulate_fn(sim_name)
        if fn is None:
            continue

        if i == len(normalized) - 1:
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
                miner_start_time=start_time,
            )
            if paths is not None and isinstance(paths, np.ndarray) and paths.ndim == 2 and paths.shape[0] > 0:
                all_paths.append(paths)
                sims_collected += int(paths.shape[0])
        except Exception:
            continue

    if not all_paths:
        return None

    combined = np.vstack(all_paths)

    # Outlier trimming on total return (1%..99%)
    start_prices = combined[:, 0]
    final_prices = combined[:, -1]
    safe_start = np.where(start_prices == 0, 1e-8, start_prices)
    returns = (final_prices - safe_start) / safe_start
    lower = np.percentile(returns, 1.0)
    upper = np.percentile(returns, 99.0)
    valid_idx = np.where((returns >= lower) & (returns <= upper))[0]
    trimmed = combined[valid_idx] if valid_idx.size > 0 else combined

    rng = np.random.RandomState(seed)
    if trimmed.shape[0] >= num_simulations:
        selected = rng.choice(trimmed.shape[0], size=num_simulations, replace=False)
        return trimmed[selected]
    if combined.shape[0] >= num_simulations:
        selected = rng.choice(combined.shape[0], size=num_simulations, replace=False)
        return combined[selected]
    selected = rng.choice(combined.shape[0], size=num_simulations, replace=True)
    return combined[selected]


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
    import warnings
    from synth.miner.entry import generate_simulations as _generate_simulations

    warnings.warn(
        "synth.miner.entry_new.generate_simulations is deprecated; use synth.miner.entry.generate_simulations",
        DeprecationWarning,
        stacklevel=2,
    )
    return _generate_simulations(
        simulation_input=simulation_input,
        asset=asset,
        start_time=start_time,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
        seed=seed,
        version=version,
    )

    if start_time == "":
        raise ValueError("Start time must be provided.")

    strategy_list = _get_strategy_list_for_asset(asset, time_length)
    ensemble_names = [name for name, _ in strategy_list[:_ENSEMBLE_TOP_N]]
    print(
        f"[INFO] entry_simulations_new: asset={asset}, time_length={time_length}, "
        f"ensemble={ensemble_names}, n={num_simulations}, seed={seed}"
    )

    # Phase 1: ensemble
    ensemble_paths = _run_ensemble(
        strategy_list,
        asset,
        start_time,
        time_increment,
        time_length,
        num_simulations,
        seed,
    )
    if ensemble_paths is not None:
        predictions = convert_prices_to_time_format(
            ensemble_paths.tolist(), start_time, time_increment
        )
        fmt = validate_responses(predictions, simulation_input, "0")
        if fmt == "CORRECT":
            return {"predictions": predictions}
        print(f"[WARN] entry_simulations_new: ensemble failed validation ({fmt})")

    # Phase 2: sequential fallback
    print("[WARN] entry_simulations_new: falling back to sequential mode")
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
                miner_start_time=start_time,
            )
            if simulations is None:
                continue

            predictions = convert_prices_to_time_format(
                simulations.tolist(), start_time, time_increment
            )
            format_validation = validate_responses(
                predictions, simulation_input, "0",
            )
            if format_validation == "CORRECT":
                print(f"[INFO] entry_simulations_new: fallback used simulator={sim_name}")
                return {"predictions": predictions}
        except Exception as e:
            print(f"[WARN] entry_simulations_new: {sim_name} failed: {e}")

    return {"predictions": None}

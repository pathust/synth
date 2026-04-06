"""
simulations_new.py

Ensemble-based simulation: for each (asset, prompt_type), run top N
strategies in parallel with different seeds, then combine all paths
into a single diverse ensemble. Falls back to sequential single-strategy
mode if ensemble fails.

Ensemble mode improves CRPS by:
  - Increasing path diversity across different model families
  - Reducing dependence on any single strategy being correct
  - Using independent seeds per strategy for uncorrelated innovations
"""

import numpy as np
from typing import Callable, Optional

from synth.simulation_input import SimulationInput
from synth.miner.price_simulation import get_asset_price

# Core simulators
from synth.miner.core.garch_simulator import simulate_single_price_path_with_garch as sim_garch_v1
from synth.miner.core.grach_simulator_v2 import simulate_single_price_path_with_garch as sim_garch_v2
from synth.miner.core.grach_simulator_v2_1 import simulate_single_price_path_with_garch as sim_garch_v2_1
from synth.miner.core.HAR_RV_simulatior import simulate_single_price_path_with_har_garch as sim_har_rv
from synth.miner.core.stock_simulator import simulate_seasonal_stock as sim_seasonal_stock
from synth.miner.core.stock_simulator_v2 import simulate_weekly_seasonal_optimized as sim_weekly_stock

# Strategies (v4 family)
from synth.miner.core.grach_simulator_v4 import simulate_single_price_path_with_garch as sim_garch_v4
from synth.miner.core.grach_simulator_v4_1 import simulate_single_price_path_with_garch as sim_garch_v4_1
from synth.miner.core.grach_simulator_v4_2 import simulate_single_price_path_with_garch as sim_garch_v4_2

from synth.miner.my_simulation import simulate_crypto_price_paths
from synth.validator.response_validation_v2 import validate_responses
from synth.utils.helpers import convert_prices_to_time_format


def _get_strategy_simulators() -> dict[str, Callable]:
    """Load simulate_fn from strategy registry (egarch, mean_reversion, etc.)."""
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
    "garch_v4": sim_garch_v4,
    "garch_v4_1": sim_garch_v4_1,
    "garch_v4_2": sim_garch_v4_2,
    "har_rv": sim_har_rv,
    "seasonal_stock": sim_seasonal_stock,
    "weekly_stock": sim_weekly_stock,
}


_simulator_functions_cache: Optional[dict[str, Callable]] = None


def _build_simulator_functions() -> dict[str, Callable]:
    """Merge core + strategy registry (cached)."""
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


# ---------------------------------------------------------------------------
# Strategy lists per (asset, prompt_type).
#
# HIGH frequency: garch_v2 (proven competitive from old sim) is always
# included alongside complementary model families for maximum diversity.
#
# LOW frequency: keep strategies that already outperform old sim by ~2:1.
# ---------------------------------------------------------------------------
STRATEGY_LIST_FOR_ASSET: dict[tuple[str, str], list[str]] = {
    # HIGH: garch_v2 + 2 complementary families
    ("BTC", "high"):   ["garch_v2", "garch_v4", "egarch"],
    ("ETH", "high"):   ["garch_v2", "garch_v4", "garch_v4_1"],
    ("XAU", "high"):   ["garch_v2", "egarch", "garch_v4_2"],
    ("SOL", "high"):   ["garch_v2", "garch_v4", "egarch"],

    # LOW: proven winners from backtest (new already beats old ~2:1)
    ("BTC", "low"):    ["egarch", "garch_v4_2", "garch_v2"],
    ("ETH", "low"):    ["seasonal_stock", "egarch", "garch_v4_2"],
    ("XAU", "low"):    ["egarch", "garch_v4_1", "garch_v4"],
    ("SOL", "low"):    ["seasonal_stock", "garch_v4_2", "egarch"],

    # Stocks (low only)
    ("NVDAX", "low"):  ["garch_v4_1", "garch_v4", "garch_v2"],
    ("TSLAX", "low"):  ["garch_v4_1", "garch_v4", "garch_v2"],
    ("AAPLX", "low"):  ["garch_v4", "garch_v2_1", "garch_v2"],
    ("GOOGLX", "low"): ["garch_v4", "garch_v4_1", "garch_v2"],
    ("SPYX", "low"):   ["garch_v4", "garch_v4_2", "garch_v2"],
}

DEFAULT_FALLBACK_CHAIN = [
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


def _get_strategy_list_for_asset(asset: str, time_length: int) -> list[str]:
    prompt = _get_prompt_type(time_length)
    key = (asset, prompt)
    if key in STRATEGY_LIST_FOR_ASSET:
        return list(STRATEGY_LIST_FOR_ASSET[key])
    key_low = (asset, "low")
    if key_low in STRATEGY_LIST_FOR_ASSET:
        return list(STRATEGY_LIST_FOR_ASSET[key_low])
    return ["garch_v2", "garch_v4", "garch_v4_1"]


def _get_simulate_fn(simulator_name: str) -> Optional[Callable]:
    funcs = _build_simulator_functions()
    if simulator_name in funcs:
        return funcs[simulator_name]
    base = simulator_name.replace("_strat", "")
    return funcs.get(base)


# ---------------------------------------------------------------------------
# Ensemble generation: run top N strategies, combine paths
# ---------------------------------------------------------------------------

def _make_sub_seed(base_seed: int, strategy_name: str) -> int:
    """Deterministic but unique seed per strategy for independent innovations."""
    return (base_seed + hash(strategy_name) % 100_000) & 0x7FFF_FFFF


def _run_ensemble(
    strategy_list: list[str],
    asset: str,
    start_time: str,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    seed: int,
) -> Optional[np.ndarray]:
    """
    Run top N strategies from strategy_list with independent seeds,
    allocate sims proportionally, and combine all paths.

    Returns ndarray (num_simulations, steps+1) or None if all failed.
    """
    n_strategies = min(len(strategy_list), _ENSEMBLE_TOP_N)
    sims_per = num_simulations // n_strategies

    all_paths: list[np.ndarray] = []
    sims_collected = 0
    used_strategies: list[str] = []

    for i, sim_name in enumerate(strategy_list[:n_strategies]):
        fn = _get_simulate_fn(sim_name)
        if fn is None:
            continue

        n_sub = (num_simulations - sims_collected) if i == n_strategies - 1 else sims_per
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

    # Adjust to exact num_simulations
    if combined.shape[0] < num_simulations:
        rng = np.random.RandomState(seed)
        extra = rng.choice(combined.shape[0], size=num_simulations - combined.shape[0], replace=True)
        combined = np.vstack([combined, combined[extra]])
    elif combined.shape[0] > num_simulations:
        combined = combined[:num_simulations]

    print(f"[ENSEMBLE] {asset}: {' + '.join(used_strategies)} = {combined.shape[0]} paths")
    return combined


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
    """
    Generate simulated price paths using ensemble of best strategies.

    Phase 1 (ensemble): run top N strategies with independent seeds,
    combine paths into a diverse ensemble. This directly improves CRPS
    by increasing distribution coverage.

    Phase 2 (fallback): if ensemble fails validation, fall back to
    sequential single-strategy mode with full fallback chain.
    """
    if start_time == "":
        raise ValueError("Start time must be provided.")

    strategy_list = _get_strategy_list_for_asset(asset, time_length)

    print(
        f"[INFO] simulations_new: asset={asset}, time_length={time_length}, "
        f"ensemble={strategy_list[:_ENSEMBLE_TOP_N]}, n={num_simulations}, seed={seed}"
    )

    # ── Phase 1: Ensemble mode ──
    ensemble_paths = _run_ensemble(
        strategy_list, asset, start_time, time_increment,
        time_length, num_simulations, seed,
    )

    if ensemble_paths is not None:
        predictions = convert_prices_to_time_format(
            ensemble_paths.tolist(), start_time, time_increment
        )
        fmt = validate_responses(predictions, simulation_input, "0")
        if fmt == "CORRECT":
            print(f"[INFO] simulations_new: ensemble SUCCESS")
            return {"predictions": predictions}
        print(f"[WARN] simulations_new: ensemble failed validation ({fmt})")

    # ── Phase 2: Sequential fallback ──
    print(f"[WARN] simulations_new: falling back to sequential mode")

    try_names = list(strategy_list)
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

            format_validation = validate_responses(
                predictions,
                simulation_input,
                "0",
            )
            if format_validation == "CORRECT":
                print(f"[INFO] simulations_new: fallback used simulator={sim_name}")
                return {"predictions": predictions}
        except Exception as e:
            print(f"[WARN] simulations_new: {sim_name} failed: {e}")

    return {"predictions": None}

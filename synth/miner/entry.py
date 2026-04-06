"""
entry.py — New unified entry point for simulation generation.

Implements docs/ARCHITECTURE.md live path:
Validator → horizon (HFT/LFT via time_length) → Regime Detector → Strategy
Selector (strategies.yaml) → Monte Carlo → validate/format.

Routing uses synth/miner/config/strategies.yaml (regime-aware keys under each
asset and high|low). Fallback: asset_strategy_config.get_strategy_list.

Usage (in neurons/miner.py):
    from synth.miner.entry import generate_simulations
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from synth.simulation_input import SimulationInput
from synth.miner.config.strategy_store import get_strategy_store
from synth.miner.config.defaults import ENSEMBLE_TOP_N
from synth.miner.strategies.base import StrategyConfig
from synth.miner.ensemble.builder import EnsembleBuilder
from synth.miner.ensemble.trimmer import OutlierTrimmer

from synth.miner.my_simulation import simulate_crypto_price_paths
from synth.utils.helpers import convert_prices_to_time_format
from synth.validator.response_validation_v2 import validate_responses

from synth.miner.data.dataloader import UnifiedDataLoader
from synth.miner.regimes.detector import detect_regime
from synth.miner.constants import HISTORY_WINDOW_DAYS


# ── Simulator function registry (same as simulations_new_v3.py) ─────

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


def _get_simulate_fn(simulator_name: str) -> Optional[Callable]:
    """Retrieve a simulate function by name."""
    funcs = _build_simulator_functions()
    if simulator_name in funcs:
        return funcs[simulator_name]
    base = simulator_name.replace("_strat", "")
    return funcs.get(base)


# ── Ensemble builder (singleton) ────────────────────────────────────

_ensemble_builder: Optional[EnsembleBuilder] = None


def _get_ensemble_builder() -> EnsembleBuilder:
    global _ensemble_builder
    if _ensemble_builder is None:
        _ensemble_builder = EnsembleBuilder(
            top_n=ENSEMBLE_TOP_N,
            trimmer=OutlierTrimmer(),
        )
    return _ensemble_builder


# ── Main entry point ────────────────────────────────────────────────

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

    Drop-in replacement for simulations_new_v3.generate_simulations().
    Reads strategy configs from config/asset_strategy_config.py.

    Phase 1 (ensemble): run top N strategies with independent seeds,
    combine paths into a diverse ensemble.

    Phase 2 (fallback): if ensemble fails validation, fall back to
    sequential single-strategy mode with full fallback chain.
    """
    if start_time == "":
        raise ValueError("Start time must be provided.")

    # ARCHITECTURE.md: horizon (HFT/LFT) is implicit in time_length/time_increment;
    # Regime Detector + Strategy Selector (strategies.yaml) pick models per regime.
    market_regime: Optional[str] = None
    regime_asset_type: Optional[str] = None
    regime_confidence: Optional[float] = None
    try:
        loader = UnifiedDataLoader()
        freq_label = "high" if time_length == 3600 else "low"
        # Already hits MySQL via DataHandler.load_price_data (5m or 1m→5m aggregate).
        # Empty dict = no candles in [start_time - window_days, start_time) or missing table.
        hist = loader.get_historical_dict(
            asset, start_time, window_days=HISTORY_WINDOW_DAYS, frequency=freq_label
        ) or {}
        rr = detect_regime(
            asset,
            start_time,
            time_increment,
            time_length,
            hist,
        )
        market_regime = rr.regime
        regime_asset_type = rr.asset_type
        regime_confidence = rr.confidence
    except Exception:
        market_regime = None

    strategy_store = get_strategy_store()
    strategy_configs = strategy_store.get_strategy_list(
        asset, time_length, market_regime=market_regime
    )
    ensemble_names = [sc.strategy_name for sc in strategy_configs[:ENSEMBLE_TOP_N]]

    print(
        f"[INFO] entry: asset={asset}, time_length={time_length}, "
        f"regime_type={regime_asset_type}, market_regime={market_regime}, "
        f"regime_conf={regime_confidence}, "
        f"ensemble={ensemble_names}, n={num_simulations}, seed={seed}"
    )

    # Build the simulate function map
    sim_fn_map = _build_simulator_functions()

    # ── Phase 1: Ensemble mode ──
    builder = _get_ensemble_builder()
    ensemble_paths = _run_ensemble_via_builder(
        builder, strategy_configs, sim_fn_map,
        asset, start_time, time_increment, time_length,
        num_simulations, seed,
    )

    if ensemble_paths is not None:
        predictions = convert_prices_to_time_format(
            ensemble_paths.tolist(), start_time, time_increment
        )
        fmt = validate_responses(predictions, simulation_input, "0")
        if fmt == "CORRECT":
            print(f"[INFO] entry: ensemble SUCCESS")
            return {"predictions": predictions}
        print(f"[WARN] entry: ensemble failed validation ({fmt})")

    # ── Phase 2: Sequential fallback ──
    print(f"[WARN] entry: falling back to sequential mode")

    try_names = [sc.strategy_name for sc in strategy_configs]
    for fb in strategy_store.get_fallback_chain():
        if fb not in try_names:
            try_names.append(fb)

    params_by_name: dict[str, dict] = {}
    for sc in strategy_configs:
        if sc.strategy_name not in params_by_name:
            params_by_name[sc.strategy_name] = sc.params or {}

    for sim_name in try_names:
        fn = _get_simulate_fn(sim_name)
        if fn is None:
            continue
        try:
            extra = dict(params_by_name.get(sim_name, {}))
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
                **extra,
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
                print(f"[INFO] entry: fallback used simulator={sim_name}")
                return {"predictions": predictions}
        except Exception as e:
            print(f"[WARN] entry: {sim_name} failed: {e}")

    return {"predictions": None}


def _run_ensemble_via_builder(
    builder: EnsembleBuilder,
    strategy_configs: list[StrategyConfig],
    sim_fn_map: dict[str, Callable],
    asset: str,
    start_time: str,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    seed: int,
) -> Optional[np.ndarray]:
    """
    Adapter: use EnsembleBuilder with the existing simulate_crypto_price_paths
    pattern (which handles data loading internally).
    """
    # For ensemble, we need to create wrapper fns that go through
    # simulate_crypto_price_paths (which handles data loading)
    def _make_ensemble_fn(fn):
        def wrapper(prices_dict, asset, time_increment, time_length, n_sims, seed=42, **kwargs):
            return simulate_crypto_price_paths(
                current_price=None,
                asset=asset,
                start_time=start_time,
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=n_sims,
                simulate_fn=fn,
                max_data_points=None,
                seed=seed,
                miner_start_time=start_time,
                **kwargs,
            )
        return wrapper

    # Build a map of strategy_name -> wrapper fn
    wrapped_map = {}
    for sc in strategy_configs[:builder.top_n]:
        raw_fn = sim_fn_map.get(sc.strategy_name)
        if raw_fn is not None:
            wrapped_map[sc.strategy_name] = _make_ensemble_fn(raw_fn)

    if not wrapped_map:
        return None

    return builder.build(
        strategy_configs=strategy_configs,
        simulate_fn_map=wrapped_map,
        prices_dict={},  # Not used — wrapper handles data loading
        asset=asset,
        start_time=start_time,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
        seed=seed,
    )

"""
simulations.py

Ported from sn50/synth/miner/simulations.py (167 lines).

Changes from sn50:
    1. Replaced GBM-only logic (synth original, 64 lines) with GARCH-based
       orchestration from sn50 (167 lines).
    2. Adapted validate_responses() call:
       - sn50 uses 4 args: (response, simulation_input, request_time, process_time_str)
       - synth uses 3 args: (response, simulation_input, process_time_str)
       → Using synth's 3-arg version.
    3. Kept generate_simulations_original() as fallback (sn50 L30-81).

Asset routing logic (sn50 L112-135):
    BTC/ETH/SOL/XAU → [garch_v2, garch_v1, har_garch] (fallback chain)
    XAU + v2_1_0    → [garch_v2_1]
    NVDAX           → [seasonal_stock]
    TSLAX/AAPLX/GOOGLX/XAU → [weekly_seasonal_optimized]
"""

from datetime import datetime
from synth.simulation_input import SimulationInput
from synth.miner.price_simulation import (
    get_asset_price,
)

# Core simulators (from sn50/synth/miner/core/)
from synth.miner.core.garch_simulator import simulate_single_price_path_with_garch
from synth.miner.core.HAR_RV_simulatior import simulate_single_price_path_with_har_garch
from synth.miner.core.grach_simulator_v2 import (
    simulate_single_price_path_with_garch as simulate_single_price_path_with_garch_v2,
)
from synth.miner.core.grach_simulator_v2_1 import (
    simulate_single_price_path_with_garch as simulate_single_price_path_with_garch_v2_1,
)
from synth.miner.core.aparch_simulator import simulate_aparch_optimized
from synth.miner.core.spyx_simulator import (
    simulate_spyx_sniper,
    simulate_spyx_robust,
    simulate_fhs_antithetic,
)
from synth.miner.core.stock_simulator import simulate_seasonal_stock
from synth.miner.core.stock_simulator_v2 import simulate_weekly_seasonal_optimized

# Data pipeline (from sn50/synth/miner/my_simulation.py)
from synth.miner.my_simulation import simulate_crypto_price_paths

# Validation & utilities
from synth.validator.response_validation_v2 import validate_responses
from synth.utils.helpers import convert_prices_to_time_format


def generate_simulations_original(
    asset="BTC",
    start_time: str = "",
    time_increment=300,
    time_length=86400,
    num_simulations=1,
    sigma=0.01,
):
    """
    Original GBM-based simulation (fallback).
    sn50 L30-81
    """
    if start_time == "":
        raise ValueError("Start time must be provided.")

    current_price = get_asset_price(asset)
    if current_price is None:
        raise ValueError(f"Failed to fetch current price for asset: {asset}")

    if asset == "BTC":
        sigma *= 3
    elif asset == "ETH":
        sigma *= 1.25
    elif asset == "XAU":
        sigma *= 0.5
    elif asset == "SOL":
        sigma *= 0.75

    simulations = simulate_crypto_price_paths(
        current_price=current_price,
        asset=asset,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
        sigma=sigma,
    )

    predictions = convert_prices_to_time_format(
        simulations.tolist(), start_time, time_increment
    )

    return predictions


def generate_simulations(
    simulation_input: SimulationInput,
    asset="BTC",
    start_time: str = "",
    time_increment=300,
    time_length=86400,
    num_simulations=1,
    seed=42,
    version=None,
):
    """
    Generate simulated price paths using GARCH-based models.
    
    Ported from sn50 L83-167. Uses a fallback chain of simulation functions:
    tries each in order until one produces valid output.
    
    Asset routing:
        BTC/ETH/SOL/XAU → [garch_v2, garch_v1, har_garch]
        XAU + v2_1_0    → [garch_v2_1]
        NVDAX           → [seasonal_stock]
        TSLAX/AAPLX/GOOGLX/XAU → [weekly_seasonal_optimized]
    
    Args:
        simulation_input: SimulationInput pydantic model
        asset: Asset to simulate
        start_time: ISO format start time
        time_increment: Time increment in seconds
        time_length: Total simulation time in seconds
        num_simulations: Number of simulation paths
        seed: Random seed
        version: Optional version string for asset-specific models
    
    Returns:
        dict: {"predictions": predictions} where predictions is a tuple
    """
    if start_time == "":
        raise ValueError("Start time must be provided.")

    current_price = None
    print(
        f"[INFO] Generating simulations for asset: {asset}, version: {version}, "
        f"time_increment: {time_increment}, time_length: {time_length}, "
        f"num_simulations: {num_simulations}, seed: {seed}"
    )

    # === Asset routing logic (sn50 L112-135) ===
    # Default: BTC/ETH/SOL/XAU → GARCH v2 with fallbacks
    if asset in ["BTC", "ETH", "SOL", "XAU"]:
        lst_simulate_fn = [
            simulate_single_price_path_with_garch_v2,
            simulate_single_price_path_with_garch,
            simulate_single_price_path_with_har_garch,
        ]
        max_data_points_list = [None, 500, 100000]
    else:
        lst_simulate_fn = [
            simulate_single_price_path_with_garch_v2,
            simulate_single_price_path_with_har_garch,
            simulate_single_price_path_with_garch,
        ]
        max_data_points_list = [None, 100000, 500]

    # XAU version override
    if asset == "XAU" and version == "v2_1_0":
        lst_simulate_fn = [simulate_single_price_path_with_garch_v2_1]
        max_data_points_list = [None]

    # Stock asset overrides
    if asset in ["NVDAX"]:
        lst_simulate_fn = [simulate_seasonal_stock]
        max_data_points_list = [None]
    if asset in ["TSLAX", "AAPLX", "GOOGLX", "XAU"]:
        lst_simulate_fn = [simulate_weekly_seasonal_optimized]
        max_data_points_list = [None]

    # === Try each simulate_fn in order (fallback chain) ===
    predictions = None
    for simulate_fn, mdp in zip(lst_simulate_fn, max_data_points_list):
        try:
            simulations = simulate_crypto_price_paths(
                current_price=current_price,
                asset=asset,
                start_time=start_time,
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=num_simulations,
                simulate_fn=simulate_fn,
                max_data_points=mdp,
                seed=seed,
            )

            predictions = convert_prices_to_time_format(
                simulations.tolist(), start_time, time_increment
            )

            # Validate format (adapted for synth's 3-arg validate_responses)
            format_validation = validate_responses(
                predictions,
                simulation_input,
                "0",  # process_time_str (dummy for backtest)
            )
            if format_validation == "CORRECT":
                print(f"[INFO] Simulation function: {simulate_fn.__name__} is correct")
                return {"predictions": predictions}
        except Exception as e:
            print(
                f"[ERROR] Error generating simulations with function "
                f"{simulate_fn.__name__}: {e}"
            )

    return {"predictions": predictions}

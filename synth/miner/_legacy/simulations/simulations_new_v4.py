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
  - (NEW) Using Weighted Ensemble and Outlier Trimming for extreme fat tails
"""

import numpy as np
from typing import Callable, Optional

from synth.simulation_input import SimulationInput
from synth.miner.price_simulation import get_asset_price

# Core simulators
from synth.miner.core.garch_simulator import simulate_single_price_path_with_garch as sim_garch_v1
from synth.miner.core.grach_simulator_v2 import simulate_single_price_path_with_garch as sim_garch_v2
from synth.miner.core.grach_simulator_v2_1 import simulate_single_price_path_with_garch as sim_garch_v2_1
from synth.miner.core.garch_simulator_v2_2 import simulate_single_price_path_with_garch as sim_garch_v2_2
from synth.miner.core.HAR_RV_simulatior import simulate_single_price_path_with_har_garch as sim_har_rv
from synth.miner.core.stock_simulator import simulate_seasonal_stock as sim_seasonal_stock
from synth.miner.core.stock_simulator_v2 import simulate_weekly_seasonal_optimized as sim_weekly_stock

# Strategies (v4 family)
from synth.miner.core.grach_simulator_v4 import simulate_single_price_path_with_garch as sim_garch_v4
from synth.miner.core.grach_simulator_v4_1 import simulate_single_price_path_with_garch as sim_garch_v4_1
from synth.miner.core.grach_simulator_v4_2 import simulate_single_price_path_with_garch as sim_garch_v4_2

# Advanced Weekly Variants
from synth.miner.core.stock_simulator_v3 import simulate_weekly_garch_v4 as sim_weekly_garch_v4
from synth.miner.core.stock_simulator_v4 import simulate_weekly_regime_switching as sim_weekly_regime_switching

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
    "garch_v2_2": sim_garch_v2_2,
    "garch_v4": sim_garch_v4,
    "garch_v4_1": sim_garch_v4_1,
    "garch_v4_2": sim_garch_v4_2,
    "har_rv": sim_har_rv,
    "seasonal_stock": sim_seasonal_stock,
    "weekly_stock": sim_weekly_stock,
    "weekly_garch_v4": sim_weekly_garch_v4,
    "weekly_regime_switching": sim_weekly_regime_switching,
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
# Strategy lists per (asset, prompt_type) with WEIGHTS.
# Format: [("strategy_name", weight_float), ...]
# ---------------------------------------------------------------------------
STRATEGY_LIST_FOR_ASSET: dict[tuple[str, str], list[tuple[str, float]]] = {
    # HIGH: Cấu hình ensemble tối ưu dựa trên backtest tổng hợp
    # ("BTC", "high"): [("garch_v2_2", 0.4), ("garch_v4", 0.3), ("mean_reversion", 0.3)],
    ("BTC", "high"): [("garch_v2_1", 1.0)],
    # ("ETH", "high"): [("garch_v2_2", 0.4), ("garch_v4", 0.3), ("mean_reversion", 0.3)],
    ("ETH", "high"): [("garch_v4", 1.0)],
    # ("XAU", "high"): [("equity_exact_hours", 0.4), ("jump_diffusion", 0.3), ("garch_v2_2", 0.3)],
    ("XAU", "high"): [("garch_v4_1", 1.0)],
    # ("SOL", "high"): [("garch_v4", 0.4), ("garch_v2_1", 0.3), ("mean_reversion", 0.3)],
    ("SOL", "high"): [("ensemble_weighted", 1.0)],

    # LOW: Cập nhật allocation tối ưu dựa trên backtest tổng hợp
    # ("BTC", "low"):    [("garch_v4", 0.4), ("garch_v2_2", 0.3), ("gjr_garch", 0.3)],
    ("BTC", "low"): [
        ("garch_v4", 0.4),         # 40% trọng số: Cỗ máy chủ lực bám trend giảm
        ("ensemble_garch_v2_v4", 0.4), # 40% trọng số: Trung hòa phương sai, chống nổ điểm
        ("arima_equity", 0.2)      # 20% trọng số: Cảm biến hồi quy tuyến tính
    ],
    ("ETH", "low"): [("garch_v4", 0.4), ("garch_v2_2", 0.3), ("regime_switching", 0.3)],
    # ("XAU", "low"): [("jump_diffusion", 0.4), ("garch_v4_1", 0.3), ("weekly_regime_switching", 0.3)],
    ("XAU", "low"): [
        ("jump_diffusion", 0.5),     # 50%: Trụ cột chống nổ CRPS khi giật Gap/Sập
        ("regime_switching", 0.3),   # 30%: Bắt nhịp tốt khi thị trường chuyển trạng thái sang Sideway
        ("garch_v4_1", 0.2)          # 20%: Hỗ trợ bám trend khi có xu hướng rõ ràng
    ],

    # ("SOL", "low"): [("garch_v2_2", 0.4), ("garch_v4", 0.3), ("regime_switching", 0.3)],
    ("SOL", "low"): [
        ("garch_v4", 0.4),            # 40%: Động cơ chính để bám trend rơi/bơm mạnh
        ("arima_equity", 0.3),        # 30%: Bắt nhịp Sideway và tạo nền móng tuyến tính
        ("mean_reversion", 0.3)       # 30%: Lực hút kéo dải phân phối lại khi SOL giật nảy 
    ],


    # Stocks (low only) - V5 Fully Deseasonalized (Weekly) Allocations
    ("NVDAX", "low"): [("arima_equity", 0.4), ("garch_v4_1", 0.3), ("markov_garch_jump", 0.3)],
    ("TSLAX", "low"): [("weekly_garch_v4", 0.4), ("garch_v4", 0.3), ("regime_switching", 0.3)],
    ("AAPLX", "low"): [("markov_garch_jump", 0.4), ("regime_switching", 0.3), ("garch_v4_1", 0.3)],
    ("GOOGLX", "low"): [("weekly_regime_switching", 0.4), ("garch_v4_1", 0.3), ("regime_switching", 0.3)],
    ("SPYX", "low"): [("weekly_regime_switching", 0.4), ("garch_v4", 0.3), ("arima_equity", 0.3)],
}

DEFAULT_FALLBACK_CHAIN = [
    "arima_equity",
    "equity_exact_hours",
    "garch_v2_2",
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
    # Default fallback list with weights
    return [("garch_v2", 0.5), ("garch_v4", 0.3), ("garch_v4_1", 0.2)]


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
    strategy_list: list[tuple[str, float]],
    asset: str,
    start_time: str,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    seed: int,
) -> Optional[np.ndarray]:
    """
    Chạy mô phỏng theo trọng số, over-request 10% để lấy không gian gọt giũa (trimming),
    và loại bỏ các path cực đoan nhằm tránh bùng nổ phương sai (variance explosion).
    """
    n_strategies = min(len(strategy_list), _ENSEMBLE_TOP_N)
    active_strategies = strategy_list[:n_strategies]

    # Chuẩn hóa trọng số
    total_weight = sum(w for _, w in active_strategies)
    normalized_strats = [(name, w / total_weight) for name, w in active_strategies]

    all_paths: list[np.ndarray] = []
    used_strategies: list[str] = []

    # OVER-REQUEST: Xin dư 10% tổng số path
    target_total_sims = int(num_simulations * 1.10)
    sims_collected = 0

    for i, (sim_name, weight) in enumerate(normalized_strats):
        fn = _get_simulate_fn(sim_name)
        if fn is None:
            continue

        # Chia path theo trọng số chuẩn hóa
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

    # --- OUTLIER TRIMMING ---
    # Tính lợi suất từ đầu đến cuối của mỗi path
    start_prices = combined[:, 0]
    final_prices = combined[:, -1]
    
    # Tránh chia cho 0
    safe_start_prices = np.where(start_prices == 0, 1e-8, start_prices)
    returns = (final_prices - safe_start_prices) / safe_start_prices

    # Xác định biên phân vị 1% và 99%
    lower_bound = np.percentile(returns, 1.0)
    upper_bound = np.percentile(returns, 99.0)

    # Chỉ giữ lại các path nằm giữa 2 biên
    valid_indices = np.where((returns >= lower_bound) & (returns <= upper_bound))[0]
    trimmed_combined = combined[valid_indices]

    rng = np.random.RandomState(seed)
    
    # Cắt chính xác về số lượng num_simulations mong muốn
    if trimmed_combined.shape[0] >= num_simulations:
        # Chọn ngẫu nhiên KHÔNG LẶP (replace=False) từ các path đã lọc
        selected_indices = rng.choice(trimmed_combined.shape[0], size=num_simulations, replace=False)
        final_ensemble = trimmed_combined[selected_indices]
    else:
        # Fallback an toàn nếu việc lọc khiến ta bị hụt path
        print(f"[WARN] Sau khi trim chỉ còn {trimmed_combined.shape[0]} paths, xử lý dự phòng.")
        if combined.shape[0] >= num_simulations:
            selected_indices = rng.choice(combined.shape[0], size=num_simulations, replace=False)
            final_ensemble = combined[selected_indices]
        else:
            selected_indices = rng.choice(combined.shape[0], size=num_simulations, replace=True)
            final_ensemble = combined[selected_indices]

    print(f"[ENSEMBLE] {asset}: {' + '.join(used_strategies)} => Trimming & Selected {final_ensemble.shape[0]} paths")
    return final_ensemble


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
    
    # Trích xuất tên strategy để in log
    ensemble_names = [name for name, _ in strategy_list[:_ENSEMBLE_TOP_N]]

    print(
        f"[INFO] simulations_new: asset={asset}, time_length={time_length}, "
        f"ensemble={ensemble_names}, n={num_simulations}, seed={seed}"
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

    # Bóc tách lấy TÊN strategy từ danh sách tuples
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
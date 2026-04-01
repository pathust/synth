"""
simulations_new_stable.py

Mục tiêu: giảm biến động rank (top 1 / top cuối) bằng cách:
- Seed hoàn toàn deterministic (không dùng built-in hash()).
- Ensemble đa dạng nhưng "an toàn": luôn sanitize/clip log-returns để tránh path outlier.

Interface giống `simulations_new.generate_simulations`.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Optional

import numpy as np

from synth.simulation_input import SimulationInput
from synth.miner.my_simulation import simulate_crypto_price_paths
from synth.validator.response_validation_v2 import validate_responses
from synth.utils.helpers import convert_prices_to_time_format

# Core simulators (re-use từ simulations_new)
from synth.miner.core.grach_simulator_v2 import simulate_single_price_path_with_garch as sim_garch_v2
from synth.miner.strategies.grach_simulator_v4 import simulate_single_price_path_with_garch as sim_garch_v4
from synth.miner.strategies.grach_simulator_v4_2 import simulate_single_price_path_with_garch as sim_garch_v4_2


SIMULATOR_FUNCTIONS: dict[str, Callable] = {
    "garch_v2": sim_garch_v2,
    "garch_v4": sim_garch_v4,
    "garch_v4_2": sim_garch_v4_2,
}

# A conservative, stable default list (order matters, deterministic)
STRATEGY_LIST_FOR_ASSET: dict[tuple[str, str], list[str]] = {
    ("BTC", "high"): ["garch_v2", "garch_v4"],
    ("ETH", "high"): ["garch_v2", "garch_v4"],
    ("SOL", "high"): ["garch_v2", "garch_v4"],
    ("XAU", "high"): ["garch_v2", "garch_v4"],
    ("BTC", "low"): ["garch_v2", "garch_v4_2"],
    ("ETH", "low"): ["garch_v2", "garch_v4_2"],
    ("SOL", "low"): ["garch_v2", "garch_v4_2"],
    ("XAU", "low"): ["garch_v2", "garch_v4_2"],
}

DEFAULT_STRATEGIES = ["garch_v2", "garch_v4", "garch_v4_2"]

_ENSEMBLE_TOP_N = 2

# Drop extreme simulated paths to reduce tail-risk / rank crashes.
# Keep central (1 - TRIM_FRACTION) paths by a robust score, then resample back to N.
_TRIM_FRACTION = 0.20  # drop 20% most extreme
_VOL_WEIGHT = 0.5      # weight for realized vol in extremeness score


def _get_prompt_type(time_length: int) -> str:
    return "high" if time_length == 3600 else "low"


def _get_strategy_list(asset: str, time_length: int) -> list[str]:
    key = (asset, _get_prompt_type(time_length))
    return list(STRATEGY_LIST_FOR_ASSET.get(key, DEFAULT_STRATEGIES))


def _deterministic_sub_seed(base_seed: int, strategy_name: str) -> int:
    """
    Deterministic across processes/machines.
    Python built-in hash() thay đổi theo PYTHONHASHSEED -> tránh dùng.
    """
    h = hashlib.sha256(f"{base_seed}|{strategy_name}".encode()).digest()
    v = int.from_bytes(h[:4], "big", signed=False)
    return v & 0x7FFF_FFFF


def _sanitize_and_clip_paths(paths: np.ndarray, time_increment: int) -> np.ndarray:
    """
    paths: (n_sims, steps+1), prices.
    - replace non-finite / <=0 với forward-fill, rồi clip log-returns để tránh outlier.
    """
    p = np.array(paths, dtype=float, copy=True)
    if p.ndim != 2 or p.shape[1] < 2:
        return p

    # Replace non-finite or <=0 with NaN then forward-fill
    bad = ~np.isfinite(p) | (p <= 0)
    if np.any(bad):
        p[bad] = np.nan
        # forward fill along axis=1
        for i in range(p.shape[0]):
            row = p[i]
            if np.all(np.isnan(row)):
                continue
            # fill leading nans with first non-nan
            first_idx = np.argmax(~np.isnan(row))
            row[:first_idx] = row[first_idx]
            # forward fill
            for t in range(first_idx + 1, row.shape[0]):
                if np.isnan(row[t]):
                    row[t] = row[t - 1]
            p[i] = row

    # log-return clipping
    # Caps: tighter for 1m than 5m to prevent extreme tails.
    # Chọn khá bảo thủ để giảm xác suất rơi vào top cuối 10%.
    if int(time_increment) == 60:
        r_cap = 0.01  # 1% per minute cap
    elif int(time_increment) == 300:
        r_cap = 0.03  # 3% per 5-min cap
    else:
        r_cap = 0.03

    logp = np.log(p)
    r = np.diff(logp, axis=1)
    r = np.clip(r, -r_cap, r_cap)
    logp2 = np.concatenate([logp[:, :1], logp[:, :1] + np.cumsum(r, axis=1)], axis=1)
    p2 = np.exp(logp2)

    # Ensure no zeros
    p2[p2 <= 0] = np.min(p2[p2 > 0]) if np.any(p2 > 0) else 1.0
    return p2


def _trim_extreme_paths(paths: np.ndarray, seed: int) -> np.ndarray:
    """
    Remove the most extreme paths (tail-heavy) deterministically.

    Score each path by:
      extremeness = |total_log_return| + VOL_WEIGHT * realized_vol
    Drop top TRIM_FRACTION by extremeness, then resample to original N.
    """
    p = np.asarray(paths, dtype=float)
    if p.ndim != 2 or p.shape[0] < 5 or p.shape[1] < 3:
        return p

    logp = np.log(p)
    r = np.diff(logp, axis=1)
    total_lr = logp[:, -1] - logp[:, 0]
    vol = np.nanstd(r, axis=1)
    score = np.abs(total_lr) + (_VOL_WEIGHT * vol)

    n = p.shape[0]
    k_keep = max(3, int(round(n * (1.0 - _TRIM_FRACTION))))
    keep_idx = np.argsort(score)[:k_keep]  # keep most "central" paths
    kept = p[keep_idx]

    # Resample back to n deterministically
    rng = np.random.RandomState(int(seed) & 0x7FFF_FFFF)
    if kept.shape[0] < n:
        extra = rng.choice(kept.shape[0], size=n - kept.shape[0], replace=True)
        kept = np.vstack([kept, kept[extra]])
    elif kept.shape[0] > n:
        kept = kept[:n]
    return kept


def _run_ensemble(
    strategy_list: list[str],
    asset: str,
    start_time: str,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    seed: int,
) -> Optional[np.ndarray]:
    n_strategies = min(len(strategy_list), _ENSEMBLE_TOP_N)
    sims_per = max(1, num_simulations // n_strategies)

    all_paths: list[np.ndarray] = []
    sims_collected = 0

    for i, name in enumerate(strategy_list[:n_strategies]):
        fn = SIMULATOR_FUNCTIONS.get(name)
        if fn is None:
            continue
        n_sub = (num_simulations - sims_collected) if i == n_strategies - 1 else sims_per
        if n_sub <= 0:
            continue
        sub_seed = _deterministic_sub_seed(seed, name)
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
            if isinstance(paths, np.ndarray) and paths.ndim == 2 and paths.shape[0] > 0:
                paths = _sanitize_and_clip_paths(paths, time_increment)
                all_paths.append(paths)
                sims_collected += paths.shape[0]
        except Exception:
            continue

    if not all_paths:
        return None

    combined = np.vstack(all_paths)
    # Trim extreme paths to reduce tail risk, then ensure exact N
    combined = _trim_extreme_paths(combined, seed)
    if combined.shape[0] < num_simulations:
        rng = np.random.RandomState(seed)
        extra = rng.choice(combined.shape[0], size=num_simulations - combined.shape[0], replace=True)
        combined = np.vstack([combined, combined[extra]])
    elif combined.shape[0] > num_simulations:
        combined = combined[:num_simulations]
    return combined


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

    strategies = _get_strategy_list(asset, time_length)
    paths = _run_ensemble(
        strategies, asset, start_time, time_increment, time_length, num_simulations, seed
    )
    if paths is None:
        return {"predictions": None}

    predictions = convert_prices_to_time_format(paths.tolist(), start_time, time_increment)
    fmt = validate_responses(predictions, simulation_input, "0")
    if fmt != "CORRECT":
        return {"predictions": None}
    return {"predictions": predictions}


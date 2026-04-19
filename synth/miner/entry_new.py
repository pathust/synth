"""
entry_new.py — Custom entry with fixed tuned GARCH configs.

Rules:
- **High** (time_length=3600): BTC / ETH / SOL / XAU — tuned GARCH in ``_HIGH_FIXED`` when it succeeds; else unified ``entry``.
  Equities **GOOGLX, AAPLX, NVDAX, TSLAX**: **Mon–Fri UTC** → unified ``entry``; **Sat/Sun UTC** → ``entry_old`` (same as low).
- **Low**:
  - BTC / ETH / SOL / XAU: always ``entry_old``.
  - SPYX: always ``entry_old``.
  - Equities GOOGLX, AAPLX, NVDAX, TSLAX: **Mon–Fri UTC** → unified ``entry``; **Sat/Sun UTC** → ``entry_old``.
  - Any other low asset: unified ``entry``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from synth.simulation_input import SimulationInput
from synth.miner.my_simulation import simulate_crypto_price_paths
from synth.utils.helpers import convert_prices_to_time_format
from synth.validator.response_validation_v2 import validate_responses


# Tuned winners (picked by lowest train_avg_score) from:
# result/tune_garch_grid_pm2_20260408_052225/garch_grid_tune_20260408_070645.json
_HIGH_FIXED: dict[str, tuple[str, dict]] = {
    "BTC": (
        "garch_v4",
        {
            "p": 2,
            "q": 1,
            "lookback_days": 5,
            "momentum_weight": 0.3,
            "drift_decay": 0.9,
        },
    ),
    "ETH": (
        "garch_v2_1",
        {
            "min_nu": 8.0,
            "vol_multiplier": 0.95,
            "grach_o": 1,
            "mean_model": "Constant",
            "lookback_days": 5.0,
        },
    ),
    "SOL": (
        "garch_v2_1",
        {
            "min_nu": 8.0,
            "vol_multiplier": 0.88,
            "grach_o": 0,
            "mean_model": "Zero",
            "lookback_days": 5.0,
        },
    ),
    "XAU": (
        "garch_v2_1",
        {
            "min_nu": 10.0,
            "vol_multiplier": 1.05,
            "grach_o": 1,
            "mean_model": "Constant",
            "lookback_days": 3.0,
        },
    ),
}

def _get_prompt_type(time_length: int) -> str:
    return "high" if time_length == 3600 else "low"


_LOW_LEGACY_ALWAYS = frozenset({"BTC", "ETH", "SOL", "XAU", "SPYX"})
# US-style equities: weekday session uses unified entry; Sat/Sun UTC uses legacy.
_EQ_TIME_REGIME = frozenset({"GOOGLX", "AAPLX", "NVDAX", "TSLAX"})


def _is_weekend_utc(start_time: str) -> bool:
    """True on Saturday or Sunday UTC (weekend → ``entry_old`` for ``_EQ_TIME_REGIME``)."""
    dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    # Monday=0 … Saturday=5, Sunday=6
    return dt.weekday() >= 5


def _legacy(
    *,
    simulation_input: SimulationInput,
    asset: str,
    start_time: str,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    seed: int,
    version: str | None,
) -> dict:
    from synth.miner.entry_old import generate_simulations_legacy

    return generate_simulations_legacy(
        simulation_input=simulation_input,
        asset=asset,
        start_time=start_time,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
        seed=seed,
        version=version,
    )


def _unified(
    *,
    simulation_input: SimulationInput,
    asset: str,
    start_time: str,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    seed: int,
    version: str | None,
) -> dict:
    from synth.miner.entry import generate_simulations as _gen

    return _gen(
        simulation_input=simulation_input,
        asset=asset,
        start_time=start_time,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
        seed=seed,
        version=version,
    )


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

    prompt = _get_prompt_type(time_length)

    # ── HIGH: force tuned config ─────────────────────────────────────────────
    if prompt == "high" and asset in _HIGH_FIXED:
        sim_name, params = _HIGH_FIXED[asset]
        from synth.miner.entry import _get_simulate_fn as _entry_get_simulate_fn

        fn = _entry_get_simulate_fn(sim_name)
        if fn is not None:
            try:
                paths = simulate_crypto_price_paths(
                    current_price=None,  # type: ignore[arg-type]
                    asset=asset,
                    start_time=start_time,
                    time_increment=time_increment,
                    time_length=time_length,
                    num_simulations=num_simulations,
                    simulate_fn=fn,
                    max_data_points=None,
                    seed=seed,
                    miner_start_time=start_time,
                    **params,
                )
                if paths is not None:
                    predictions = convert_prices_to_time_format(
                        paths.tolist(), start_time, time_increment
                    )
                    fmt = validate_responses(predictions, simulation_input, "0")
                    if fmt == "CORRECT":
                        return {"predictions": predictions}
            except Exception:
                pass

    # ── LOW: majors + SPYX always legacy ──────────────────────────────────
    if prompt == "low" and asset in _LOW_LEGACY_ALWAYS:
        return _legacy(
            simulation_input=simulation_input,
            asset=asset,
            start_time=start_time,
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            seed=seed,
            version=version,
        )

    # ── Equities (high + low): weekday → entry, weekend UTC → entry_old ────
    if asset in _EQ_TIME_REGIME:
        if _is_weekend_utc(start_time):
            return _legacy(
                simulation_input=simulation_input,
                asset=asset,
                start_time=start_time,
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=num_simulations,
                seed=seed,
                version=version,
            )
        return _unified(
            simulation_input=simulation_input,
            asset=asset,
            start_time=start_time,
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            seed=seed,
            version=version,
        )

    # Default: unified entry (strategies.yaml routing + ensemble/fallback).
    return _unified(
        simulation_input=simulation_input,
        asset=asset,
        start_time=start_time,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_simulations,
        seed=seed,
        version=version,
    )


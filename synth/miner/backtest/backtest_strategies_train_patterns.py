"""
backtest_strategies_train_patterns.py

Read pattern windows from a CSV (e.g. btc.csv) and backtest ALL strategies
(strategies registry + core simulators) on TRAIN windows.

Requirements:
- Use TRAIN set windows as the evaluation schedule
- HIGH mode (1m increment, 1h horizon): compute CRPS per hour slot
- Simulate 1000 paths per strategy per slot
- CRPS computed using the validator's interval-change pipeline + point-formula
- Save JSON under: result/assert/train/pattern/

Output format is compatible with existing result/backtest_strategies high layout:
  {
    "asset": "BTC",
    "prompt": "high",
    "set": "TRAIN",
    "pattern": "...",
    "windows": [...],
    "num_sims": 1000,
    "seed": 42,
    "hourly": [
      {
        "window_index": 0,
        "window_start": "...",
        "window_end": "...",
        "hour": 12,
        "slot_index": 0,
        "start_time": "...",
        "strategies": [{"name": "...", "crps": 123.0, "status":"SUCCESS"}, ...],
        "best_strategy": "..."
      }, ...
    ]
  }
"""

from __future__ import annotations
import contextlib
import csv
import io
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np

from synth.miner.data_handler import DataHandler
from synth.miner.my_simulation import simulate_crypto_price_paths
from synth.validator.prompt_config import HIGH_FREQUENCY

from synth.validator.crps_calculation import (
    get_interval_steps,
    calculate_price_changes_over_intervals,
    label_observed_blocks,
)


def _parse_utc(s: str) -> datetime:
    d = datetime.fromisoformat((s or "").strip().replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def _floor_hour(d: datetime) -> datetime:
    return d.replace(minute=0, second=0, microsecond=0)


def crps_point_formula(x: float, y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n == 0:
        return np.nan
    term1 = np.mean(np.abs(y - x))
    term2 = np.sum(np.abs(y[:, None] - y[None, :])) / (2.0 * (n**2))
    return float(term1 - term2)


def calculate_crps_with_formula(
    simulation_runs: np.ndarray,
    real_price_path: np.ndarray,
    time_increment: int,
    scoring_intervals: dict[str, int],
) -> float:
    sum_all_scores = 0.0
    real_path = np.asarray(real_price_path, dtype=float).ravel()
    sim = np.asarray(simulation_runs, dtype=float)
    if sim.size == 0 or real_path.size == 0 or np.any(sim == 0):
        return -1.0

    for interval_name, interval_seconds in scoring_intervals.items():
        interval_steps = get_interval_steps(interval_seconds, time_increment)
        absolute_price = interval_name.endswith("_abs")
        is_gap = interval_name.endswith("_gap")

        if absolute_price:
            while real_path[::interval_steps].shape[0] == 1 and interval_steps > 1:
                interval_steps -= 1

        simulated_changes = calculate_price_changes_over_intervals(
            sim, interval_steps, absolute_price, is_gap
        )
        real_changes = calculate_price_changes_over_intervals(
            real_path.reshape(1, -1), interval_steps, absolute_price, is_gap
        )
        blocks = label_observed_blocks(real_changes[0])
        if len(blocks) == 0:
            continue

        for block in np.unique(blocks):
            if block == -1:
                continue
            mask = blocks == block
            sim_block = simulated_changes[:, mask]
            real_block = real_changes[0, mask]
            for t in range(sim_block.shape[1]):
                x = float(real_block[t])
                y = sim_block[:, t]
                crps_t = crps_point_formula(x, y)
                if np.isfinite(crps_t):
                    if absolute_price:
                        crps_t = crps_t / real_path[-1] * 10_000
                    sum_all_scores += crps_t

    return float(sum_all_scores)


def _get_all_strategy_names() -> list[str]:
    from synth.miner.entry import _build_simulator_functions

    return sorted(_build_simulator_functions().keys())


def _get_simulate_fn(name: str) -> Optional[Callable]:
    from synth.miner.entry import _get_simulate_fn

    return _get_simulate_fn(name)


def _get_real_prices(
    data_handler: DataHandler,
    asset: str,
    start_time: datetime,
    time_length: int,
    time_increment: int,
) -> np.ndarray:
    from synth.db.models import ValidatorRequest

    req = ValidatorRequest(
        asset=asset,
        start_time=start_time,
        time_length=time_length,
        time_increment=time_increment,
    )
    real = data_handler.get_real_prices(validator_request=req)
    return np.asarray(real, dtype=float)


def _run_one_strategy(
    *,
    strategy_name: str,
    asset: str,
    slot_start: datetime,
    real_prices: np.ndarray,
    num_sims: int,
    seed: int,
    verbose: bool,
) -> dict:
    fn = _get_simulate_fn(strategy_name)
    if fn is None:
        return {"name": strategy_name, "crps": None, "status": "SKIP"}

    cfg = HIGH_FREQUENCY

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    cm_out = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(out_buf)
    cm_err = contextlib.nullcontext() if verbose else contextlib.redirect_stderr(err_buf)

    with cm_out, cm_err:
        try:
            paths = simulate_crypto_price_paths(
                current_price=None,
                asset=asset,
                start_time=slot_start.isoformat(),
                time_increment=cfg.time_increment,
                time_length=cfg.time_length,
                num_simulations=num_sims,
                simulate_fn=fn,
                max_data_points=None,
                seed=seed,
            )
        except Exception:
            return {"name": strategy_name, "crps": None, "status": "ERROR"}

    if paths is None or not getattr(paths, "shape", (0,))[0]:
        return {"name": strategy_name, "crps": None, "status": "FAIL"}

    if real_prices.size == 0:
        return {"name": strategy_name, "crps": None, "status": "NO_TRUTH"}

    crps = calculate_crps_with_formula(
        np.asarray(paths, dtype=float),
        real_prices,
        cfg.time_increment,
        cfg.scoring_intervals,
    )
    if crps < 0 or not np.isfinite(crps):
        return {"name": strategy_name, "crps": None, "status": "CRPS_INVALID"}
    return {"name": strategy_name, "crps": float(crps), "status": "SUCCESS"}


def _best(rows: list[dict]) -> Optional[str]:
    valid = [r for r in rows if r.get("crps") is not None and np.isfinite(r["crps"])]
    valid.sort(key=lambda x: x["crps"])
    return valid[0]["name"] if valid else None


@dataclass(frozen=True)
class Window:
    set_name: str
    pattern: str
    start_utc: datetime
    end_utc: datetime
    description: str


def read_windows(csv_path: str, set_name: str = "TRAIN", pattern: str | None = None) -> list[Window]:
    windows: list[Window] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("Set") or "").strip().upper() != set_name.upper():
                continue
            p = (row.get("Pattern") or "").strip()
            if pattern and p != pattern:
                continue
            windows.append(
                Window(
                    set_name=set_name.upper(),
                    pattern=p,
                    start_utc=_parse_utc(row.get("Start_UTC") or ""),
                    end_utc=_parse_utc(row.get("End_UTC") or ""),
                    description=(row.get("Description") or "").strip(),
                )
            )
    return windows


def run(
    *,
    asset: str,
    csv_path: str,
    set_name: str,
    num_sims: int,
    seed: int,
    output_dir: str,
    pattern: str | None = None,
    verbose: bool = False,
) -> list[str]:
    data_handler = DataHandler()
    all_names = _get_all_strategy_names()

    windows = read_windows(csv_path, set_name=set_name, pattern=pattern)
    if not windows:
        raise ValueError(f"No windows found for set={set_name} pattern={pattern!r} in {csv_path}")

    # Group by pattern
    by_pattern: dict[str, list[Window]] = {}
    for w in windows:
        by_pattern.setdefault(w.pattern, []).append(w)

    saved_paths: list[str] = []

    for pat, pat_windows in by_pattern.items():
        pat_windows = sorted(pat_windows, key=lambda w: w.start_utc)

        hourly_rows: list[dict] = []
        slot_index = 0
        print(f"\n=== PATTERN: {pat} | windows={len(pat_windows)} | strategies={len(all_names)} ===", flush=True)
        for w_idx, w in enumerate(pat_windows):
            cur = _floor_hour(w.start_utc)
            end = _floor_hour(w.end_utc)
            print(
                f"[WINDOW {w_idx+1}/{len(pat_windows)}] {w.start_utc.isoformat()} -> {w.end_utc.isoformat()}",
                flush=True,
            )
            while cur < end:
                t_slot0 = time.perf_counter()
                # Fetch truth once per slot (big speed-up)
                real_prices = _get_real_prices(
                    data_handler,
                    asset,
                    cur,
                    HIGH_FREQUENCY.time_length,
                    HIGH_FREQUENCY.time_increment,
                )
                if real_prices.size == 0:
                    print(f"  [SLOT] {cur.isoformat()} -> NO_TRUTH, skip", flush=True)
                    cur += timedelta(hours=1)
                    slot_index += 1
                    continue

                print(f"  [SLOT] {cur.isoformat()}  (idx={slot_index})", flush=True)
                rows = [
                    _run_one_strategy(
                        strategy_name=name,
                        asset=asset,
                        slot_start=cur,
                        real_prices=real_prices,
                        num_sims=num_sims,
                        seed=seed,
                        verbose=verbose,
                    )
                    for name in all_names
                ]
                rows_sorted = sorted(
                    rows,
                    key=lambda r: r["crps"] if r.get("crps") is not None else float("inf"),
                )
                best_name = _best(rows_sorted)
                dt_slot = time.perf_counter() - t_slot0
                print(
                    f"    done: best={best_name} slot_elapsed={dt_slot:.1f}s",
                    flush=True,
                )
                hourly_rows.append(
                    {
                        "window_index": int(w_idx),
                        "window_start": w.start_utc.isoformat(),
                        "window_end": w.end_utc.isoformat(),
                        "hour": int(cur.hour),
                        "slot_index": int(slot_index),
                        "start_time": cur.isoformat(),
                        "strategies": rows_sorted,
                        "best_strategy": best_name,
                    }
                )
                slot_index += 1
                cur += timedelta(hours=1)

        payload = {
            "asset": asset,
            "prompt": "high",
            "set": set_name.upper(),
            "pattern": pat,
            "windows": [
                {
                    "start_time": w.start_utc.isoformat(),
                    "end_time": w.end_utc.isoformat(),
                    "description": w.description,
                }
                for w in pat_windows
            ],
            "num_sims": int(num_sims),
            "seed": int(seed),
            "hourly": hourly_rows,
        }

        safe_pat = pat.lower().replace(" ", "_").replace("/", "_")
        out_dir = os.path.join(output_dir, set_name.lower(), safe_pat)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"backtest_strategies_{asset}_high_{set_name.lower()}_{safe_pat}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved: {out_path}")
        saved_paths.append(out_path)

    return saved_paths


def main():
    # Backtest cho đồng HIGH (nến 1m, horizon 1h) dựa trên windows trong CSV.
    # Chỉnh các biến dưới đây rồi chạy:
    #   PYTHONPATH=. python3 -m synth.miner.backtest.backtest_strategies_train_patterns

    _ASSET = "SOL"
    _CSV_PATH = "/home/user/synth/sol.csv"
    _SET_NAME = "TRAIN"  # "TRAIN" | "TEST"
    _PATTERN: str | None = None  # ví dụ: "Indecision"
    _NUM_SIMS = 100
    _SEED = 42
    _OUTPUT_DIR = "result/assert/train/pattern"
    _VERBOSE = False

    run(
        asset=_ASSET,
        csv_path=_CSV_PATH,
        set_name=_SET_NAME,
        num_sims=_NUM_SIMS,
        seed=_SEED,
        output_dir=_OUTPUT_DIR,
        pattern=_PATTERN,
        verbose=_VERBOSE,
    )


if __name__ == "__main__":
    main()


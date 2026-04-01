"""
backtest_strategies.py

Backtest tất cả chiến thuật (strategies + core) cho một đồng, loại request (high/low), và ngày.
Tính điểm CRPS theo công thức:

  CRPS = (1/N) * Σ_n |y_n - x| - (1/(2*N²)) * Σ_n Σ_m |y_n - y_m|

với x = giá trị quan sát (real), y_1..y_N = N mẫu dự báo (simulated paths tại cùng bước thời gian).
Tổng CRPS = tổng theo từng bước thời gian (sau khi chuyển sang thay đổi giá theo interval như validator).

Usage:
    python -m synth.miner.backtest_strategies --asset BTC --prompt high --date 2026-03-08 [--num-sims 500]
    python -m synth.miner.backtest_strategies --asset ETH --prompt low --date 2026-03-10
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np

from synth.miner.data_handler import DataHandler
from synth.db.models import ValidatorRequest
from synth.validator.prompt_config import (
    LOW_FREQUENCY,
    HIGH_FREQUENCY,
    get_prompt_labels_for_asset,
)
from synth.miner.my_simulation import simulate_crypto_price_paths
from synth.miner.price_aggregation import aggregate_1m_to_5m

# Reuse interval/changes logic from validator
from synth.validator.crps_calculation import (
    get_interval_steps,
    calculate_price_changes_over_intervals,
    label_observed_blocks,
)


def crps_point_formula(x: float, y: np.ndarray) -> float:
    """
    CRPS theo công thức đã cho cho một bước thời gian:
      CRPS = (1/N) * Σ_n |y_n - x| - (1/(2*N²)) * Σ_n Σ_m |y_n - y_m|
    x: giá trị quan sát (real), y: array 1D N mẫu dự báo.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = y.size
    if N == 0:
        return np.nan
    term1 = np.mean(np.abs(y - x))
    term2 = np.sum(np.abs(y[:, None] - y[None, :])) / (2.0 * (N ** 2))
    return float(term1 - term2)


def calculate_crps_with_formula(
    simulation_runs: np.ndarray,
    real_price_path: np.ndarray,
    time_increment: int,
    scoring_intervals: dict[str, int],
) -> float:
    """
    Tính tổng CRPS giống validator (cùng intervals, price changes) nhưng dùng công thức
    CRPS = (1/N)*Σ|y_n - x| - (1/(2*N²))*ΣΣ|y_n - y_m| thay cho crps_ensemble.
    """
    sum_all_scores = 0.0
    real_path = np.asarray(real_price_path, dtype=float).ravel()
    if np.any(simulation_runs == 0):
        return -1.0

    for interval_name, interval_seconds in scoring_intervals.items():
        interval_steps = get_interval_steps(interval_seconds, time_increment)
        absolute_price = interval_name.endswith("_abs")
        is_gap = interval_name.endswith("_gap")

        if absolute_price:
            while (
                real_path[::interval_steps].shape[0] == 1
                and interval_steps > 1
            ):
                interval_steps -= 1

        simulated_changes = calculate_price_changes_over_intervals(
            simulation_runs,
            interval_steps,
            absolute_price,
            is_gap,
        )
        real_changes = calculate_price_changes_over_intervals(
            real_path.reshape(1, -1),
            interval_steps,
            absolute_price,
            is_gap,
        )
        data_blocks = label_observed_blocks(real_changes[0])
        if len(data_blocks) == 0:
            continue

        for block in np.unique(data_blocks):
            if block == -1:
                continue
            mask = data_blocks == block
            sim_block = simulated_changes[:, mask]
            real_block = real_changes[0, mask]
            num_t = sim_block.shape[1]
            for t in range(num_t):
                x = float(real_block[t])
                y = sim_block[:, t]
                crps_t = crps_point_formula(x, y)
                if np.isfinite(crps_t):
                    if absolute_price:
                        crps_t = crps_t / real_path[-1] * 10_000
                    sum_all_scores += crps_t

    return float(sum_all_scores)


def get_prompt_config(prompt_label: str):
    """prompt_label: 'high' | 'low'."""
    if prompt_label.strip().lower() == "high":
        return HIGH_FREQUENCY
    return LOW_FREQUENCY


def get_all_strategy_names() -> list[str]:
    """Lấy tên tất cả chiến thuật: core (SIMULATOR_FUNCTIONS) + registry (strategies)."""
    from synth.miner.simulations_new_v2 import SIMULATOR_FUNCTIONS, _get_strategy_simulators

    names = set(SIMULATOR_FUNCTIONS.keys())
    names.add("garch_v2_2")
    names.add("equity_exact_hours")
    names.add("arima_equity")
    try:
        registry = __import__("synth.miner.strategies.registry", fromlist=["StrategyRegistry"]).StrategyRegistry()
        registry.auto_discover()
        for name in registry.list_all():
            names.add(name)
    except Exception:
        pass
    return sorted(names)


def get_simulate_fn(strategy_name: str) -> Optional[Callable]:
    """Trả về hàm simulate cho strategy_name (từ simulations_new_v2._build_simulator_functions)."""
    if strategy_name == "garch_v2_2":
        from synth.miner.core.garch_simulator_v2_2 import simulate_single_price_path_with_garch
        return simulate_single_price_path_with_garch
        
    if strategy_name == "equity_exact_hours":
        from synth.miner.core.equity_simulator import simulate_us_equity_exact
        return simulate_us_equity_exact
        
    if strategy_name == "arima_equity":
        from synth.miner.core.arima_equity_simulator import simulate_arima_us_equity_exact
        return simulate_arima_us_equity_exact
        
    from synth.miner.simulations_new_v2 import _get_simulate_fn
    return _get_simulate_fn(strategy_name)


def run_one_strategy(
    strategy_name: str,
    asset: str,
    start_time: datetime,
    prompt_config,
    data_handler: DataHandler,
    num_sims: int,
    seed: int,
    verbose: bool = False,
) -> tuple[Optional[float], str]:
    """
    Chạy một chiến thuật cho (asset, start_time, prompt_config).
    Trả về (crps, status) với status = "SUCCESS" | "FAIL" | "ERROR".
    """
    fn = get_simulate_fn(strategy_name)
    if fn is None:
        return None, "SKIP"

    start_iso = start_time.isoformat()
    if verbose:
        print(f"    [{strategy_name}] simulate ...", flush=True)
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
        try:
            paths = simulate_crypto_price_paths(
                current_price=None,
                asset=asset,
                start_time=start_iso,
                time_increment=prompt_config.time_increment,
                time_length=prompt_config.time_length,
                num_simulations=num_sims,
                simulate_fn=fn,
                max_data_points=None,
                seed=seed,
            )
        except Exception as e:
            if verbose:
                print(f"    [{strategy_name}] ERROR: {e}", flush=True)
            return None, "ERROR"

    if paths is None or not getattr(paths, "shape", (0,))[0]:
        if verbose:
            print(f"    [{strategy_name}] FAIL: no paths", flush=True)
        return None, "FAIL"

    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[0] == 0:
        return None, "FAIL"

    return paths, "OK"


def _time_frame_for_increment(time_increment: int) -> str:
    """1m cho 60s, 5m cho 300s."""
    return "1m" if time_increment == 60 else ("5m" if time_increment == 300 else str(time_increment))


def _parse_datetime_input(value: str) -> tuple[datetime, bool]:
    """
    Parse flexible datetime input.
    Returns (datetime_utc, has_time_component).
    Supported:
    - YYYY-MM-DD
    - YYYY-MM-DDTHH:MM[:SS][+TZ]
    - YYYY-MM-DD HH:MM[:SS]
    """
    s = (value or "").strip()
    if not s:
        raise ValueError("empty datetime input")
    has_time = ("T" in s) or (" " in s)
    if has_time:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    else:
        d = datetime.fromisoformat(f"{s}T00:00:00+00:00")
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d, has_time


def fetch_real_prices_from_db(
    data_handler: DataHandler,
    asset: str,
    start_time: datetime,
    prompt_config,
) -> Optional[np.ndarray]:
    """
    Lấy đường giá thực từ DB (MySQL price_data) cho (asset, start_time, prompt_config).
    Nếu cần 5m mà DB không có: load 1m rồi aggregate_1m_to_5m.
    Build path: start_ts, start_ts+inc, ...; với mỗi bước lấy giá từ DB (nearest past nếu không đúng ts).
    """
    time_increment = prompt_config.time_increment
    time_length = prompt_config.time_length
    time_frame = _time_frame_for_increment(time_increment)
    num_steps = time_length // time_increment + 1
    start_ts = int(start_time.timestamp())
    required_ts = [start_ts + k * time_increment for k in range(num_steps)]

    loaded = data_handler.load_price_data(asset, time_frame, load_from_file=False)
    if loaded and time_frame in loaded and loaded[time_frame]:
        prices_dict = loaded[time_frame]
    elif time_frame == "5m":
        # DB không có 5m → load 1m và aggregate sang 5m
        loaded_1m = data_handler.load_price_data(asset, "1m", load_from_file=False)
        if not loaded_1m or "1m" not in loaded_1m or not loaded_1m["1m"]:
            return None
        prices_dict = aggregate_1m_to_5m(loaded_1m["1m"])
        if not prices_dict:
            return None
    else:
        return None
    # prices_dict: { "timestamp_str": price, ... }
    sorted_ts = sorted(int(k) for k in prices_dict.keys())
    if not sorted_ts:
        return None

    path = []
    for ts in required_ts:
        if str(ts) in prices_dict:
            path.append(float(prices_dict[str(ts)]))
            continue
        # Nearest past: largest sorted_ts <= ts
        idx = np.searchsorted(sorted_ts, ts, side="right") - 1
        if idx < 0:
            path.append(float(prices_dict[str(sorted_ts[0])]))
        else:
            path.append(float(prices_dict[str(sorted_ts[idx])]))
    return np.array(path, dtype=float)


def fetch_real_prices(
    data_handler: DataHandler,
    asset: str,
    start_time: datetime,
    prompt_config,
    use_db: bool = True,
) -> Optional[np.ndarray]:
    """
    Lấy đường giá thực: use_db=True → từ DB (MySQL), use_db=False → Pyth API.
    """
    if use_db:
        return fetch_real_prices_from_db(data_handler, asset, start_time, prompt_config)
    vr = ValidatorRequest(
        asset=asset,
        start_time=start_time,
        time_length=prompt_config.time_length,
        time_increment=prompt_config.time_increment,
    )
    try:
        real = data_handler.get_real_prices(validator_request=vr)
        if real is None or len(real) == 0:
            return None
        return np.array(real, dtype=float)
    except Exception:
        return None


def _run_backtest_one_asset(
    asset: str,
    prompt_config,
    start_time: datetime,
    data_handler: DataHandler,
    strategy_names: list[str],
    num_sims: int,
    seed: int,
    use_db: bool,
    verbose: bool,
) -> tuple[list[dict], Optional[str]]:
    """Chạy backtest cho 1 đồng. Trả về (results, best_name)."""
    real_prices = fetch_real_prices(
        data_handler, asset, start_time, prompt_config, use_db=use_db
    )
    if real_prices is None or len(real_prices) == 0:
        return [], None
    results = []
    for name in strategy_names:
        paths, status = run_one_strategy(
            name, asset, start_time, prompt_config,
            data_handler, num_sims, seed, verbose=verbose,
        )
        if status != "OK" or paths is None:
            results.append({"name": name, "crps": None, "status": status})
            continue
        crps = calculate_crps_with_formula(
            paths,
            real_prices,
            prompt_config.time_increment,
            prompt_config.scoring_intervals,
        )
        if crps < 0 or not np.isfinite(crps):
            results.append({"name": name, "crps": None, "status": "CRPS_INVALID"})
        else:
            results.append({"name": name, "crps": float(crps), "status": "SUCCESS"})
    valid = [r for r in results if r["crps"] is not None and np.isfinite(r["crps"])]
    valid.sort(key=lambda x: x["crps"])
    best_name = valid[0]["name"] if valid else None
    return results, best_name


def main(
    asset: str | list[str],
    prompt_label: str,
    date: str | None,
    num_sims: int = 500,
    seed: int = 42,
    verbose: bool = False,
    output_dir: Optional[str] = None,
    use_db: bool = True,
    strategies_to_test: Optional[list[str]] = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """
    Backtest tất cả chiến thuật cho (asset hoặc nhiều đồng, prompt, date).
    asset: một str "BTC" hoặc list ["BTC", "ETH", "XAU", "SOL"].
    Trả về dict có per_asset, table (bảng CRPS theo strategy x asset), best_per_asset.
    """
    if start_date or end_date:
        # Run date range inclusive; keep output naming per day like before.
        s = (start_date or date or "").strip()
        e = (end_date or start_date or date or "").strip()
        if not s or not e:
            raise ValueError("start_date/end_date require date fallback, got empty range.")

        start_dt, _ = _parse_datetime_input(s)
        end_dt, _ = _parse_datetime_input(e)
        if start_dt > end_dt:
            raise ValueError("start_date must be <= end_date")

        per_day: dict[str, dict] = {}
        cur = start_dt
        while cur <= end_dt:
            day_str = cur.strftime("%Y-%m-%d")
            per_day[day_str] = main(
                asset=asset,
                prompt_label=prompt_label,
                date=day_str,
                num_sims=num_sims,
                seed=seed,
                verbose=verbose,
                output_dir=output_dir,
                use_db=use_db,
                strategies_to_test=strategies_to_test,
                start_date=None,
                end_date=None,
            )
            cur += timedelta(days=1)

        return {
            "date_range": {"start_date": s, "end_date": e},
            "per_day": per_day,
        }

    if date is None:
        raise ValueError("date is required when start_date/end_date not provided")

    if isinstance(asset, (list, tuple)):
        assets = [str(a).strip() for a in asset if str(a).strip()]
    else:
        assets = [str(asset).strip()]
    if not assets:
        assets = ["BTC"]

    prompt_label_norm = prompt_label.strip().lower()
    prompt_config = get_prompt_config(prompt_label_norm)
    is_high_mode = prompt_label_norm == "high"
    # Chỉ chạy đồng có trong prompt_config; đồng không đúng loại → cảnh báo và bỏ qua
    allowed = set(prompt_config.asset_list)
    skipped = [a for a in assets if a not in allowed]
    assets = [a for a in assets if a in allowed]
    for a in skipped:
        print(f"[WARN] Bỏ qua '{a}': không thuộc {prompt_label}.asset_list (cho phép: {sorted(allowed)}).")
    if not assets:
        print("[ERROR] Không còn đồng nào hợp lệ sau khi lọc theo prompt_config. Thoát.")
        return {"error": "no_valid_assets", "skipped": skipped, "per_asset": {}, "table": [], "best_per_asset": {}}

    has_time_input = False
    try:
        d, has_time_input = _parse_datetime_input(date.strip())
    except Exception:
        d = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_time = d if d.tzinfo else d.replace(tzinfo=timezone.utc)

    data_handler = DataHandler()
    strategy_names = get_all_strategy_names()

    # Lọc danh sách nếu test chỉ định
    if strategies_to_test:
        strategy_names = [name for name in strategy_names if name in strategies_to_test]
        if not strategy_names:
            print(f"[WARN] Không tìm thấy chiến thuật nào trong list {strategies_to_test}!")

    per_asset: dict[str, dict] = {}
    best_per_asset: dict[str, Optional[str]] = {}

    for a in assets:
        print(f"\n--- Asset: {a} (real prices từ {'DB' if use_db else 'Pyth'}) ---")
        if is_high_mode:
            hourly_rows: list[dict] = []
            # If caller passed date+time, run only that high slot.
            slot_starts = [start_time] if has_time_input else [start_time + timedelta(hours=hour) for hour in range(24)]
            for idx, slot_start in enumerate(slot_starts):
                print(f"  [HIGH] {a} @ {slot_start.strftime('%H:%M')} ...")
                results, best_name = _run_backtest_one_asset(
                    a, prompt_config, slot_start, data_handler,
                    strategy_names, num_sims, seed, use_db, verbose,
                )
                hourly_rows.append(
                    {
                        "hour": int(slot_start.hour),
                        "slot_index": idx,
                        "start_time": slot_start.isoformat(),
                        "strategies": results,
                        "best_strategy": best_name,
                    }
                )
            per_asset[a] = {"hourly": hourly_rows}
            best_per_asset[a] = None
            continue

        results, best_name = _run_backtest_one_asset(
            a, prompt_config, start_time, data_handler,
            strategy_names, num_sims, seed, use_db, verbose,
        )
        if not results and best_name is None:
            src = "DB (MySQL price_data)" if use_db else "Pyth API"
            print(f"[ERROR] {a}: Không lấy được real prices từ {src}. Bỏ qua.")
            best_per_asset[a] = None
            continue
        per_asset[a] = {"strategies": results, "best_strategy": best_name}
        best_per_asset[a] = best_name
        if verbose:
            for r in results:
                crps_str = f"{r['crps']:.4f}" if r["crps"] is not None else "—"
                print(f"  {r['name']:<22} {crps_str:>10} {r['status']}")

    table_rows: list[dict] = []
    if not is_high_mode:
        # Bảng tổng hợp: mỗi dòng = 1 strategy, mỗi cột = 1 asset (CRPS)
        all_strategies = sorted(set(r["name"] for data in per_asset.values() for r in data["strategies"]))
        for name in all_strategies:
            row: dict = {"strategy": name}
            for a in assets:
                data = per_asset.get(a)
                if not data:
                    row[a] = None
                    continue
                r = next((x for x in data["strategies"] if x["name"] == name), None)
                row[a] = r["crps"] if r and r.get("crps") is not None else None
            table_rows.append(row)

        # In bảng nhiều đồng
        col_w = 10
        header = f"{'Strategy':<22}"
        for a in assets:
            header += f" {a:>{col_w}}"
        header += "  Best?"
        print("\n" + "=" * (22 + (col_w + 1) * len(assets) + 8))
        print(f"BACKTEST STRATEGIES — assets={assets} prompt={prompt_label_norm} date={date}")
        print("=" * (22 + (col_w + 1) * len(assets) + 8))
        print(header)
        print("-" * (22 + (col_w + 1) * len(assets) + 8))
        for row in table_rows:
            name = row["strategy"]
            line = f"{name:<22}"
            wins = 0
            for a in assets:
                v = row.get(a)
                if v is not None and np.isfinite(v):
                    line += f" {v:>{col_w}.2f}"
                    if best_per_asset.get(a) == name:
                        wins += 1
                else:
                    line += f" {'—':>{col_w}}"
            line += f"  {wins}/{len(assets)}"
            print(line)
        print("-" * (22 + (col_w + 1) * len(assets) + 8))
        print("Best per asset:", best_per_asset)
        print("=" * (22 + (col_w + 1) * len(assets) + 8))

    out = {
        "prompt": prompt_label_norm,
        "date": date,
        "assets": assets,
        "start_time": start_time.isoformat(),
        "num_sims": num_sims,
        "seed": seed,
        "per_asset": (
            {
                a: {"hourly": per_asset[a]["hourly"]}
                for a in assets if a in per_asset
            }
            if is_high_mode
            else {
                a: {"strategies": per_asset[a]["strategies"], "best_strategy": per_asset[a]["best_strategy"]}
                for a in assets if a in per_asset
            }
        ),
        "table": table_rows,
        "best_per_asset": best_per_asset,
    }
    if output_dir:
        safe_date = date.replace("-", "_")
        # Lưu theo folder asset / date: output_dir / {asset} / {date} / file.json
        for a in assets:
            if a not in per_asset:
                continue
            dir_asset_date = os.path.join(output_dir, a, safe_date)
            os.makedirs(dir_asset_date, exist_ok=True)
            path_one = os.path.join(
                dir_asset_date,
                f"backtest_strategies_{a}_{prompt_label}_{safe_date}.json",
            )
            out_one = {
                "asset": a,
                "prompt": prompt_label_norm,
                "date": date,
                "start_time": start_time.isoformat(),
                "num_sims": num_sims,
                "seed": seed,
            }
            if is_high_mode:
                out_one["hourly"] = per_asset[a]["hourly"]
            else:
                out_one["strategies"] = per_asset[a]["strategies"]
                out_one["best_strategy"] = per_asset[a]["best_strategy"]
            with open(path_one, "w", encoding="utf-8") as f:
                json.dump(out_one, f, indent=2, ensure_ascii=False)
            print(f"Saved: {path_one}")
        # File tổng hợp (bảng nhiều đồng) lưu vào output_dir / multi / {date} /
        if len(assets) > 1:
            dir_multi = os.path.join(output_dir, "multi", safe_date)
            os.makedirs(dir_multi, exist_ok=True)
            path_multi = os.path.join(
                dir_multi,
                f"backtest_strategies_multi_{prompt_label}_{safe_date}.json",
            )
            with open(path_multi, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"Saved (multi): {path_multi}")
    return out


if __name__ == "__main__":
    # Chỉnh trực tiếp tham số ở đây rồi chạy:
    #   python -m synth.miner.backtest_strategies
    _ASSET = ["BTC"]
    _PROMPT = "high"  # "high" | "low"

    # Run single day (set _START_DATE/_END_DATE = None)
    # _DATE = "2026-03-18"

    # Run date range (inclusive). If set, it will iterate day-by-day and still save per-day files like before.
    _START_DATE: str = "2026-03-23 06:00:00" # e.g. "2026-03-18"
    _END_DATE: str = "2026-03-23 18:00:00" # e.g. "2026-03-20"

    _NUM_SIMS = 1000
    _SEED = 42
    _VERBOSE = False
    _OUTPUT_DIR = "result/backtest_strategies/shock"
    _USE_DB = True  # False = lấy real prices từ Pyth API

    main(
        asset=_ASSET,
        prompt_label=_PROMPT,
        date=None,
        start_date=_START_DATE,
        end_date=_END_DATE,
        num_sims=_NUM_SIMS,
        seed=_SEED,
        verbose=_VERBOSE,
        output_dir=_OUTPUT_DIR,
        use_db=_USE_DB,
    )

"""
backtest_compare_simulations.py

Backtest so sánh simulations.py vs simulations_new.py theo đúng lịch validator:
- High: 12 phút 1 lần (nến 1m, time_length=3600)
- Low: 60 phút 1 lần (nến 5m, time_length=86400)

Truyền vào 1 ngày để test. Thống kê: số lần thắng/thua giữa hai module.

Usage:
    from synth.miner.backtest_compare_simulations import main
    main(date="2026-03-03", assets=["BTC", "ETH"], num_sims=200)

    # CLI:
    python -m synth.miner.backtest_compare_simulations --date 2026-03-03 [--asset BTC] [--num-sims 200]
"""

import argparse
import contextlib
import io
import json
import os
import sys
from collections import defaultdict
from datetime import datetime as dt, timezone, timedelta

try:
    import matplotlib.pyplot as plt
except Exception:  # matplotlib không bắt buộc
    plt = None

from synth.validator.prompt_config import (
    LOW_FREQUENCY,
    HIGH_FREQUENCY,
    get_prompt_labels_for_asset,
)
from synth.miner.compute_score import cal_reward
from synth.miner.data_handler import DataHandler
from synth.db.models import ValidatorRequest
from synth.simulation_input import SimulationInput


def parse_date(date: str) -> dt:
    """Parse date string 'YYYY-MM-DD' -> datetime midnight UTC."""
    s = (date or "").strip()
    if not s:
        raise ValueError("date is required")
    if "T" in s or " " in s:
        d = dt.fromisoformat(s.replace("Z", "+00:00"))
    else:
        d = dt.fromisoformat(f"{s}T00:00:00+00:00")
    return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)


def get_high_request_slots_for_day(day_start: dt, asset: str) -> list[dt]:
    """Lịch high: mỗi 12 phút, time_length=3600. Trả về list start_time trong ngày."""
    if asset not in HIGH_FREQUENCY.asset_list:
        return []
    cycle_min = HIGH_FREQUENCY.total_cycle_minutes  # 12
    n_slots = (24 * 60) // cycle_min  # 120
    return [day_start + timedelta(minutes=k * cycle_min) for k in range(n_slots)]


def get_low_request_slots_for_day(day_start: dt, asset: str) -> list[dt]:
    """Lịch low: mỗi 60 phút, time_length=86400. Trả về list start_time trong ngày."""
    if asset not in LOW_FREQUENCY.asset_list:
        return []
    n_slots = 24
    return [day_start + timedelta(hours=k) for k in range(n_slots)]


def _ensure_1m_in_db(asset: str, start: dt, end: dt, data_handler: DataHandler) -> None:
    """Đảm bảo có dữ liệu 1m trong DB cho [start, end]. Dùng fetch_historical_data_backwards."""
    try:
        days_back = max(1, (end - start).days + 1)
        data_handler.fetch_historical_data_backwards(
            asset,
            end_time_utc=end,
            days_back=days_back,
            time_frame="1m",
        )
    except Exception as e:
        print(f"[backtest] ensure_1m_in_db {asset}: {e}", flush=True)


def _safe_json_load(path: str) -> dict | list | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_json_save(path: str, obj) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _slot_cache_key(asset: str, prompt: str, start_time_iso: str, time_length: int, time_increment: int) -> str:
    # start_time_iso có thể có +00:00; keep nguyên để ổn định key
    return f"{asset}|{prompt}|{start_time_iso}|L{int(time_length)}|I{int(time_increment)}"


def _rank_from_crps(my_crps: float | None, other_crps: list[float]) -> int | None:
    if my_crps is None:
        return None
    try:
        if my_crps == float("inf"):
            return None
        my = float(my_crps)
    except Exception:
        return None
    # Rank = 1 + số miner có CRPS thấp hơn (lower is better).
    better = 0
    for c in other_crps:
        if c < my:
            better += 1
    return better + 1


def add_db_ranking_and_topk_stats(
    result: dict,
    *,
    cache_path: str,
    refresh_cache: bool = False,
    min_crps: float = 0.0,
    db_time_increment_high: int | None = None,
    db_time_increment_low: int | None = None,
    auto_fallback_increment_300: bool = True,
) -> dict:
    """
    Với mỗi slot trong result["slots"], lấy CRPS của miner khác từ MySQL,
    tính rank của module_new/module_old và thống kê top10/top50/top100/out>200.

    Ghi cache CRPS list theo slot vào cache_path để reuse.
    """
    slots = result.get("slots") if isinstance(result, dict) else None
    if not isinstance(slots, dict):
        return result

    module_new = str(result.get("module_new", "module_new"))
    module_old = str(result.get("module_old", "module_old"))

    try:
        from synth.miner.mysql_handler import MySQLHandler  # local import (optional dep)
    except Exception as e:
        result["db_rank"] = {
            "cache_path": cache_path,
            "min_crps": float(min_crps),
            "error": f"Cannot import MySQLHandler: {e}",
        }
        return result

    # Load cache
    cache_obj = {} if refresh_cache else (_safe_json_load(cache_path) or {})
    if not isinstance(cache_obj, dict):
        cache_obj = {}

    try:
        db = MySQLHandler()
    except Exception as e:
        result["db_rank"] = {
            "cache_path": cache_path,
            "min_crps": float(min_crps),
            "error": f"Cannot init MySQLHandler: {e}",
        }
        return result

    def _init_stats():
        return {
            "requests": 0,
            "ranked": 0,
            "no_db": 0,
            "fail": 0,
            "top10": 0,
            "top50": 0,
            "top100": 0,
            "out_top200": 0,
        }

    stats_total = {
        module_new: _init_stats(),
        module_old: _init_stats(),
    }
    stats_by_prompt = {
        "high": {module_new: _init_stats(), module_old: _init_stats()},
        "low": {module_new: _init_stats(), module_old: _init_stats()},
    }

    def _bump(stats: dict, *, status: str, rank: int | None, has_db: bool):
        stats["requests"] += 1
        if not has_db:
            stats["no_db"] += 1
            return

        if status != "SUCCESS" or rank is None:
            stats["fail"] += 1
            # fail xem như ngoài top 200 (theo ý nghĩa "không đạt")
            stats["out_top200"] += 1
            return

        stats["ranked"] += 1
        if rank <= 10:
            stats["top10"] += 1
        if rank <= 50:
            stats["top50"] += 1
        if rank <= 100:
            stats["top100"] += 1
        if rank > 200:
            stats["out_top200"] += 1

    # Iterate slots and fill rank fields
    for prompt in ("high", "low"):
        series = slots.get(prompt) or []
        if not isinstance(series, list):
            continue

        time_length = HIGH_FREQUENCY.time_length if prompt == "high" else LOW_FREQUENCY.time_length
        time_increment_cfg = HIGH_FREQUENCY.time_increment if prompt == "high" else LOW_FREQUENCY.time_increment
        time_increment_override = db_time_increment_high if prompt == "high" else db_time_increment_low

        # Candidates in priority order
        candidates: list[int] = []
        if time_increment_override is not None:
            candidates.append(int(time_increment_override))
        candidates.append(int(time_increment_cfg))
        if auto_fallback_increment_300 and 300 not in candidates:
            candidates.append(300)
        # de-dup keep order
        seen_inc: set[int] = set()
        candidates = [x for x in candidates if not (x in seen_inc or seen_inc.add(x))]

        for s in series:
            if not isinstance(s, dict):
                continue
            asset = s.get("asset")
            start_time_iso = s.get("start_time")
            if not asset or not start_time_iso:
                continue

            other_crps: list[float] | None = None
            used_increment: int | None = None

            for inc in candidates:
                key = _slot_cache_key(str(asset), prompt, str(start_time_iso), time_length, inc)
                cached = cache_obj.get(key)
                if cached is not None:
                    if isinstance(cached, list) and cached:
                        other_crps = [float(x) for x in cached if isinstance(x, (int, float))]
                        used_increment = inc
                        break
                    # cached empty list → keep looking next candidate
                    continue

                try:
                    rows = db.get_validation_scores_for_slot(
                        asset=str(asset),
                        scored_time=str(start_time_iso),
                        time_length=int(time_length),
                        time_increment=int(inc),
                        min_crps=float(min_crps),
                    )
                except Exception:
                    rows = []

                crps_list: list[float] = []
                if isinstance(rows, list):
                    for r in rows:
                        try:
                            c = float(r.get("crps"))
                        except Exception:
                            continue
                        crps_list.append(c)

                cache_obj[key] = crps_list
                if crps_list:
                    other_crps = crps_list
                    used_increment = inc
                    break

            if other_crps is None:
                other_crps = []
            has_db = bool(other_crps)
            s["db_miners_count"] = len(other_crps)
            s["db_time_increment_used"] = used_increment
            s["db_time_increment_cfg"] = int(time_increment_cfg)

            # module_new
            rank_new = _rank_from_crps(s.get("crps_new"), other_crps) if has_db else None
            s["rank_new"] = rank_new
            _bump(stats_total[module_new], status=str(s.get("status_new")), rank=rank_new, has_db=has_db)
            _bump(stats_by_prompt[prompt][module_new], status=str(s.get("status_new")), rank=rank_new, has_db=has_db)

            # module_old
            rank_old = _rank_from_crps(s.get("crps_old"), other_crps) if has_db else None
            s["rank_old"] = rank_old
            _bump(stats_total[module_old], status=str(s.get("status_old")), rank=rank_old, has_db=has_db)
            _bump(stats_by_prompt[prompt][module_old], status=str(s.get("status_old")), rank=rank_old, has_db=has_db)

            # nếu cache rỗng, ghi dấu
            if not has_db:
                s["db_note"] = "NO_DB_SCORES"
            elif used_increment is not None and int(used_increment) != int(time_increment_cfg):
                s["db_note"] = f"DB_TIME_INCREMENT_MISMATCH_USED_{used_increment}"

    result["db_rank"] = {
        "cache_path": cache_path,
        "min_crps": float(min_crps),
        "db_time_increment_high": db_time_increment_high,
        "db_time_increment_low": db_time_increment_low,
        "auto_fallback_increment_300": bool(auto_fallback_increment_300),
        "stats_total": stats_total,
        "stats_by_prompt": stats_by_prompt,
    }

    # Save cache
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    _safe_json_save(cache_path, cache_obj)

    return result


def _run_generate_and_crps(
    module_name: str,
    asset: str,
    start_time: dt,
    prompt_config,
    data_handler: DataHandler,
    num_sims: int,
    seed: int,
    verbose: bool = False,
    log: bool = False,
) -> tuple[float | None, str]:
    """
    Gọi generate_simulations từ module, tính CRPS.
    Trả về (crps, status) với status = "SUCCESS" | "FAIL" | "ERROR".

    module_name có thể là:
      - "simulations"          → synth.miner.simulations
      - "simulations_new"      → synth.miner.simulations_new
      - "simulations_new_v2"   → synth.miner.simulations_new_v2
      - "simulations_new_v3"   → synth.miner.simulations_new_v3
      - "simulations_new_v4"   → synth.miner.simulations_new_v4
    """
    import synth.miner.simulations as sim_old
    import synth.miner.simulations_new as sim_new
    import synth.miner.simulations_new_v2 as sim_new_v2
    import synth.miner.simulations_new_v3 as sim_new_v3
    import synth.miner.simulations_new_v4 as sim_new_v4

    if log:
        print(f"    [{module_name}] generate_simulations ...", flush=True)
    _modules = {
        "simulations": sim_old,
        "simulations_new": sim_new,
        "simulations_new_v2": sim_new_v2,
        "simulations_new_v3": sim_new_v3,
        "simulations_new_v4": sim_new_v4,
    }
    module = _modules.get(module_name, sim_old)
    simulation_input = SimulationInput(
        asset=asset,
        start_time=start_time.isoformat(),
        time_increment=prompt_config.time_increment,
        time_length=prompt_config.time_length,
        num_simulations=num_sims,
    )

    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            result = module.generate_simulations(
                simulation_input=simulation_input,
                asset=asset,
                start_time=start_time.isoformat(),
                time_increment=prompt_config.time_increment,
                time_length=prompt_config.time_length,
                num_simulations=num_sims,
                seed=seed,
            )
        except Exception as e:
            if verbose or log:
                print(f"    [{module_name}] ERROR: {e}", flush=True)
            return None, "ERROR"

    predictions = result.get("predictions") if result else None
    if predictions is None:
        if log:
            print(f"    [{module_name}] FAIL: no predictions", flush=True)
        return None, "FAIL"

    if log:
        print(f"    [{module_name}] cal_reward ...", flush=True)
    validator_request = ValidatorRequest(
        asset=asset,
        start_time=start_time,
        time_length=prompt_config.time_length,
        time_increment=prompt_config.time_increment,
    )

    out2 = io.StringIO()
    err2 = io.StringIO()
    with contextlib.redirect_stdout(out2), contextlib.redirect_stderr(err2):
        crps, _, _, _ = cal_reward(data_handler, validator_request, predictions)

    if crps == -1:
        if log:
            print(f"    [{module_name}] cal_reward FAIL (crps=-1)", flush=True)
        return float("inf"), "FAIL"
    if log:
        print(f"    [{module_name}] done crps={crps:.2f}", flush=True)
    return float(crps), "SUCCESS"


def run_compare_day(
    date: str,
    assets: list[str],
    num_sims: int = 200,
    seed: int = 42,
    fetch_if_missing: bool = False,
    verbose: bool = False,
    max_high_slots: int | None = None,
    max_low_slots: int | None = None,
    log_slots: bool = True,
    module_new: str = "simulations_new",
    module_old: str = "simulations",
    progress_every_high: int = 30,
    progress_every_low: int = 6,
    config_type: str = "both",
) -> dict:
    """
    Chạy backtest 1 ngày: so sánh module_old vs module_new tại mỗi slot.

    module_new/module_old nằm trong:
      - "simulations"
      - "simulations_new"
      - "simulations_new_v2"
      - "simulations_new_v3"
      - "simulations_new_v4"

    Trả về thống kê: wins_new, wins_old, ties, fail_both, fail_new, fail_old.
    wins_new / fail_new tương ứng module_new, wins_old / fail_old tương ứng module_old.
    """
    day_start = parse_date(date)
    data_handler = DataHandler()

    if fetch_if_missing:
        for asset in assets:
            end = day_start + timedelta(days=1)
            start = day_start - timedelta(days=max(HIGH_FREQUENCY.window_days, LOW_FREQUENCY.window_days) + 5)
            _ensure_1m_in_db(asset, start, end, data_handler)

    stats = {
        "high": {"wins_new": 0, "wins_old": 0, "ties": 0, "fail_both": 0, "fail_new": 0, "fail_old": 0},
        "low": {"wins_new": 0, "wins_old": 0, "ties": 0, "fail_both": 0, "fail_new": 0, "fail_old": 0},
    }
    # Lưu chi tiết CRPS theo từng slot để dùng lại sau này
    slot_details: dict[str, list[dict]] = {"high": [], "low": []}
    total_high = 0
    total_low = 0

    run_high_global = config_type in ("both", "high")
    run_low_global = config_type in ("both", "low")

    for asset in assets:
        labels = get_prompt_labels_for_asset(asset)
        if not labels:
            continue
        run_high = run_high_global and ("high" in labels)
        run_low = run_low_global and ("low" in labels)
        high_slots = get_high_request_slots_for_day(day_start, asset) if run_high else []
        low_slots = get_low_request_slots_for_day(day_start, asset) if run_low else []
        if max_high_slots is not None:
            high_slots = high_slots[:max_high_slots]
        if max_low_slots is not None:
            low_slots = low_slots[:max_low_slots]

        for idx, start_time in enumerate(high_slots):
            total_high += 1
            if verbose or progress_every_high <= 1 or (idx + 1) % progress_every_high == 0 or idx == 0:
                print(f"  High {asset} slot {idx + 1}/{len(high_slots)} @ {start_time.strftime('%H:%M')} ...", flush=True)

            slot_log = log_slots and (verbose or progress_every_high <= 1 or (idx + 1) % progress_every_high == 0 or idx == 0)
            crps_new, status_new = _run_generate_and_crps(
                module_new, asset, start_time, HIGH_FREQUENCY,
                data_handler, num_sims, seed, verbose=verbose, log=slot_log,
            )
            crps_old, status_old = _run_generate_and_crps(
                module_old, asset, start_time, HIGH_FREQUENCY,
                data_handler, num_sims, seed, verbose=verbose, log=slot_log,
            )

            slot_details["high"].append(
                {
                    "asset": asset,
                    "prompt": "high",
                    "start_time": start_time.isoformat(),
                    "module_new": module_new,
                    "module_old": module_old,
                    "crps_new": crps_new,
                    "status_new": status_new,
                    "crps_old": crps_old,
                    "status_old": status_old,
                }
            )

            if status_new != "SUCCESS" and status_old != "SUCCESS":
                stats["high"]["fail_both"] += 1
            elif status_new != "SUCCESS":
                stats["high"]["fail_new"] += 1
            elif status_old != "SUCCESS":
                stats["high"]["fail_old"] += 1
            elif crps_new < crps_old:
                stats["high"]["wins_new"] += 1
            elif crps_old < crps_new:
                stats["high"]["wins_old"] += 1
            else:
                stats["high"]["ties"] += 1

        for idx, start_time in enumerate(low_slots):
            total_low += 1
            if verbose or progress_every_low <= 1 or (idx + 1) % progress_every_low == 0 or idx == 0:
                print(f"  Low  {asset} slot {idx + 1}/{len(low_slots)} @ {start_time.strftime('%H:%M')} ...", flush=True)

            slot_log = log_slots and (verbose or progress_every_low <= 1 or (idx + 1) % progress_every_low == 0 or idx == 0)
            crps_new, status_new = _run_generate_and_crps(
                module_new, asset, start_time, LOW_FREQUENCY,
                data_handler, num_sims, seed, verbose=verbose, log=slot_log,
            )
            crps_old, status_old = _run_generate_and_crps(
                module_old, asset, start_time, LOW_FREQUENCY,
                data_handler, num_sims, seed, verbose=verbose, log=slot_log,
            )

            slot_details["low"].append(
                {
                    "asset": asset,
                    "prompt": "low",
                    "start_time": start_time.isoformat(),
                    "module_new": module_new,
                    "module_old": module_old,
                    "crps_new": crps_new,
                    "status_new": status_new,
                    "crps_old": crps_old,
                    "status_old": status_old,
                }
            )

            if status_new != "SUCCESS" and status_old != "SUCCESS":
                stats["low"]["fail_both"] += 1
            elif status_new != "SUCCESS":
                stats["low"]["fail_new"] += 1
            elif status_old != "SUCCESS":
                stats["low"]["fail_old"] += 1
            elif crps_new < crps_old:
                stats["low"]["wins_new"] += 1
            elif crps_old < crps_new:
                stats["low"]["wins_old"] += 1
            else:
                stats["low"]["ties"] += 1

    return {
        "date": date,
        "assets": assets,
        "num_sims": num_sims,
        "seed": seed,
        "module_new": module_new,
        "module_old": module_old,
        "total_high": total_high,
        "total_low": total_low,
        "high": stats["high"],
        "low": stats["low"],
        "slots": slot_details,
    }


def main(
    date: str = "2026-03-03",
    assets: list[str] | None = None,
    num_sims: int = 200,
    seed: int = 42,
    fetch_if_missing: bool = False,
    verbose: bool = False,
    output_dir: str | None = None,
    max_high_slots: int | None = None,
    max_low_slots: int | None = None,
    log_slots: bool = True,
    module_new: str = "simulations_new",
    module_old: str = "simulations",
    plot_crps: bool = False,
    reuse_if_exists: bool = True,
    rank_vs_db: bool = True,
    refresh_db_cache: bool = False,
    progress_every_high: int = 30,
    progress_every_low: int = 6,
    db_high_increment: int | None = None,
    db_low_increment: int | None = None,
    db_auto_fallback_300: bool = True,
    config_type: str = "both",
) -> dict:
    """
    Entry point: backtest so sánh module_old vs module_new cho 1 ngày.

    Args:
        date: Ngày test (YYYY-MM-DD).
        assets: Danh sách đồng; None = tất cả asset.
        num_sims: Số path mô phỏng mỗi request.
        seed: Random seed.
        fetch_if_missing: Nếu True thì fetch dữ liệu thiếu vào DB.
        verbose: In chi tiết từng request.
        output_dir: Thư mục lưu JSON; None = result/compare_simulations.
        max_high_slots: Giới hạn số slot high/asset (None = tất cả ~120).
        max_low_slots: Giới hạn số slot low/asset (None = tất cả ~24).
        log_slots: In log chi tiết (generate_simulations, cal_reward) cho mỗi slot.
        module_new: Module mới (ví dụ: simulations_new, simulations_new_v2).
        module_old: Module baseline (ví dụ: simulations).
    """
    if assets is None:
        assets = list(set(HIGH_FREQUENCY.asset_list) | set(LOW_FREQUENCY.asset_list))
    assets = [a for a in assets if isinstance(a, str) and a.strip()]
    if not assets:
        assets = ["BTC", "ETH", "XAU", "SOL"]

    config_type = (config_type or "both").strip().lower()
    if config_type not in {"both", "high", "low"}:
        raise ValueError("config_type must be one of: both, high, low")

    day_start = parse_date(date)
    n_high = len(get_high_request_slots_for_day(day_start, assets[0])) if assets else 0
    n_low = len(get_low_request_slots_for_day(day_start, assets[0])) if assets else 0

    print("=" * 70)
    print(f"Backtest so sánh: {module_old} vs {module_new} — date={date}")
    print(f"  Assets: {assets}")
    print(f"  Config type: {config_type}")
    if config_type in {"both", "high"}:
        print(f"  High: 12m cycle, ~{n_high} requests/asset/day (nến 1m)")
    if config_type in {"both", "low"}:
        print(f"  Low:  60m cycle, ~{n_low} requests/asset/day (nến 5m)")
    print(f"  num_sims={num_sims}, seed={seed}")
    print("=" * 70)

    # Output path (dùng cho cả cache + lưu mới)
    out_dir = output_dir or os.path.join("result", "compare_simulations")
    os.makedirs(out_dir, exist_ok=True)
    safe_date = date.replace("-", "_")
    suffix = f"{module_new}_vs_{module_old}"
    path = os.path.join(out_dir, f"compare_{suffix}_{safe_date}.json")
    db_cache_path = os.path.join(out_dir, f"db_crps_cache_{safe_date}.json")

    used_cache = False
    if reuse_if_exists and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
        used_cache = True
        print(f"\nDùng lại kết quả đã lưu: {path}")
        cached_assets = result.get("assets")
        if isinstance(cached_assets, list) and cached_assets != assets:
            print(f"  (Lưu ý: assets hiện tại={assets} nhưng file cache assets={cached_assets})")
    else:
        result = run_compare_day(
            date=date,
            assets=assets,
            num_sims=num_sims,
            seed=seed,
            fetch_if_missing=fetch_if_missing,
            verbose=verbose,
            max_high_slots=max_high_slots,
            max_low_slots=max_low_slots,
            log_slots=log_slots,
            module_new=module_new,
            module_old=module_old,
            progress_every_high=progress_every_high,
            progress_every_low=progress_every_low,
            config_type=config_type,
        )

    # Rank vs DB + thống kê top-k
    if rank_vs_db:
        result = add_db_ranking_and_topk_stats(
            result,
            cache_path=db_cache_path,
            refresh_cache=refresh_db_cache,
            db_time_increment_high=db_high_increment,
            db_time_increment_low=db_low_increment,
            auto_fallback_increment_300=db_auto_fallback_300,
        )

    # In thống kê
    def _print_stats(label: str, s: dict, total: int):
        if total == 0:
            return
        print(f"\n--- {label} (tổng {total} requests) ---")
        print(f"  {module_new} thắng: {s['wins_new']} ({100*s['wins_new']/total:.1f}%)")
        print(f"  {module_old} thắng: {s['wins_old']} ({100*s['wins_old']/total:.1f}%)")
        print(f"  Hòa:                  {s['ties']} ({100*s['ties']/total:.1f}%)")
        print(f"  Fail cả hai:          {s['fail_both']}")
        print(f"  Fail {module_new}: {s['fail_new']}")
        print(f"  Fail {module_old}:     {s['fail_old']}")

    _print_stats("HIGH", result["high"], result["total_high"])
    _print_stats("LOW", result["low"], result["total_low"])

    total = result["total_high"] + result["total_low"]
    if total > 0:
        wins_new = result["high"]["wins_new"] + result["low"]["wins_new"]
        wins_old = result["high"]["wins_old"] + result["low"]["wins_old"]
        print(f"\n=== TỔNG ({total} requests) ===")
        print(f"  {module_new} thắng: {wins_new} ({100*wins_new/total:.1f}%)")
        print(f"  {module_old} thắng:    {wins_old} ({100*wins_old/total:.1f}%)")

    # In thêm thống kê rank nếu có
    if isinstance(result.get("db_rank"), dict):
        st = result["db_rank"].get("stats_total") or {}
        if isinstance(st, dict):
            print("\n=== RANK vs DB (tổng) ===")
            for m in (module_new, module_old):
                x = st.get(m) or {}
                if not isinstance(x, dict):
                    continue
                print(
                    f"  {m:<22} "
                    f"top10={x.get('top10', 0):>4}  "
                    f"top50={x.get('top50', 0):>4}  "
                    f"top100={x.get('top100', 0):>4}  "
                    f"out>200={x.get('out_top200', 0):>4}  "
                    f"fail={x.get('fail', 0):>4}  "
                    f"no_db={x.get('no_db', 0):>4}"
                )

    # Lưu JSON (lưu cả khi dùng cache nếu vừa bổ sung db_rank)
    should_save = (not used_cache) or (rank_vs_db and isinstance(result.get("db_rank"), dict))
    if should_save:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nKết quả: {path}")

    # Plot CRPS theo thời gian nếu được yêu cầu
    if plot_crps and plt is not None and "slots" in result:
        slots = result["slots"]

        def _ensure_plot_dir(prompt: str, asset: str | None = None) -> str:
            """
            Cấu trúc thư mục:
              - Plot tổng theo prompt: {out_dir}/{prompt}/{date}/
              - Plot theo asset:       {out_dir}/{asset.lower()}/{prompt}/{date}/
            """
            parts: list[str] = [out_dir]
            if asset:
                parts.append(str(asset).lower())
            parts.append(str(prompt))
            parts.append(str(date))
            d = os.path.join(*parts)
            os.makedirs(d, exist_ok=True)
            return d

        def _plot_series(series: list[dict], title: str, png_path: str):
            times = [dt.fromisoformat(s["start_time"]) for s in series]
            crps_new = [
                s["crps_new"] if s.get("status_new") == "SUCCESS" else None
                for s in series
            ]
            crps_old = [
                s["crps_old"] if s.get("status_old") == "SUCCESS" else None
                for s in series
            ]

            times_f, new_f, old_f = [], [], []
            for t, cn, co in zip(times, crps_new, crps_old):
                if cn is not None and co is not None:
                    times_f.append(t)
                    new_f.append(cn)
                    old_f.append(co)
            if not times_f:
                return False

            plt.figure(figsize=(11, 4))
            # Với HIGH (nhiều điểm) tránh marker để đỡ rối
            plt.plot(times_f, new_f, label=module_new, linewidth=1.1, alpha=0.9)
            plt.plot(times_f, old_f, label=module_old, linewidth=1.1, alpha=0.9)
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("CRPS (lower is better)")
            plt.legend()
            plt.grid(True, alpha=0.25)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close()
            return True

        # 1) Plot tổng theo prompt (high/low)
        for prompt in ("high", "low"):
            series = slots.get(prompt) or []
            if not series:
                continue
            plot_dir = _ensure_plot_dir(prompt)
            png_path = os.path.join(plot_dir, f"crps_{prompt}_{suffix}.png")
            ok = _plot_series(
                series,
                title=f"CRPS {prompt.upper()} — {module_new} vs {module_old} — {date}",
                png_path=png_path,
            )
            if ok:
                print(f"Đã lưu biểu đồ CRPS {prompt} tại: {png_path}")

        # 2) Plot theo từng asset
        for prompt in ("high", "low"):
            series = slots.get(prompt) or []
            if not series:
                continue

            assets_in_series = sorted({s.get("asset") for s in series if s.get("asset")})
            for a in assets_in_series:
                sub = [s for s in series if s.get("asset") == a]
                if not sub:
                    continue
                plot_dir = _ensure_plot_dir(prompt, asset=a)
                png_path = os.path.join(plot_dir, f"crps_{prompt}_{a}_{suffix}.png")
                ok = _plot_series(
                    sub,
                    title=f"CRPS {prompt.upper()} {a} — {module_new} vs {module_old} — {date}",
                    png_path=png_path,
                )
                if ok:
                    print(f"Đã lưu biểu đồ CRPS {prompt} {a} tại: {png_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="So sánh hai module simulations theo lịch validator")
    parser.add_argument("--date", type=str, default="2026-03-03", help="Ngày test YYYY-MM-DD")
    parser.add_argument("--asset", type=str, action="append", dest="assets", help="Đồng test (lặp được)")
    parser.add_argument("--num-sims", type=int, default=200, help="Số path mô phỏng mỗi lần")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fetch", action="store_true", dest="fetch_if_missing", help="Fetch thiếu data vào DB")
    parser.add_argument("--verbose", action="store_true", help="In chi tiết từng request")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-high", type=int, default=None, help="Giới hạn slot high/asset (test nhanh)")
    parser.add_argument("--max-low", type=int, default=None, help="Giới hạn slot low/asset (test nhanh)")
    parser.add_argument("--progress-high", type=int, default=30, help="In progress mỗi N slot HIGH (1 = mỗi slot)")
    parser.add_argument("--progress-low", type=int, default=6, help="In progress mỗi N slot LOW (1 = mỗi slot)")
    parser.add_argument("--no-log", action="store_true", dest="no_log", help="Tắt log chi tiết từng bước")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bỏ qua cache JSON nếu đã có, chạy lại backtest",
    )
    parser.add_argument(
        "--no-rank-db",
        action="store_true",
        help="Không query DB để rank vs miners khác",
    )
    parser.add_argument(
        "--refresh-db-cache",
        action="store_true",
        help="Bỏ cache DB CRPS và query lại",
    )
    parser.add_argument(
        "--db-high-increment",
        type=int,
        default=None,
        help="Override time_increment khi query DB cho HIGH (vd 300). None = theo config + fallback.",
    )
    parser.add_argument(
        "--db-low-increment",
        type=int,
        default=None,
        help="Override time_increment khi query DB cho LOW. None = theo config + fallback.",
    )
    parser.add_argument(
        "--no-db-fallback-300",
        action="store_true",
        help="Tắt fallback thử time_increment=300 khi query DB",
    )
    parser.add_argument(
        "--config-type",
        type=str,
        default="both",
        choices=["both", "high", "low"],
        help="Chọn loại config để chạy: both | high | low",
    )
    parser.add_argument(
        "--module-new",
        type=str,
        default="simulations_new_v4",
        choices=["simulations", "simulations_new", "simulations_new_v2", "simulations_new_v3", "simulations_new_v4"],
        help="Module mới để so sánh (default: simulations_new_v4)",
    )
    parser.add_argument(
        "--module-old",
        type=str,
        default="simulations",
        choices=["simulations", "simulations_new", "simulations_new_v2", "simulations_new_v3", "simulations_new_v4"],
        help="Module baseline để so sánh (default: simulations)",
    )
    parser.add_argument(
        "--plot-crps",
        action="store_true",
        help="Vẽ và lưu biểu đồ CRPS theo thời gian cho từng prompt (high/low)",
    )
    args = parser.parse_args()

    main(
        date="2026-03-22",
        assets=["XAU"],
        num_sims=1000,
        config_type="low",
        seed=42,
        fetch_if_missing=False,
        verbose=False,
        output_dir=None,
        max_high_slots=None,
        max_low_slots=None,
        log_slots=True,
        module_new="simulations_new_v3",
        module_old="simulations_new_v4",
        plot_crps=False,
        reuse_if_exists=False,
        rank_vs_db=True,
        refresh_db_cache=False,
    )

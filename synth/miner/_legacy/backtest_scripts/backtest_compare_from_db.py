"""
backtest_compare_from_db.py

So sánh hai simulations (module_old vs module_new) giống
`backtest_compare_simulations.py` nhưng:
- Lấy các request từ chính `scored_time` đã lưu CRPS trong MySQL.
- Chỉ chạy tại những thời điểm mà DB đã có CRPS của miner khác.
- Với mỗi request: chạy 2 simulations → CRPS → xếp hạng so với CRPS trong DB.
- Thống kê:
    + Số lần Top 10, Top 50, Top 100, và out > 200
    + Số lần fail (sim không ra CRPS hợp lệ)
    + Số lần không có dữ liệu DB (no_db)

Cách dùng đơn giản:
    - Mở file này
    - Sửa tham số ở cuối file (khối `if __name__ == "__main__":`)
    - Chạy:
          python synth/miner/backtest_compare_from_db.py
"""

from __future__ import annotations

import contextlib
import io
import json
import hashlib
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from synth.miner.backtest.runner import _get_prompt_config
from synth.miner.compute_score import cal_reward
from synth.miner.data_handler import DataHandler
from synth.miner.mysql_handler import MySQLHandler
from synth.miner.synthdata_client import _iso as iso_format  # type: ignore
from synth.simulation_input import SimulationInput
from synth.db.models import ValidatorRequest
from synth.utils.helpers import adjust_predictions
from synth.validator import prompt_config as v_prompt_config
from synth.validator.crps_calculation import (
    get_interval_steps,
    calculate_price_changes_over_intervals,
    label_observed_blocks,
)


_COMPARE_MINER_UID = 95


def _crps_point_formula(x: float, y: np.ndarray) -> float:
    """
    CRPS theo công thức: (1/N)*Σ|y_n - x| - (1/(2*N²))*ΣΣ|y_n - y_m|
    x = quan sát, y = array N mẫu dự báo.
    """
    y = np.asarray(y, dtype=float).ravel()
    N = y.size
    if N == 0:
        return np.nan
    term1 = np.mean(np.abs(y - x))
    term2 = np.sum(np.abs(y[:, None] - y[None, :])) / (2.0 * (N ** 2))
    return float(term1 - term2)


def _calculate_crps_with_formula(
    simulation_runs: np.ndarray,
    real_price_path: np.ndarray,
    time_increment: int,
    scoring_intervals: dict[str, int],
) -> float:
    """
    Tính tổng CRPS dùng công thức (1/N)*Σ|y_n - x| - (1/(2*N²))*ΣΣ|y_n - y_m|
    Cùng intervals/price changes như validator, không dùng properscoring.crps_ensemble.
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
                crps_t = _crps_point_formula(x, y)
                if np.isfinite(crps_t):
                    if absolute_price:
                        crps_t = crps_t / real_path[-1] * 10_000
                    sum_all_scores += crps_t

    return float(sum_all_scores)


def _get_scoring_intervals(time_length: int) -> dict:
    """Lấy scoring_intervals theo time_length (high 3600 vs low 86400)."""
    if time_length == v_prompt_config.HIGH_FREQUENCY.time_length:
        return v_prompt_config.HIGH_FREQUENCY.scoring_intervals
    return v_prompt_config.LOW_FREQUENCY.scoring_intervals


@dataclass
class SlotRankResult:
    asset: str
    scored_time: str
    time_length: int
    time_increment: int

    # Miner trong DB để so sánh
    miner_uid: int
    crps_miner: float
    rank_miner: int
    status_miner: str

    total_miners: int

    # Real price summary (để debug mệnh giá)
    real_price_start: float | None = None
    real_price_end: float | None = None


def _run_generate_and_crps_for_time(
    module_name: str,
    asset: str,
    scored_time: datetime,
    time_length: int,
    time_increment: int,
    data_handler: DataHandler,
    num_sims: int,
    seed: int,
) -> Tuple[float | None, str, float | None, float | None]:
    """
    Gọi generate_simulations cho 1 module tại đúng scored_time/time_length/time_increment
    rồi tính CRPS qua cal_reward (giống validator).
    """
    import synth.miner.simulations as sim_old
    import synth.miner.simulations_new as sim_new
    import synth.miner.simulations_new_v2 as sim_new_v2
    import synth.miner.simulations_new_stable as sim_new_stable

    modules = {
        "simulations": sim_old,
        "simulations_new": sim_new,
        "simulations_new_v2": sim_new_v2,
        "simulations_new_stable": sim_new_stable,
    }
    module = modules.get(module_name, sim_old)

    start_dt = scored_time - timedelta(seconds=time_length)
    start_time = start_dt.isoformat()

    sim_input = SimulationInput(
        asset=asset,
        start_time=start_time,
        time_increment=time_increment,
        time_length=time_length,
        num_simulations=num_sims,
    )

    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            result = module.generate_simulations(
                simulation_input=sim_input,
                asset=asset,
                start_time=start_time,
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=num_sims,
                seed=seed,
            )
        except Exception:
            return None, "ERROR"

    predictions = result.get("predictions") if result else None
    if predictions is None:
        return None, "FAIL"

    validator_request = ValidatorRequest(
        asset=asset,
        start_time=scored_time,
        time_length=time_length,
        time_increment=time_increment,
    )

    out2 = io.StringIO()
    err2 = io.StringIO()
    with contextlib.redirect_stdout(out2), contextlib.redirect_stderr(err2):
        # Lấy real_prices (cal_reward gọi get_real_prices); dùng để tính CRPS theo công thức giống backtest_strategies
        _, _, _, real_prices = cal_reward(data_handler, validator_request, predictions)

    if real_prices is None or len(real_prices) == 0:
        return float("inf"), "FAIL", None, None

    # Chuyển predictions sang (n_paths, n_steps) và tính CRPS bằng công thức, không dùng validator
    adjusted = adjust_predictions(list(predictions))
    if not adjusted:
        return float("inf"), "FAIL", None, None
    simulation_runs = np.array(adjusted).astype(float)
    if simulation_runs.ndim != 2 or simulation_runs.shape[0] == 0:
        return float("inf"), "FAIL", None, None

    scoring_intervals = _get_scoring_intervals(time_length)
    crps = _calculate_crps_with_formula(
        simulation_runs,
        np.array(real_prices),
        time_increment,
        scoring_intervals,
    )
    if crps < 0 or not math.isfinite(crps):
        return float("inf"), "FAIL", None, None

    rp_start = None
    rp_end = None
    try:
        if isinstance(real_prices, list) and len(real_prices) > 0:
            rp_start = float(real_prices[0])
            rp_end = float(real_prices[-1])
    except Exception:
        rp_start, rp_end = None, None

    return float(crps), "SUCCESS", rp_start, rp_end


def _get_scored_times_for_day(
    mysql: MySQLHandler,
    asset: str,
    day: datetime,
) -> List[Tuple[datetime, int]]:
    """
    Lấy tất cả (scored_time, time_length) trong DB cho 1 asset
    trong khoảng [day, day+1).

    Lưu ý: KHÔNG dùng time_increment trong DB (DB có thể lưu sai inc=300 cho tl=3600).
    """
    table = mysql._val_table(asset)  # type: ignore[attr-defined]
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    conn = mysql._get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT scored_time, time_length
                FROM {table}
                WHERE scored_time >= %s AND scored_time < %s
                  AND crps > 0
                ORDER BY scored_time ASC
                """,
                (start, end),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    out: List[Tuple[datetime, int]] = []
    for st, tl in rows:
        if isinstance(st, datetime):
            out.append((st, int(tl)))
    return out


def _rank_against_db(
    mysql: MySQLHandler,
    asset: str,
    scored_time: datetime,
    time_length: int,
    crps_value: float | None,
) -> Tuple[int, int]:
    """
    So sánh 1 CRPS với tất cả miner trong DB cho cùng (asset, scored_time, time_length).
    Trả về (rank, total_miners) với rank=1 là tốt nhất; -1 nếu không xếp hạng được.
    CRPS <= 0 hoặc không finite (NaN, inf) coi là invalid → rank -1.
    """
    if (
        crps_value is None
        or crps_value == float("inf")
        or not math.isfinite(crps_value)
        or crps_value <= 0
    ):
        return -1, 0

    st = scored_time.strftime("%Y-%m-%d %H:%M:%S")
    rows = mysql.get_validation_scores(asset, st, st, time_length=time_length)
    crps_list = [r["crps"] for r in rows if r.get("crps") is not None and r["crps"] > 0]
    if not crps_list:
        return -1, 0

    total = len(crps_list)
    # Rank 1 = tốt nhất (CRPS nhỏ nhất). Nếu CRPS của simulation tốt hơn tất cả miner,
    # rank sẽ = 1; nếu tệ hơn tất cả, rank = total.
    rank = sum(1 for c in crps_list if c < crps_value) + 1
    return rank, total


def _rank_miner_uid_against_db(
    mysql: MySQLHandler,
    asset: str,
    scored_time: datetime,
    time_length: int,
    miner_uid: int,
) -> Tuple[int, int, float | None]:
    """
    Lấy CRPS của miner_uid trong DB tại đúng scored_time, và tính rank của miner đó
    so với các miner khác trong cùng (asset, scored_time, time_length).

    Lưu ý: Theo yêu cầu, KHÔNG quan tâm time_increment trong DB.

    Returns: (rank_miner, total_miners, crps_miner)
      - rank_miner = -1 nếu không có dữ liệu / không rank được
      - total_miners = 0 nếu không có dữ liệu DB
      - crps_miner = None nếu không tìm thấy miner_uid
    """
    st = scored_time.strftime("%Y-%m-%d %H:%M:%S")
    rows = mysql.get_validation_scores(asset, st, st, time_length=int(time_length))

    crps_list: list[float] = []
    crps_miner: float | None = None

    for r in rows:
        try:
            crps = float(r.get("crps"))
        except Exception:
            continue
        if not math.isfinite(crps) or crps <= 0:
            continue
        crps_list.append(crps)
        if int(r.get("miner_uid", -1)) == int(miner_uid):
            crps_miner = crps

    if not crps_list:
        return -1, 0, None
    if crps_miner is None:
        return -1, len(crps_list), None

    rank_miner = sum(1 for c in crps_list if c < crps_miner) + 1
    return int(rank_miner), int(len(crps_list)), float(crps_miner)


def _plot_ranks_for_day(result: Dict[str, Any]) -> None:
    """
    Vẽ và lưu biểu đồ rank high/low (2 subplot) cho miner_uid trên cùng hình.
    File PNG sẽ được lưu cạnh file JSON trong result/compare_from_db/{asset}.
    """
    asset = result.get("asset", "?")
    date = result.get("date", "?")
    miner_uid = int(result.get("miner_uid", _COMPARE_MINER_UID))

    per_slot: List[Dict[str, Any]] = result.get("per_slot", [])

    # Load price series from DB once (fallback 1m→5m for low)
    data_handler = DataHandler()
    safe_asset = str(asset)
    prices_1m: dict[str, float] = {}
    prices_5m: dict[str, float] = {}
    try:
        loaded_1m = data_handler.load_price_data(safe_asset, "1m", load_from_file=False)
        if isinstance(loaded_1m, dict) and "1m" in loaded_1m and loaded_1m["1m"]:
            prices_1m = {str(k): float(v) for k, v in loaded_1m["1m"].items()}
    except Exception:
        prices_1m = {}
    try:
        loaded_5m = data_handler.load_price_data(safe_asset, "5m", load_from_file=False)
        if isinstance(loaded_5m, dict) and "5m" in loaded_5m and loaded_5m["5m"]:
            prices_5m = {str(k): float(v) for k, v in loaded_5m["5m"].items()}
        elif prices_1m:
            from synth.miner.price_aggregation import aggregate_1m_to_5m
            prices_5m = aggregate_1m_to_5m(prices_1m)
    except Exception:
        if prices_1m:
            try:
                from synth.miner.price_aggregation import aggregate_1m_to_5m
                prices_5m = aggregate_1m_to_5m(prices_1m)
            except Exception:
                prices_5m = {}

    def _nearest_past_price(prices: dict[str, float], ts: int) -> float | None:
        if not prices:
            return None
        # prefer exact match
        if str(ts) in prices:
            try:
                return float(prices[str(ts)])
            except Exception:
                return None
        # nearest past
        try:
            keys = sorted(int(k) for k in prices.keys())
        except Exception:
            return None
        if not keys:
            return None
        idx = int(np.searchsorted(keys, ts, side="right")) - 1
        if idx < 0:
            idx = 0
        return float(prices.get(str(keys[idx])))

    times_high: List[datetime] = []
    ranks_miner_high: List[int] = []
    real_end_high: List[float] = []

    times_low: List[datetime] = []
    ranks_miner_low: List[int] = []
    real_end_low: List[float] = []

    for s in per_slot:
        try:
            st = datetime.fromisoformat(str(s["scored_time"]).replace("Z", "+00:00"))
        except Exception:
            continue

        tl = int(s.get("time_length", 0))
        rm = int(s.get("rank_miner", -1))
        status_miner = s.get("status_miner")
        # Chỉ vẽ các slot có rank hợp lệ (>0) và miner có rank OK.
        if rm <= 0 or status_miner != "OK":
            continue
        # price at scored_time (from DB)
        ts = int(st.timestamp())
        rp_end_f = None
        if tl == 3600:
            rp_end_f = _nearest_past_price(prices_1m, ts)
        elif tl == 86400:
            rp_end_f = _nearest_past_price(prices_5m, ts)

        if tl == 3600:
            times_high.append(st)
            ranks_miner_high.append(rm)
            if rp_end_f is not None:
                real_end_high.append(rp_end_f)
        elif tl == 86400:
            times_low.append(st)
            ranks_miner_low.append(rm)
            if rp_end_f is not None:
                real_end_low.append(rp_end_f)

    n_rows = int(bool(times_high)) + int(bool(times_low))
    if n_rows == 0:
        print("[from_db][plot] Không có slot hợp lệ để vẽ.")
        return

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]  # type: ignore[list-item]

    idx = 0
    max_rank_seen = 0

    if times_high:
        max_rank_seen = max(max_rank_seen, max(ranks_miner_high))
        ax_high = axes[idx]
        ax_high.plot(times_high, ranks_miner_high, "o-", label=f"miner:{miner_uid}", alpha=0.85)
        ax2 = None
        if len(real_end_high) == len(times_high) and len(real_end_high) > 0:
            ax2 = ax_high.twinx()
            ax2.plot(times_high, real_end_high, "-", color="gray", alpha=0.35, label="real_price")
            ax2.set_ylabel("Real price")
        ax_high.set_title(f"HIGH (tl=3600) — {asset} on {date}")
        ax_high.set_ylabel("Rank")
        ax_high.grid(True, alpha=0.3)
        lines, labels = ax_high.get_legend_handles_labels()
        if ax2 is not None:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax_high.legend(lines + lines2, labels + labels2, loc="best")
        else:
            ax_high.legend(loc="best")
        idx += 1

    if times_low:
        max_rank_seen = max(max_rank_seen, max(ranks_miner_low))
        ax_low = axes[idx]
        ax_low.plot(times_low, ranks_miner_low, "s--", label=f"miner:{miner_uid}", alpha=0.85)
        ax2 = None
        if len(real_end_low) == len(times_low) and len(real_end_low) > 0:
            ax2 = ax_low.twinx()
            ax2.plot(times_low, real_end_low, "-", color="gray", alpha=0.35, label="real_price")
            ax2.set_ylabel("Real price")
        ax_low.set_title(f"LOW (tl=86400) — {asset} on {date}")
        ax_low.set_xlabel("scored_time (UTC)")
        ax_low.set_ylabel("Rank")
        ax_low.grid(True, alpha=0.3)
        lines, labels = ax_low.get_legend_handles_labels()
        if ax2 is not None:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax_low.legend(lines + lines2, labels + labels2, loc="best")
        else:
            ax_low.legend(loc="best")

    fig.autofmt_xdate()
    plt.tight_layout()

    # Cố định trục Y: 0 ở dưới, rank lớn ở trên (ví dụ tối đa ~250)
    if max_rank_seen > 0:
        y_max = max(250, int(max_rank_seen * 1.05))
        for ax in axes:
            ax.set_ylim(0, y_max)

    safe_date = str(date).replace("-", "_")
    out_dir = os.path.join("result", "compare_from_db", str(asset), safe_date)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"fromdb_{asset}_miner{miner_uid}_rank_{safe_date}.png"
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=150)
    print(f"[from_db][plot] Saved plot to {out_path}")


def _augment_cache_with_real_prices(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cache JSON cũ có thể thiếu real_price_start/end. Hàm này sẽ bổ sung bằng cách
    fetch real prices cho từng slot trong per_slot (không chạy lại simulations).
    """
    per_slot: List[Dict[str, Any]] = result.get("per_slot", [])
    if not per_slot:
        return result

    # Nếu đã có đầy đủ thì thôi
    missing = [
        s for s in per_slot
        if s.get("real_price_start") is None or s.get("real_price_end") is None
    ]
    if not missing:
        return result

    asset = result.get("asset", "?")
    data_handler = DataHandler()

    # Reuse cùng cache file như compute_score.py để tránh fetch lại Pyth nhiều lần
    log_dir = "synth/miner/logs/real_prices"
    os.makedirs(log_dir, exist_ok=True)

    def _load_or_fetch_real_prices(vr: ValidatorRequest) -> List[float] | None:
        dict_vr = vr.__dict__.copy()
        dict_vr.pop("_sa_instance_state", None)
        hash_request = hashlib.sha256(str(dict_vr).encode()).hexdigest()
        cache_path = os.path.join(log_dir, f"{hash_request}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else None
            except Exception:
                pass
        try:
            t0 = time.time()
            data = data_handler.get_real_prices(validator_request=vr)
            # Best-effort write cache
            if isinstance(data, list):
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                except Exception:
                    pass
            dt = time.time() - t0
            if dt > 5:
                print(f"[from_db][cache] get_real_prices slow: {dt:.1f}s for {vr.asset} {vr.start_time} tl={vr.time_length} inc={vr.time_increment}")
            return data if isinstance(data, list) else None
        except Exception:
            return None

    filled = 0
    processed = 0
    start_t = time.time()
    for s in per_slot:
        if s.get("real_price_start") is not None and s.get("real_price_end") is not None:
            continue

        try:
            st = datetime.fromisoformat(str(s["scored_time"]).replace("Z", "+00:00"))
        except Exception:
            continue

        tl = int(s.get("time_length", 0))
        ti = int(s.get("time_increment", 0))
        if not tl or not ti:
            continue

        vr = ValidatorRequest(
            asset=str(asset),
            start_time=st,
            time_length=tl,
            time_increment=ti,
        )
        real_prices = _load_or_fetch_real_prices(vr)
        if isinstance(real_prices, list) and len(real_prices) > 0:
            try:
                s["real_price_start"] = float(real_prices[0])
                s["real_price_end"] = float(real_prices[-1])
                filled += 1
            except Exception:
                pass

        processed += 1
        if processed % 10 == 0:
            elapsed = time.time() - start_t
            print(
                f"[from_db][cache] augment progress: {processed}/{len(missing)} "
                f"filled={filled} elapsed={elapsed:.1f}s (asset={asset})"
            )

    print(
        f"[from_db][cache] Augmented real prices for {filled}/{len(missing)} missing slots "
        f"(asset={asset})"
    )
    return result


def run_backtest_from_db(
    date: str,
    asset: str,
    miner_uid: int = _COMPARE_MINER_UID,
) -> Dict:
    """
    Backtest 1 ngày cho 1 asset dựa trên các scored_time trong DB.
    Chỉ lấy CRPS của `miner_uid` trong DB và so rank với các miner khác.
    """
    # Đường dẫn cache: chia folder theo từng asset
    out_dir = os.path.join("result", "compare_from_db", asset)
    os.makedirs(out_dir, exist_ok=True)
    safe_date = date.replace("-", "_")
    filename = f"fromdb_{asset}_miner{int(miner_uid)}_rank_{safe_date}.json"
    path = os.path.join(out_dir, filename)

    # Nếu đã có cache thì dùng luôn
    if os.path.exists(path):
        print(f"[from_db] Dùng cache: {path}")
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)

        # Bổ sung real_price_start/end nếu cache cũ thiếu, rồi ghi đè lại cache.
        augmented = _augment_cache_with_real_prices(cached)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(augmented, f, indent=2)
        except Exception:
            pass
        return augmented

    day = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
    day_naive = day.astimezone(timezone.utc).replace(tzinfo=None)

    mysql = MySQLHandler()

    scored_slots = _get_scored_times_for_day(mysql, asset, day_naive)
    print(f"Found {len(scored_slots)} scored_time in DB for {asset} on {date}")

    per_slot: List[SlotRankResult] = []
    processed_slots = 0
    skipped_by_time_length: Dict[int, int] = {}
    def _init_stat() -> Dict[str, int]:
        # Bucket không chồng lấp:
        # - top_1_10:   1..10
        # - top_11_50:  11..50
        # - top_51_100: 51..100
        # - out_200:    >200
        return {
            "top_1_10": 0,
            "top_11_50": 0,
            "top_51_100": 0,
            "out_200": 0,
            "missing": 0,
            "no_db": 0,
        }

    stats = {
        "miner": _init_stat(),
        "high": {"miner": _init_stat()},
        "low": {"miner": _init_stat()},
    }

    for idx, (scored_time, tl) in enumerate(scored_slots, start=1):
        # Không tin cậy time_increment trong DB.
        # Map cố định: 3600 -> 60s (high), 86400 -> 300s (low).
        if tl == 3600:
            ti = 60
        elif tl == 86400:
            ti = 300
        else:
            skipped_by_time_length[tl] = skipped_by_time_length.get(tl, 0) + 1
            print(f"  Skipping: unsupported time_length {tl} (only 3600/86400 supported).")
            continue
        print(
            f"[{idx}/{len(scored_slots)}] {asset} @ {iso_format(scored_time)} "
            f"tl={tl} inc={ti}"
        )

        # Chọn prompt_config (high/low) nếu cần cho thống kê
        cfg = _get_prompt_config("high" if tl == 3600 else "low")
        if cfg.time_length != tl:
            print(f"  Skipping: time_length {tl} không khớp prompt_config.")
            continue

        rank_miner, total_m, crps_miner = _rank_miner_uid_against_db(
            mysql=mysql,
            asset=asset,
            scored_time=scored_time,
            time_length=tl,
            miner_uid=int(miner_uid),
        )
        total_miners = int(total_m)

        if total_miners == 0:
            stats["miner"]["no_db"] += 1
            if tl == 3600:
                stats["high"]["miner"]["no_db"] += 1
            else:
                stats["low"]["miner"]["no_db"] += 1
        else:
            def _bump(bucket: Dict[str, int], rank: int):
                if rank < 1:
                    return
                if 1 <= rank <= 10:
                    bucket["top_1_10"] += 1
                elif 11 <= rank <= 50:
                    bucket["top_11_50"] += 1
                elif 51 <= rank <= 100:
                    bucket["top_51_100"] += 1
                if rank > 200:
                    bucket["out_200"] += 1

            if rank_miner > 0:
                _bump(stats["miner"], rank_miner)
                if tl == 3600:
                    _bump(stats["high"]["miner"], rank_miner)
                else:
                    _bump(stats["low"]["miner"], rank_miner)
            else:
                stats["miner"]["missing"] += 1
                if tl == 3600:
                    stats["high"]["miner"]["missing"] += 1
                else:
                    stats["low"]["miner"]["missing"] += 1

        # In status của từng request sau khi xử lý xong
        print(
            f"  STATUS miner={int(miner_uid)} rank_miner={rank_miner} "
            f"total_miners={total_miners}"
        )

        processed_slots += 1
        per_slot.append(
            SlotRankResult(
                asset=asset,
                scored_time=iso_format(scored_time),
                time_length=tl,
                time_increment=ti,
                miner_uid=int(miner_uid),
                crps_miner=float(crps_miner) if crps_miner is not None else float("inf"),
                rank_miner=int(rank_miner),
                status_miner=("OK" if crps_miner is not None else "MISSING"),
                total_miners=int(total_miners),
                real_price_start=None,
                real_price_end=None,
            )
        )

    result = {
        "date": date,
        "asset": asset,
        "miner_uid": int(miner_uid),
        "total_slots": len(scored_slots),
        "stats": stats,
        "per_slot": [s.__dict__ for s in per_slot],
    }

    # Debug: thống kê số slot được xử lý và số slot bị skip theo time_length
    print(
        f"[from_db][DEBUG] total_slots={len(scored_slots)} "
        f"processed={processed_slots} "
        f"skipped_by_time_length={skipped_by_time_length}"
    )

    # Lưu JSON (cache)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[from_db] Đã lưu kết quả: {path}")

    return result


def main(
    date: str,
    assets: List[str],
    miner_uid: int = _COMPARE_MINER_UID,
) -> List[Dict]:
    """
    Main backtest function có thể gọi trực tiếp từ code.
    """
    results: List[Dict] = []
    for asset in assets:
        print("\n" + "=" * 70)
        print(f"[from_db] Asset={asset}  Date={date}")
        print("=" * 70)

        result = run_backtest_from_db(
            date=date,
            asset=asset,
            miner_uid=int(miner_uid),
        )
        results.append(result)

        stats = result.get("stats", {})
        total = result.get("total_slots", 0)
        print("\n=== SUMMARY vs DB (miner CRPS rank) ===")
        def _print_bucket(label: str, s: dict):
            print(
                f"  {label:<20} "
                f"top1-10={s.get('top_1_10',0):4d}  top11-50={s.get('top_11_50',0):4d}  "
                f"top51-100={s.get('top_51_100',0):4d}  out>200={s.get('out_200',0):4d}  "
                f"missing={s.get('missing',0):4d}  no_db={s.get('no_db',0):4d}"
            )

        _print_bucket(f"miner{int(miner_uid)} (all)", stats.get("miner", {}))
        _print_bucket(f"miner{int(miner_uid)} (high)", stats.get("high", {}).get("miner", {}))
        _print_bucket(f"miner{int(miner_uid)} (low)", stats.get("low", {}).get("miner", {}))
        print(f"  total_slots={total:4d}")

        # Thống kê thêm: rank trung bình của miner cho asset
        per_slot = result.get("per_slot", [])
        ranks_miner = [
            s["rank_miner"]
            for s in per_slot
            if s.get("status_miner") == "OK" and s.get("rank_miner", -1) > 0
        ]
        if ranks_miner:
            print("\n=== AVERAGE RANK (per asset) ===")
            avg_old = sum(ranks_miner) / len(ranks_miner)
            print(
                f"  {'miner'+str(int(miner_uid)):<20} avg_rank={avg_old:7.2f}  n={len(ranks_miner):4d}"
            )

        # Sau khi tính toán xong cho asset này, plot rank theo thời gian và lưu PNG
        try:
            _plot_ranks_for_day(result)
        except Exception as e:
            print(f"[from_db][WARN] Plot failed for {asset}: {e}")
    return results


if __name__ == "__main__":
    # CHỈNH THAM SỐ Ở ĐÂY RỒI CHẠY FILE
    _DATE = "2026-03-23"
    _ASSETS = [
        "BTC",
        "ETH",
        "XAU",
        "SOL",
        "SPYX",
        "NVDAX",
        "TSLAX",
        "AAPLX",
        "GOOGLX",
        ] # ví dụ: ["BTC", "ETH"]
    _MINER_UID = 95

    main(
        date=_DATE,
        assets=_ASSETS,
        miner_uid=_MINER_UID,
    )


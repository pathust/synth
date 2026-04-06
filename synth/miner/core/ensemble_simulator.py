"""
ensemble_simulator.py

Ensemble simulator cho crypto assets (BTC, ETH, SOL, XAU).

Thay vì dùng 1 model cố định để sinh 1000 paths, simulator này:
  1. Sinh số p ~ Normal(mu_paths, sigma_paths) — truncated về [1, n_sims]
  2. Chọn ngẫu nhiên 1 GARCH variant để sinh p paths
  3. Lặp lại cho đến khi đủ n_sims paths

Kết quả: 1000 paths từ nhiều GARCH variant khác nhau,
phân phối bao quát được nhiều regime hơn bất kỳ model đơn lẻ nào.

Mỗi lần gọi với seed khác nhau → phân phối p và thứ tự model khác nhau,
nhưng vẫn reproducible khi cùng seed.

Usage:
    from synth.miner.core.ensemble_simulator import simulate_ensemble_crypto

    paths = simulate_ensemble_crypto(
        prices_dict=prices_dict,
        asset="BTC",
        time_increment=300,
        time_length=86400,
        n_sims=1000,
        seed=42,
    )
    # paths.shape == (1000, 289)
"""

import numpy as np
from typing import Optional

from synth.miner.core.garch_simulator import (
    simulate_single_price_path_with_garch as garch_v1,
)
from synth.miner.core.grach_simulator_v2 import (
    simulate_single_price_path_with_garch as garch_v2,
)

from synth.miner.core.grach_simulator_v4 import (
    simulate_single_price_path_with_garch as garch_v4,
)
from synth.miner.core.grach_simulator_v4_1 import (
    simulate_single_price_path_with_garch as garch_v4_1,
)
from synth.miner.core.grach_simulator_v4_2 import (
    simulate_single_price_path_with_garch as garch_v4_2,
)
from synth.miner.core.HAR_RV_simulatior import (
    simulate_single_price_path_with_har_garch as har_rv,
)

# ── Pool of models theo asset ─────────────────────────────────────────

_CRYPTO_MODELS = [garch_v1, garch_v2, garch_v4, garch_v4_1, garch_v4_2, har_rv]

_ASSET_MODELS = {
    "BTC": _CRYPTO_MODELS,
    "ETH": _CRYPTO_MODELS,
    "SOL": _CRYPTO_MODELS,
    "XAU": [garch_v1, garch_v2, garch_v4, garch_v4_1, garch_v4_2, har_rv],
}

# ── Phân phối số paths mỗi lần draw ──────────────────────────────────
# p ~ Normal(mu, sigma), truncated về [p_min, p_max]
# Mỗi lần draw chọn ~1 model và sinh ~p paths từ model đó
P_MU    = 150   # trung bình ~150 paths/lần → cần ~7 lần cho 1000
P_SIGMA = 60    # độ lệch chuẩn → có lần 50 paths, có lần 250 paths
P_MIN   = 10    # tối thiểu 10 paths/lần
P_MAX   = 400   # tối đa 400 paths/lần

def _sample_n_paths(rng: np.random.Generator) -> int:
    """Sinh số paths cho 1 lần draw từ phân phối chuẩn truncated."""
    while True:
        p = int(round(rng.normal(P_MU, P_SIGMA)))
        if P_MIN <= p <= P_MAX:
            return p

def simulate_ensemble_crypto(
    prices_dict: dict,
    asset: str,
    time_increment: int,
    time_length: int,
    n_sims: int = 1000,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Ensemble simulator: sinh n_sims paths từ nhiều GARCH variant.

    Thuật toán:
        rng = np.random.default_rng(seed)
        collected = []
        while len(collected) < n_sims:
            p = sample từ Normal(P_MU, P_SIGMA) truncated [P_MIN, P_MAX]
            model = chọn ngẫu nhiên từ pool models cho asset
            paths = model(prices_dict, ..., n_sims=p, seed=rng_seed)
            collected += paths[:min(p, n_sims - len(collected))]

    Args:
        prices_dict: {timestamp_str: price} — dữ liệu lịch sử
        asset: "BTC", "ETH", "SOL", "XAU"
        time_increment: bước thời gian (giây), VD: 300
        time_length: tổng độ dài dự báo (giây), VD: 86400
        n_sims: tổng số paths muốn sinh (mặc định 1000)
        seed: random seed để reproducible

    Returns:
        np.ndarray shape (n_sims, steps+1)

    Raises:
        ValueError: nếu asset không được hỗ trợ
        RuntimeError: nếu không sinh được đủ paths sau max_attempts
    """
    if asset not in _ASSET_MODELS:
        raise ValueError(
            f"Asset '{asset}' không được hỗ trợ. "
            f"Các asset hợp lệ: {list(_ASSET_MODELS.keys())}"
        )

    model_pool = _ASSET_MODELS[asset]
    rng = np.random.default_rng(seed)

    collected: list[np.ndarray] = []
    collected_count = 0
    draw_log: list[dict] = []  # log để debug
    max_attempts = n_sims * 5  # safety: tối đa 5× n_sims lần thử
    attempt = 0

    print(
        f"[Ensemble] {asset}: target={n_sims} paths, "
        f"pool={len(model_pool)} models, "
        f"p~N({P_MU},{P_SIGMA}) truncated [{P_MIN},{P_MAX}]"
    )

    while collected_count < n_sims and attempt < max_attempts:
        attempt += 1

        # 1. Chọn model ngẫu nhiên từ pool
        model_idx = int(rng.integers(0, len(model_pool)))
        model_fn = model_pool[model_idx]

        # 2. Sinh số paths p ~ Normal truncated
        p = _sample_n_paths(rng)
        # Không sinh quá số còn thiếu
        p = min(p, n_sims - collected_count)

        # 3. Seed riêng cho lần này (deterministic từ rng chính)
        sub_seed = int(rng.integers(0, 2**31))

        try:
            paths = model_fn(
                prices_dict,
                asset=asset,
                time_increment=time_increment,
                time_length=time_length,
                n_sims=p,
                seed=sub_seed,
            )

            if paths is None or not isinstance(paths, np.ndarray) or paths.ndim != 2:
                continue
            if paths.shape[0] == 0:
                continue

            # Lấy đúng số paths cần (model có thể trả về nhiều hơn p)
            take = min(paths.shape[0], n_sims - collected_count)
            collected.append(paths[:take])
            collected_count += take

            draw_log.append({
                "attempt": attempt,
                "model": model_fn.__name__,
                "requested": p,
                "got": take,
                "total_so_far": collected_count,
            })

        except Exception as e:
            # Model này fail → thử model khác ở vòng tiếp theo
            draw_log.append({
                "attempt": attempt,
                "model": model_fn.__name__,
                "requested": p,
                "got": 0,
                "error": str(e)[:80],
                "total_so_far": collected_count,
            })
            continue

    if collected_count < n_sims:
        raise RuntimeError(
            f"[Ensemble] Không sinh đủ paths: có {collected_count}/{n_sims} "
            f"sau {attempt} lần thử. Kiểm tra lại dữ liệu lịch sử."
        )

    result = np.vstack(collected)  # shape (n_sims, steps+1)

    # Summary log
    model_counts: dict[str, int] = {}
    for log in draw_log:
        if "error" not in log:
            name = log["model"]
            model_counts[name] = model_counts.get(name, 0) + log["got"]

    total_draws = sum(1 for l in draw_log if "error" not in l)
    total_errors = sum(1 for l in draw_log if "error" in l)
    print(
        f"[Ensemble] Done: {n_sims} paths from {total_draws} draws "
        f"({total_errors} errors), distribution:"
    )
    for name, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        pct = count / n_sims * 100
        print(f"  {name:<45}: {count:>4} paths ({pct:.1f}%)")

    return result

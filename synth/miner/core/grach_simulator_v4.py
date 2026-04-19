import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import SkewStudent
from scipy.stats import t as student_t
from typing import Optional, Dict

# ==========================================
# ⚙️ 1. CẤU HÌNH TỐI ƯU THEO TÀI SẢN
# ==========================================
def get_optimal_config(asset: str, time_increment: int) -> dict:
    """
    Cấu hình được tinh chỉnh để giảm CRPS dựa trên đặc tính từng loại tài sản.
    """
    asset_upper = asset.upper()
    is_xau = asset_upper in ["XAU", "GOLD"]
    is_wtioil = asset_upper == "WTIOIL"
    is_high_freq = time_increment <= 60  # Khung 1h (1m steps)

    config = {
        "scale": 10000.0,      # Chuyển về Basis points để model ổn định
        "dist": "skewt",       # Skew Student-t để bắt đuôi lệch
        "vol_model": "GARCH",
        "p": 1, "q": 1,
        "o": 1 if not (is_xau or is_wtioil) else 0,  # GJR-GARCH (o=1) cho Crypto, GARCH thường cho Vàng
        "simulation_method": "FHS"   # Luôn ưu tiên FHS để tối ưu CRPS
    }

    if is_xau:
        config["mean_model"] = "Constant"  # Vàng có trend rõ hơn nên dùng Constant
        config["lookback_days"] = 15 if is_high_freq else 30
        config["momentum_weight"] = 0.8   # Vàng bám trend rất mạnh
        config["drift_decay"] = 0.995     # Lực trend bền (giảm chậm)
    elif is_wtioil:
        config["mean_model"] = "Constant"
        config["lookback_days"] = 15 if is_high_freq else 30
        config["momentum_weight"] = 0.8   
        config["drift_decay"] = 0.995     
    elif asset_upper == "BTC":
        # BTC (low): phản ứng nhanh khi đổi pha pump/dump 4-5 ngày.
        # Thu hẹp lookback để bám regime mới và tăng nhẹ trọng số momentum.
        config["mean_model"] = "Zero"
        config["lookback_days"] = 7 if is_high_freq else 25
        config["momentum_weight"] = 0.5
        config["drift_decay"] = 0.92
    elif asset_upper == "ETH":
        config["mean_model"] = "Zero"
        config["lookback_days"] = 7 if is_high_freq else 20
        config["momentum_weight"] = 0.6
        config["drift_decay"] = 0.92
    elif asset_upper == "HYPE":
        config["mean_model"] = "Zero"
        config["lookback_days"] = 7 if is_high_freq else 20
        config["momentum_weight"] = 0.6
        config["drift_decay"] = 0.92
    elif asset_upper == "XRP":
        config["mean_model"] = "Zero"
        config["lookback_days"] = 7 if is_high_freq else 20
        config["momentum_weight"] = 0.6
        config["drift_decay"] = 0.92
    elif asset_upper == "SOL":
        # SOL (low): đổi pha rất nhanh, cần nhìn lịch sử ngắn và bám trend mạnh.
        config["mean_model"] = "Zero"
        config["lookback_days"] = 7 if is_high_freq else 15
        config["momentum_weight"] = 0.6
        config["drift_decay"] = 0.92
    elif asset_upper == "GOOGLX":
        # GOOGLX: uptrend mạnh trong ngày → cần drift mạnh + focus lịch sử gần
        config["mean_model"] = "Constant"  # Có drift rõ ràng, không dùng Zero
        config["lookback_days"] = 7 if is_high_freq else 15  # Hạ xuống 15d (low) để bắt sóng gần
        config["momentum_weight"] = 0.8    # Tăng từ 0.4 → 0.8, bám trend dốc đứng
        config["drift_decay"] = 0.97       # Lực trend giảm chậm hơn crypto thông thường
    elif asset_upper == "NVDAX":
        # NVDAX: cổ phiếu tăng trưởng, spike mạnh, cần drift dương cao và lookback ngắn
        config["mean_model"] = "Constant"  # Tăng trưởng cốt lõi → kỳ vọng lợi suất dương
        config["lookback_days"] = 7 if is_high_freq else 25  # 25d (low): bỏ qua lịch sử quá cũ
        config["momentum_weight"] = 0.8    # Giống XAU: bám trend rất mạnh khi đang pump
        config["drift_decay"] = 0.97       # Giữ lực đẩy lâu, không tan nhanh như crypto
    elif asset_upper == "TSLAX":
        # TSLAX: biên độ giật mạnh, cần drift cực mạnh và lookback rất ngắn
        config["mean_model"] = "Constant"  # Cổ phiếu tăng trưởng → lợi suất kỳ vọng dương
        config["lookback_days"] = 7 if is_high_freq else 15  # 15d (low): chỉ nhớ hành vi gần nhất
        config["momentum_weight"] = 0.85   # Mạnh hơn XAU chút (0.85 vs 0.8), đánh đúng đạn
        config["drift_decay"] = 0.97       # Giữ lực đẩy khá lâu
    elif asset_upper == "AAPLX":
        # AAPLX: tăng trưởng từ tốn, liền mạch → drift vừa phải, lookback dài
        config["mean_model"] = "Constant"  # Cổ phiếu có kỳ vọng dương, không dùng Zero
        config["lookback_days"] = 7 if is_high_freq else 45  # 45d (low): giữ lịch sử dài cho variance ổn
        config["momentum_weight"] = 0.5    # Vừa phải: không tăng như TSLAX (0.85), đủ bám trend nhẹ
        config["drift_decay"] = 0.95       # Giảm chậm vừa phải, không nhanh như crypto
    elif asset_upper == "SPYX":
        # SPYX đi chậm hơn crypto, cần hãm drift momentum để tránh bay quá đà.
        config["mean_model"] = "Constant"
        config["lookback_days"] = 7 if is_high_freq else 45
        config["momentum_weight"] = 0.2
        config["drift_decay"] = 0.95
    else:
        config["mean_model"] = "Zero"     # Crypto nhiễu mạnh nên dùng Zero mean
        config["lookback_days"] = 7 if is_high_freq else 45
        config["momentum_weight"] = 0.4   # Crypto dễ đảo chiều, dùng ít momentum thôi
        config["drift_decay"] = 0.92      # Lực trend yếu nhanh (giảm nhanh)

    return config

# ==========================================
# ⚙️ 1.5. PARAM GRID CHO TUNER TỪNG TÀI SẢN
# ==========================================
def get_optimal_param_grid(asset: str, time_increment: int) -> dict:
    """
    Tạo không gian tìm kiếm (param_grid) xoay quanh các giá trị optimal
    của từng tài sản.
    """
    asset_upper = asset.upper()
    is_xau = asset_upper in ["XAU", "GOLD"]
    is_wtioil = asset_upper == "WTIOIL"
    is_high_freq = time_increment <= 60  # Khung 1h (1m steps)

    grid = {
        "p": [1],
        "q": [1],
    }

    if is_xau:
        grid["lookback_days"] = [10, 15] if is_high_freq else [20, 30]
        grid["momentum_weight"] = [0.6, 0.8]
        grid["drift_decay"] = [0.98, 0.995]
    elif is_wtioil:
        grid["lookback_days"] = [10, 15] if is_high_freq else [20, 30]
        grid["momentum_weight"] = [0.6, 0.8]
        grid["drift_decay"] = [0.98, 0.995]
    elif asset_upper == "BTC":
        grid["lookback_days"] = [5, 7] if is_high_freq else [15, 25]
        grid["momentum_weight"] = [0.3, 0.5]
        grid["drift_decay"] = [0.90, 0.92]
    elif asset_upper == "ETH":
        grid["lookback_days"] = [5, 7] if is_high_freq else [15, 20]
        grid["momentum_weight"] = [0.4, 0.6]
        grid["drift_decay"] = [0.90, 0.92]
    elif asset_upper == "HYPE":
        grid["lookback_days"] = [5, 7] if is_high_freq else [15, 20]
        grid["momentum_weight"] = [0.4, 0.6]
        grid["drift_decay"] = [0.90, 0.92]
    elif asset_upper == "XRP":
        grid["lookback_days"] = [5, 7] if is_high_freq else [15, 20]
        grid["momentum_weight"] = [0.4, 0.6]
        grid["drift_decay"] = [0.90, 0.92]
    elif asset_upper == "SOL":
        grid["lookback_days"] = [5, 7] if is_high_freq else [10, 15]
        grid["momentum_weight"] = [0.4, 0.6]
        grid["drift_decay"] = [0.90, 0.92]
    elif asset_upper == "GOOGLX":
        grid["lookback_days"] = [5, 10] if is_high_freq else [10, 15]
        grid["momentum_weight"] = [0.6, 0.8]
        grid["drift_decay"] = [0.95, 0.97]
    elif asset_upper == "NVDAX":
        grid["lookback_days"] = [5, 10] if is_high_freq else [15, 25]
        grid["momentum_weight"] = [0.6, 0.8]
        grid["drift_decay"] = [0.95, 0.97]
    elif asset_upper == "TSLAX":
        grid["lookback_days"] = [5, 10] if is_high_freq else [10, 15]
        grid["momentum_weight"] = [0.7, 0.85]
        grid["drift_decay"] = [0.95, 0.97]
    elif asset_upper == "AAPLX":
        grid["lookback_days"] = [5, 10] if is_high_freq else [30, 45]
        grid["momentum_weight"] = [0.3, 0.5]
        grid["drift_decay"] = [0.95, 0.97]
    elif asset_upper == "SPYX":
        grid["lookback_days"] = [5, 10] if is_high_freq else [30, 45]
        grid["momentum_weight"] = [0.1, 0.2]
        grid["drift_decay"] = [0.95, 0.97]
    else:
        grid["lookback_days"] = [5, 10] if is_high_freq else [30, 45]
        grid["momentum_weight"] = [0.2, 0.4]
        grid["drift_decay"] = [0.90, 0.92]

    return grid

# ==========================================
# 🛠️ 2. BỘ PHÁT HIỆN TRẠNG THÁI THỊ TRƯỜNG
# ==========================================
def detect_market_regime(returns_bps: pd.Series, window: int = 20):
    """
    Sử dụng Z-Score để phát hiện Breakout.
    Nếu Z > 1.5: Thị trường đang bùng nổ (Trending).
    """
    if len(returns_bps) < window:
        return 'sideways', 0.0

    recent_mome = returns_bps.tail(window).mean()
    hist_std = returns_bps.tail(window * 5).std()

    # Tính Z-Score cho momentum
    z_score = recent_mome / (hist_std / np.sqrt(window)) if hist_std > 0 else 0

    if abs(z_score) > 1.5:
        return 'trending', float(recent_mome)
    return 'sideways', 0.0

# ==========================================
# 📈 3. LÕI MÔ PHỎNG NÂNG CAO
# ==========================================
def simulate_paths_advanced(fitted_res, S0, steps, n_sims, config, drift_bps=0.0, seed=42):
    if seed is not None:
        np.random.seed(seed)

    params = fitted_res.params
    scale = config["scale"]

    # Trích xuất tham số model
    mu_model = float(params.get("mu", 0.0))
    omega = float(params.get("omega", 0.01))
    alpha = float(params.get("alpha[1]", 0.05))
    beta = float(params.get("beta[1]", 0.90))
    gamma = float(params.get("gamma[1]", 0.0))  # Tham số bất đối xứng (GJR)
    nu = float(params.get("nu", 10.0))
    lambda_ = float(params.get("lambda", 0.0))

    # Khởi tạo trạng thái tại T=0
    last_vol = float(fitted_res.conditional_volatility.iloc[-1])
    last_shock = float(fitted_res.resid.iloc[-1])

    sigma_prev = np.full(n_sims, last_vol)
    eps_prev = np.full(n_sims, last_shock)

    # Sinh số ngẫu nhiên (Innovations)
    if config["simulation_method"] == "FHS":
        # Filtered Historical Simulation: Bốc thăm phần dư chuẩn hóa thực tế
        std_resids = fitted_res.std_resid.dropna().values
        z = np.random.choice(std_resids, size=(steps, n_sims), replace=True)
    else:
        # Parametric: Dùng phân phối Skew-T (Nếu FHS không khả dụng)
        dist_gen = SkewStudent()
        z = dist_gen.simulate([nu, lambda_], steps * n_sims).reshape((steps, n_sims))

    # Vòng lặp mô phỏng
    safe_drift = np.clip(drift_bps, -5.0, 5.0)
    returns_bps = np.zeros((steps, n_sims))
    decay = config.get("drift_decay", 0.95)

    for t in range(steps):
        indicator = (eps_prev < 0).astype(float)
        sigma2 = omega + alpha * (eps_prev ** 2) + gamma * (eps_prev ** 2) * indicator + beta * (sigma_prev ** 2)
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))

        current_step_drift = safe_drift * (decay ** t)

        # 2. KIỂM SOÁT SHOCK: Giới hạn cú sốc ngẫu nhiên
        raw_shocks = sigma_t * z[t, :]
        safe_shocks = np.clip(raw_shocks, -100.0, 100.0)  # Giới hạn 1% mỗi nến

        returns_bps[t, :] = mu_model + current_step_drift + safe_shocks

        sigma_prev = sigma_t
        eps_prev = safe_shocks

    # 3. CHUYỂN ĐỔI VỀ GIÁ VÀ KIỂM TRA CUỐI CÙNG
    log_ret = returns_bps / scale

    # Giới hạn log_ret tổng không quá +/- 15% trong 1 ngày (tránh giá 10 triệu)
    cum_log_ret = np.cumsum(log_ret, axis=0)
    cum_log_ret = np.clip(cum_log_ret, -0.15, 0.15)

    prices = np.zeros((n_sims, steps + 1))
    prices[:, 0] = S0
    prices[:, 1:] = S0 * np.exp(cum_log_ret).T

    return prices

# ==========================================
# 🚀 4. HÀM ĐIỀU KHIỂN CHÍNH (CONTROLLER)
# ==========================================
def simulate_single_price_path_with_garch(
    prices_dict: Dict,
    asset: str,
    time_increment: int,
    time_length: int,
    n_sims: int,
    seed: Optional[int] = 42,
    **kwargs,
):
    # 1. Tiền xử lý dữ liệu
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()

    config = get_optimal_config(asset, time_increment)
    if kwargs:
        config.update(kwargs)

    # Cắt dữ liệu theo lookback window
    points_per_day = 86400 // time_increment
    needed_points = int(config["lookback_days"] * points_per_day)
    if len(full_prices) > needed_points:
        hist_prices = full_prices.tail(needed_points)
        print(f"[INFO] {asset}: Cắt dữ liệu xuống {config['lookback_days']} ngày gần nhất ({len(hist_prices)} nến).")
    else:
        hist_prices = full_prices
        print(f"[WARN] {asset}: Dữ liệu lịch sử ({len(full_prices)} nến) ngắn hơn mức tối ưu ({needed_points}).")

    # 2. Tính Returns (Scale lên Basis Points)
    returns = np.log(hist_prices).diff().dropna() * config["scale"]

    # 3. Phát hiện Regime & Tính Drift
    regime, momentum_bps = detect_market_regime(returns)
    drift_to_use = 0.0
    if regime == 'trending':
        drift_to_use = momentum_bps * config["momentum_weight"]

    # 4. Fit Model GARCH
    print(f"[PROCESS] Fitting {asset} | Regime: {regime} | Drift: {drift_to_use:.4f}")
    model = arch_model(
        returns,
        mean=config["mean_model"],
        vol=config["vol_model"],
        p=config["p"], o=config["o"], q=config["q"],
        dist=config["dist"],
    )
    try:
        res = model.fit(disp="off", show_warning=False, options={"maxiter": 500})

        # KIỂM TRA HỘI TỤ: Nếu tham số ra kết quả dị thường (ví dụ mu quá lớn)
        if abs(res.params.get("mu", 0)) > 50:  # Nếu mu > 50 bps (~0.5% mỗi nến)
            raise ValueError("Inconsistent parameters detected")

    except Exception:
        # FALLBACK: Nếu lỗi, dùng mô hình đơn giản hơn hoặc trả về tham số mặc định
        print("[WARN] Using fallback parameters to prevent price explosion.")
        res = model.fit(disp="off", rescale=True)

    # 5. Chạy mô phỏng
    steps = time_length // time_increment
    S0 = float(hist_prices.iloc[-1])

    paths = simulate_paths_advanced(
        fitted_res=res,
        S0=S0,
        steps=steps,
        n_sims=n_sims,
        config=config,
        drift_bps=drift_to_use,
        seed=seed,
    )

    print(f"[SUCCESS] {asset} simulation complete.")
    return paths

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import SkewStudent # Đảm bảo bạn dùng bản arch mới nhất
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
    is_high_freq = time_increment <= 60 # Khung 1h (1m steps)

    config = {
        "scale": 10000.0,      # Chuyển về Basis points để model ổn định
        "dist": "skewt",       # Skew Student-t để bắt đuôi lệch
        "vol_model": "GARCH", 
        "p": 1, "q": 1, 
        "o": 1 if not is_xau else 0, # GJR-GARCH (o=1) cho Crypto, GARCH thường cho Vàng
        "simulation_method": "FHS"   # Luôn ưu tiên FHS để tối ưu CRPS
    }

    if is_xau:
        config["mean_model"] = "Constant" # Vàng có trend rõ hơn nên dùng Constant
        config["lookback_days"] = 15 if is_high_freq else 30
        config["momentum_weight"] = 0.8   # Vàng bám trend rất mạnh
        config["drift_decay"] = 0.995     # Lực trend bền (giảm chậm)
    else:
        config["mean_model"] = "Zero"     # Crypto nhiễu mạnh nên dùng Zero mean
        config["lookback_days"] = 7 if is_high_freq else 45
        config["momentum_weight"] = 0.4   # Crypto dễ đảo chiều, dùng ít momentum thôi
        config["drift_decay"] = 0.92      # Lực trend yếu nhanh (giảm nhanh)

    return config

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
        return 'trending', recent_mome
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
    mu_model = params.get("mu", 0.0) 
    omega    = params.get("omega", 0.01)
    alpha    = params.get("alpha[1]", 0.05)
    beta     = params.get("beta[1]", 0.90)
    gamma    = params.get("gamma[1]", 0.0) # Tham số bất đối xứng (GJR)
    nu       = params.get("nu", 10.0)
    lambda_  = params.get("lambda", 0.0)

    # Khởi tạo trạng thái tại T=0
    last_vol = fitted_res.conditional_volatility.iloc[-1]
    last_shock = fitted_res.resid.iloc[-1]

    sigma_prev = np.full(n_sims, last_vol)
    eps_prev   = np.full(n_sims, last_shock)

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
        sigma2 = omega + alpha*(eps_prev**2) + gamma*(eps_prev**2)*indicator + beta*(sigma_prev**2)
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))

        current_step_drift = safe_drift * (decay ** t)
        
        # 2. KIỂM SOÁT SHOCK: Giới hạn cú sốc ngẫu nhiên
        raw_shocks = sigma_t * z[t, :]
        safe_shocks = np.clip(raw_shocks, -100.0, 100.0) # Giới hạn 1% mỗi nến
        
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
    
    # for t in range(steps):
    #     # Logic GJR-GARCH: Giá giảm (shock < 0) làm tăng biến động mạnh hơn giá tăng
    #     indicator = (eps_prev < 0).astype(float)
    #     sigma2 = omega + alpha*(eps_prev**2) + gamma*(eps_prev**2)*indicator + beta*(sigma_prev**2)
    #     sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))

    #     # Áp dụng Damped Drift (Quán tính giảm dần)
    #     current_step_drift = kd * (decay ** t)
        
    #     eps_t = sigma_t * z[t, :]
    #     returns_bps[t, :] = mu_model + current_step_drift + eps_t

    #     # Cập nhật cho bước tiếp theo
    #     sigma_prev = sigma_t
    #     eps_prev = eps_t

    # # Chuyển đổi về giá: P_t = P_0 * exp(sum(r))
    # log_ret = returns_bps / scale
    # prices = np.zeros((n_sims, steps + 1))
    # prices[:, 0] = S0
    # prices[:, 1:] = S0 * np.exp(np.cumsum(log_ret, axis=0)).T
    # return prices

# ==========================================
# 🚀 4. HÀM ĐIỀU KHIỂN CHÍNH (CONTROLLER)
# ==========================================
def simulate_single_price_path_with_garch(prices_dict: Dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: int = 42):
    # 1. Tiền xử lý dữ liệu
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    
    config = get_optimal_config(asset, time_increment)
    
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
    model = arch_model(returns, 
                       mean=config["mean_model"], 
                       vol=config["vol_model"], 
                       p=config["p"], o=config["o"], q=config["q"], 
                       dist=config["dist"])
    try:
        # Sử dụng phương pháp 'cobyla' hoặc 'slsqp' và thêm biến rescale=True
        res = model.fit(disp="off", show_warning=False, options={"maxiter": 500})
        
        # KIỂM TRA HỘI TỤ: Nếu tham số ra kết quả dị thường (ví dụ mu quá lớn)
        if abs(res.params.get("mu", 0)) > 50: # Nếu mu > 50 bps (~0.5% mỗi nến)
             raise ValueError("Inconsistent parameters detected")
             
    except:
        # FALLBACK: Nếu lỗi, dùng mô hình đơn giản hơn hoặc trả về tham số mặc định
        print("[WARN] Using fallback parameters to prevent price explosion.")
        res = model.fit(disp="off", rescale=True)

    # try:
    #     res = model.fit(disp="off", options={"maxiter": 400})
    # except:
    #     res = model.fit(disp="off", rescale=True)

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
        seed=seed
    )
    
    print(f"[SUCCESS] {asset} simulation complete.")
    return paths

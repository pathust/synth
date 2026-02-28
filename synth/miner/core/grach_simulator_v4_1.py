import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import SkewStudent
from typing import Optional, Dict

# ==========================================
# ⚙️ 1. CẤU HÌNH TỐI ƯU (TỐI ƯU CRPS)
# ==========================================
def get_optimal_config(asset: str, time_increment: int) -> dict:
    asset_upper = asset.upper()
    is_xau = asset_upper in ["XAU", "GOLD"]
    is_high_freq = time_increment <= 60 # Khung 1h

    config = {
        "scale": 10000.0,      # Chuyển về Basis points
        "dist": "skewt",       
        "vol_model": "GARCH", 
        "p": 1, "q": 1, 
        "o": 1 if not is_xau else 0, # GJR-GARCH cho Crypto
        "simulation_method": "FHS"   
    }

    if is_xau:
        config["mean_model"] = "Constant"
        config["lookback_days"] = 10 if is_high_freq else 25 # Thu ngắn để nhạy trend
        config["momentum_weight"] = 0.5   # Giảm xuống 0.5 để an toàn (tránh nổ giá)
        config["drift_decay"] = 0.99      
    else:
        config["mean_model"] = "Zero"
        config["lookback_days"] = 7 if is_high_freq else 40
        config["momentum_weight"] = 0.2   
        config["drift_decay"] = 0.90      

    return config

# ==========================================
# 🛠️ 2. PHÁT HIỆN TREND & DRIFT
# ==========================================
def detect_market_regime(returns_bps: pd.Series, window: int = 15):
    """
    Tính toán xem giá đang có xu hướng mạnh hay không.
    Trả về: loại regime và giá trị drift trung bình mỗi bước (bps).
    """
    if len(returns_bps) < window:
        return 'sideways', 0.0

    recent_mome = returns_bps.tail(window).mean()
    hist_std = returns_bps.std()
    
    # Z-Score của momentum ngắn hạn so với độ lệch chuẩn tổng thể
    z_score = recent_mome / (hist_std / np.sqrt(window)) if hist_std > 0 else 0
    
    # Ngưỡng 1.5 là điểm bắt đầu kích hoạt bám trend
    if abs(z_score) > 1.5:
        return 'trending', recent_mome
    return 'sideways', 0.0

# ==========================================
# 📈 3. MÔ PHỎNG NÂNG CAO (CÓ CHỐT CHẶN)
# ==========================================
def simulate_paths_advanced(fitted_res, S0, steps, n_sims, config, drift_bps=0.0, seed=42):
    if seed is not None:
        np.random.seed(seed)

    params = fitted_res.params
    scale = config["scale"]
    
    # Trích xuất tham số - Kiểm soát giá trị Mu để tránh nổ giá
    mu_model = np.clip(params.get("mu", 0.0), -2.0, 2.0) 
    omega    = params.get("omega", 0.01)
    alpha    = params.get("alpha[1]", 0.05)
    beta     = params.get("beta[1]", 0.90)
    gamma    = params.get("gamma[1]", 0.0) 
    
    # Khởi tạo T=0
    last_vol = fitted_res.conditional_volatility.iloc[-1]
    last_shock = fitted_res.resid.iloc[-1]

    sigma_prev = np.full(n_sims, last_vol)
    eps_prev   = np.full(n_sims, last_shock)

    # Shocks từ phần dư thực tế (FHS) giúp bắt đúng độ dày đuôi (CRPS tốt nhất)
    std_resids = fitted_res.std_resid.dropna().values
    if len(std_resids) < 10: # Phòng hờ dữ liệu lỗi
        z = np.random.standard_t(df=8, size=(steps, n_sims))
    else:
        z = np.random.choice(std_resids, size=(steps, n_sims), replace=True)

    returns_bps_matrix = np.zeros((steps, n_sims))
    decay = config.get("drift_decay", 0.95)
    
    # Khóa an toàn cho drift: Không quá 3 bps mỗi bước (0.03%)
    safe_drift = np.clip(drift_bps, -3.0, 3.0)

    for t in range(steps):
        # GJR-GARCH Equation
        indicator = (eps_prev < 0).astype(float)
        sigma2 = omega + alpha*(eps_prev**2) + gamma*(eps_prev**2)*indicator + beta*(sigma_prev**2)
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))

        current_step_drift = safe_drift * (decay ** t)
        
        # Shock ngẫu nhiên
        eps_t = sigma_t * z[t, :]
        
        # TỔNG LỢI NHUẬN BPS: Mu + Drift + Volatility Shock
        step_return = mu_model + current_step_drift + eps_t
        
        # GIỚI HẠN: Một cây nến 1m/5m không được biến động quá 2% (200 bps)
        returns_bps_matrix[t, :] = np.clip(step_return, -200.0, 200.0)

        # Update
        sigma_prev = sigma_t
        eps_prev = returns_bps_matrix[t, :]

    # Chuyển về giá và chặn biến động tổng cả ngày (Max 15% move)
    log_ret = returns_bps_matrix / scale
    cum_log_ret = np.cumsum(log_ret, axis=0)
    
    # Giới hạn tổng biến động để tránh lỗi Format giá (ví dụ giá vàng không thể lên 10 triệu)
    cum_log_ret_clipped = np.clip(cum_log_ret, -0.15, 0.15) 

    prices = np.zeros((n_sims, steps + 1))
    prices[:, 0] = S0
    prices[:, 1:] = S0 * np.exp(cum_log_ret_clipped).T
    
    return prices

# ==========================================
# 🚀 4. HÀM ĐIỀU KHIỂN CHÍNH
# ==========================================
def simulate_single_price_path_with_garch(prices_dict: Dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: int = 42):
    # 1. Load Data
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    S0 = float(full_prices.iloc[-1])
    
    config = get_optimal_config(asset, time_increment)
    
    # 2. Pre-processing
    points_per_day = 86400 // time_increment
    needed_points = int(config["lookback_days"] * points_per_day)
    hist_prices = full_prices.tail(needed_points)
    
    # Log-returns nhân với scale (bps)
    returns_bps = np.log(hist_prices).diff().dropna() * config["scale"]
    
    # 3. Regime Detection
    regime, momentum_bps = detect_market_regime(returns_bps)
    drift_to_use = momentum_bps * config["momentum_weight"] if regime == 'trending' else 0.0
    
    # 4. Fit Model với xử lý lỗi hội tụ
    print(f"[PROCESS] Asset: {asset} | Regime: {regime} | Drift: {drift_to_use:.4f} bps")
    
    model = arch_model(returns_bps, 
                       mean=config["mean_model"], 
                       vol=config["vol_model"], 
                       p=config["p"], o=config["o"], q=config["q"], 
                       dist=config["dist"])
    
    try:
        # Thử fit với maxiter cao
        res = model.fit(disp="off", show_warning=False, options={"maxiter": 600})
        
        # Kiểm tra nếu tham số bị vọt lên mức vô lý (dấu hiệu fit hỏng)
        if abs(res.params.get("mu", 0)) > 10 or res.params.get("beta[1]", 0) > 0.999:
            raise ValueError("Unstable parameters")
            
    except Exception as e:
        print(f"[WARN] Convergence failed or unstable for {asset}. Using safety fallback.")
        # Fallback: dùng model đơn giản (Constant Mean + GARCH) với rescaling
        res = model.fit(disp="off", show_warning=False, rescale=True)

    # 5. Simulation
    steps = time_length // time_increment
    paths = simulate_paths_advanced(
        fitted_res=res,
        S0=S0,
        steps=steps,
        n_sims=n_sims,
        config=config,
        drift_bps=drift_to_use,
        seed=seed
    )
    
    return paths
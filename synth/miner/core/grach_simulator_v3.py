import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from arch import arch_model
from typing import Tuple, Optional, Dict, Union

# ==========================================
# ⚙️ 1. HYPER-PARAMETERS CONFIGURATION
# ==========================================
def get_optimal_config(asset: str, time_increment: int) -> dict:
    """
    Trả về cấu hình tối ưu để đạt CRPS thấp nhất cho từng Asset/Task.
    
    Arguments:
        time_increment: 300 (5m) hoặc 60 (1m)
    """
    is_crypto = asset.lower() not in ["xau"]
    is_high_freq = time_increment <= 60  # 1 phút (1 Hour Forecast)

    config = {
        "mean_model": "Zero",  # Luôn dùng Zero cho high-freq để giảm nhiễu
        # "dist": "StudentsT",   # Bắt buộc để bắt đuôi dày (Fat-tails)
        "scale": 10000.0,      # Basis points
        "dist": "skewt",      # Skew Student's t tốt hơn Student's t thường
    }

    # --- A. CẤU HÌNH CHO CRYPTO (BTC, ETH, SOL) ---
    if is_crypto:
        # Volatility Model: Crypto cần GJR-GARCH để bắt hiệu ứng sợ hãi (Leverage)
        config["vol_model"] = "GARCH" # 'EGARCH' hay 'GARCH' (với o=1 là GJR)
        config["p"] = 1
        config["o"] = 1  # o=1 kích hoạt GJR-GARCH (Asymmetric term)
        config["q"] = 1
        
        # Simulation Method: FHS là bắt buộc cho Crypto để bắt đuôi béo
        config["simulation_method"] = "FHS" 

        if is_high_freq: # 1 Hour Forecast
            config["lookback_days"] = 14  # Tăng nhẹ lên 14 để FHS có đủ mẫu để bốc
            config["use_drift"] = False   # 1h ngắn quá, drift thường là nhiễu
        else: # 1 Day Forecast
            config["lookback_days"] = 45  # 45 ngày là đủ
            config["use_drift"] = True    # 1 ngày cần có quán tính trend
            config["drift_decay"] = 0.995 # Giảm lực trend 0.5% mỗi bước 5m

    # --- B. CẤU HÌNH CHO COMMODITIES (XAU) ---
    else:
        # Vàng ổn định hơn, GARCH thường là đủ
        config["vol_model"] = "GARCH"
        config["p"] = 1
        config["o"] = 0  # Không cần Asymmetric
        config["q"] = 1
        
        # Vàng tuân theo phân phối toán học tốt hơn, FHS vẫn tốt nhưng Parametric cũng ổn
        # Tuy nhiên FHS vẫn an toàn hơn cho CRPS.
        config["simulation_method"] = "FHS"

        if is_high_freq:
            config["lookback_days"] = 30  # Vàng cần lịch sử dài hơn để học
            config["use_drift"] = False
        else:
            config["lookback_days"] = 90  # 3 tháng
            config["use_drift"] = True
            config["drift_decay"] = 0.999 # Trend của vàng bền hơn Crypto

    return config

# ==========================================
# 🛠️ 2. CORE FUNCTIONS
# ==========================================

def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()

def fit_garch_optimized(returns: pd.Series, config: dict):
    # arch_model tự động xử lý GJR nếu p=1, o=1, q=1
    model = arch_model(returns * config["scale"],
                       mean=config["mean_model"],
                       vol=config["vol_model"],
                       p=config["p"], 
                       o=config["o"], # Asymmetric lag
                       q=config["q"],
                       dist=config["dist"])
    
    try:
        # Tăng iter và đổi method nếu cần
        res = model.fit(disp="off", options={"maxiter": 1000})
    except:
        print("[WARN] GRACH Fit failed, retrying with rescale...")
        res = model.fit(disp="off", options={"maxiter": 1000}, rescale=True)
    return res

def calculate_recent_drift(returns_bps: pd.Series, window: int = 12) -> float:
    """
    Tính quán tính xu hướng (Drift) từ dữ liệu gần nhất.
    Returns: Average return per step (basis points)
    """
    # Lấy trung bình moving average của n phiên gần nhất
    # window=12 (với nến 5m là 1 tiếng, nến 1m là 12 phút)
    return float(returns_bps.tail(window).mean())

def simulate_garch_paths(fitted_res, S0, steps, n_sims, config, seed=None):
    print(f"[INFO] SEED {seed}")
    """
    Hàm mô phỏng nâng cao hỗ trợ:
    1. FHS (Filtered Historical Simulation)
    2. GJR-GARCH Logic
    3. Damped Drift (Trend)
    """
    if seed is not None:
        np.random.seed(seed)

    params = fitted_res.params
    scale = config["scale"]
    
    # 1. Trích xuất tham số GARCH / GJR-GARCH
    mu_model = params.get("mu", 0.0) # Thường là 0 do mean="Zero"
    omega = params.get("omega", 0.01)
    alpha = params.get("alpha[1]", 0.05)
    beta = params.get("beta[1]", 0.90)
    gamma = params.get("gamma[1]", 0.0) # Tham số GJR (chỉ có khi o=1)
    nu = params.get("nu", 100.0)        # Degrees of freedom
    lambda_ = params.get("lambda", 0.0) # Skew parameter (cho skewt)

    # 2. Khởi tạo trạng thái đầu
    last_vol = fitted_res.conditional_volatility.iloc[-1]
    last_shock = fitted_res.resid.iloc[-1]

    sigma_prev = np.full(n_sims, last_vol, dtype=np.float64)
    eps_prev = np.full(n_sims, last_shock, dtype=np.float64)

    # 3. Tạo Shocks (Z)
    # --- PHƯƠNG PHÁP 1: FHS (Bootstrapping Residuals) - KHUYÊN DÙNG ---
    if config["simulation_method"] == "FHS":
        # Lấy residuals chuẩn hóa từ lịch sử fit
        std_resids = fitted_res.std_resid.dropna().values
        # Bốc thăm lại từ quá khứ (với thay thế)
        z = np.random.choice(std_resids, size=(steps, n_sims), replace=True)
        
    # --- PHƯƠNG PHÁP 2: Parametric (Monte Carlo thuần) ---
    else:
        # Logic sinh số Skew-T hoặc T
        if config["dist"] == "skewt":
            from arch.univariate import SkewStudent
            dist_gen = SkewStudent()
            random_values = dist_gen.simulate([nu, lambda_], steps * n_sims)
            z = random_values.reshape((steps, n_sims))
        else:
            z = student_t.rvs(df=nu, size=(steps, n_sims))
            if nu > 2: z /= np.sqrt(nu / (nu - 2.0))

    # 4. Xử lý Drift (Cho task 1 ngày)
    drift_decay = config.get("drift_decay", 1.0)
    recent_trend_bps = config.get("recent_trend_bps", 0.0)

    # 5. Vòng lặp Mô phỏng
    returns_bps = np.zeros((steps, n_sims))
    
    for t in range(steps):
        # A. Cập nhật phương sai (Conditional Variance)
        # GJR-GARCH Term: gamma * eps^2 * I(eps < 0)
        # Indicator I: 1 nếu eps_prev < 0 (tin xấu), 0 nếu eps_prev >= 0
        indicator = (eps_prev < 0).astype(float)
        
        # Công thức GJR-GARCH đầy đủ
        sigma2 = omega + \
                 alpha * (eps_prev**2) + \
                 gamma * (eps_prev**2) * indicator + \
                 beta * (sigma_prev**2)
                 
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))

        # B. Tính Epsilon & Return
        eps_t = sigma_t * z[t, :]
        
        # C. Áp dụng Drift (Damped Trend)
        current_drift = recent_trend_bps * (drift_decay ** t)
        
        # Return = Drift + Shock
        returns_bps[t, :] = mu_model + current_drift + eps_t

        sigma_prev = sigma_t
        eps_prev = eps_t

    # 6. Chuyển đổi về giá
    log_ret = returns_bps / scale
    
    prices = np.zeros((n_sims, steps + 1))
    prices[:, 0] = S0
    
    # Tính giá tích lũy (Cumulative Product)
    # P_t = P_0 * exp(cumsum(r))
    cum_ret = np.cumsum(log_ret, axis=0)
    prices[:, 1:] = S0 * np.exp(cum_ret).T

    return prices

# ==========================================
# 🚀 3. MAIN SIMULATION CONTROLLER
# ==========================================
def simulate_single_price_path_with_garch(prices_dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: Optional[int] = 42):
    """
    Simulate a single price path with GARCH model with advanced features:
    1. FHS (Filtered Historical Simulation)
    2. GJR-GARCH Logic
    3. Damped Drift (Trend)
    - prices_dict: dictionary of prices {"timestamp": "price"}
    - time_increment: time increment in seconds
    - time_length: time length in seconds
    - n_sims: number of simulation paths
    - seed: seed for the random number generator
    Returns:
      prices: ndarray shape (n_sims, steps+1) including initial price at index 0
    """
    # 1. Chuẩn bị dữ liệu
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    
    # 2. Lấy cấu hình tối ưu
    config = get_optimal_config(asset, time_increment)
    
    # 3. Cắt gọt dữ liệu (Lookback Window)
    # Tính số lượng điểm dữ liệu cần lấy dựa trên time_increment
    # Ví dụ: 45 ngày * 24h * (60/5) nến/giờ = số nến
    points_per_day = 86400 // time_increment
    needed_points = int(config["lookback_days"] * points_per_day)
    
    if len(full_prices) > needed_points:
        hist_prices = full_prices.tail(needed_points)
        print(f"[INFO] {asset}: Cắt dữ liệu xuống {config['lookback_days']} ngày gần nhất ({len(hist_prices)} nến).")
    else:
        hist_prices = full_prices
        print(f"[WARN] {asset}: Dữ liệu lịch sử ({len(full_prices)} nến) ngắn hơn mức tối ưu ({needed_points}).")

    # 4. Fit Model
    returns = compute_log_returns(hist_prices)
    print(f"[INFO] Fitting GARCH({config['dist']}) cho {asset}...")
    res = fit_garch_optimized(returns, config)
    
    # Tính toán Drift (Recent Trend) CHÍNH XÁC từ Returns đầu vào
    # Returns này đã được scale (bps), nên drift cũng là bps
    recent_trend_bps = 0.0
    
    if config.get("use_drift", False):
        # Lấy trung bình 12 cây nến gần nhất (1 tiếng nếu nến 5m)
        recent_window = 12 
        recent_trend_bps = float(returns.tail(recent_window).mean())
        
        # <trcik> Nếu trend quá nhỏ (nhiễu), cho về 0 để an toàn
        if abs(recent_trend_bps) < 0.1: # < 0.1 bps per step
            recent_trend_bps = 0.0
        config["recent_trend_bps"] = recent_trend_bps
        
    # Log thông tin để debug
    params = res.params
    print(f"[MODEL] {asset} ({time_increment}s) | Mode: {config['simulation_method']} | Drift: {config.get('use_drift', False)}")
    print(f"        -> Params: Alpha: {params.get('alpha[1]', 0):.3f} | Beta: {params.get('beta[1]', 0):.3f} | Gamma: {params.get('gamma[1]', 0):.3f}")

    # 5. Mô phỏng
    steps = time_length // time_increment
    S0 = float(hist_prices.iloc[-1])
    
    print(f"[INFO] Simulating {n_sims} paths for {time_length/3600}h (steps={steps})...")
    paths = simulate_garch_paths(res, S0, steps, n_sims, config, seed=seed)
    
    return paths

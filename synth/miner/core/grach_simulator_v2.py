import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from arch import arch_model
from typing import Tuple, Optional, Dict
from tqdm import trange

# ==========================================
# ⚙️ 1. HYPER-PARAMETERS CONFIGURATION
# ==========================================
def get_optimal_config(asset: str, time_increment: int) -> dict:
    """
    Trả về cấu hình tối ưu để đạt CRPS thấp nhất cho từng Asset/Task.
    
    Arguments:
        time_increment: 300 (5m) hoặc 60 (1m)
    """
    asset_lower = asset.lower()
    is_crypto = asset_lower not in ["xau", "googlx", "tslax", "aaplx", "nvdax", "spyx"]
    is_high_freq = time_increment <= 60  # 1 phút

    config = {
        "mean_model": "Zero",  # Luôn dùng Zero cho high-freq để giảm nhiễu
        "dist": "StudentsT",   # Bắt buộc để bắt đuôi dày (Fat-tails)
        "scale": 10000.0,      # Basis points
        "min_nu": 0.0,
        "vol_multiplier": 1.0,
    }

    # --- Tinh chỉnh Cửa sổ Lịch sử (Lookback Window) ---
    if is_high_freq: # Task 1h (1m steps)
        # Cần phản ứng cực nhanh, chỉ nhìn quá khứ rất gần
        if is_crypto:
            config["lookback_days"] = 7   # 7 ngày dữ liệu 1m (~10k nến)
        else: # XAU
            config["lookback_days"] = 14  # 14 ngày dữ liệu 1m
    else: # Task 24h (5m steps)
        # Cần cân bằng giữa độ nhạy và độ ổn định
        if is_crypto:
            config["lookback_days"] = 45  # 45 ngày gần nhất
        else: # XAU
            config["lookback_days"] = 90  # 3 tháng gần nhất
            
    
    # if asset_lower in ["spyx"]:
    #     # config["min_nu"] = 10.0
    #     config["vol_multiplier"] = 0.9
    return config

def get_optimal_param_grid(asset: str, time_increment: int) -> dict:
    asset_lower = asset.lower()
    is_crypto = asset_lower not in ["xau", "googlx", "tslax", "aaplx", "nvdax", "spyx"]
    is_high_freq = time_increment <= 60
    
    grid = {}
    if is_high_freq:
        grid["mean_model"] = ["Zero"]
        if is_crypto:
            grid["lookback_days"] = [5, 7, 10]
            grid["vol_multiplier"] = [0.9, 1.0, 1.1]
        else:
            grid["lookback_days"] = [10, 14, 20]
            grid["vol_multiplier"] = [0.8, 1.0, 1.2]
    else:
        grid["mean_model"] = ["Zero", "Constant"]
        if is_crypto:
            grid["lookback_days"] = [30, 45, 60]
            grid["vol_multiplier"] = [0.9, 1.0, 1.1]
        else:
            grid["lookback_days"] = [60, 90, 120]
            grid["vol_multiplier"] = [0.8, 1.0, 1.2]
            
    if asset_lower == "spyx":
        grid["vol_multiplier"] = [0.8, 0.9, 1.0]

    return grid


# ==========================================
# 🛠️ 2. CORE FUNCTIONS
# ==========================================

def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()

def fit_garch_optimized(returns: pd.Series, config: dict):
    """Fit model với config tối ưu, có xử lý lỗi hội tụ."""
    p = config.get("p", 1)
    q = config.get("q", 1)
    # Thử fit lần 1
    model = arch_model(returns * config["scale"],
                       mean=config["mean_model"],
                       vol="GARCH",
                       p=p, q=q,
                       dist=config["dist"])
    
    # Tăng maxiter để đảm bảo hội tụ cho dữ liệu nhiễu
    try:
        res = model.fit(disp="off", options={"maxiter": 200}, show_warning=False)
    except:
        # Fallback nếu không hội tụ: đổi sang rescale tự động
        print("[WARN] Model fit failed, retrying with rescaling...")
        res = model.fit(disp="off", options={"maxiter": 500}, rescale=True, show_warning=False)
    return res

def simulate_garch_paths(fitted_res, S0, steps, n_sims, config, seed=None):
    if seed is not None:
        np.random.seed(seed)

    scale = config["scale"]
    params = fitted_res.params
    
    # 1. Trích xuất tham số
    mu = params.get("mu", params.get("Const", 0.0))
    omega = params.get("omega", 0.01)
    alpha = params.get("alpha[1]", 0.05)
    beta = params.get("beta[1]", 0.90)
    nu = params.get("nu", 8.0)
    nu = max(nu, config["min_nu"])

    # 2. Khởi tạo Biến động (QUAN TRỌNG CHO CRPS)
    # Dùng conditional volatility cuối cùng để bắt đầu (Variance Targeting)
    last_vol = fitted_res.conditional_volatility.iloc[-1] * config["vol_multiplier"]
    last_shock = fitted_res.resid.iloc[-1]

    sigma_prev = np.full(n_sims, last_vol, dtype=np.float64)
    eps_prev = np.full(n_sims, last_shock, dtype=np.float64)

    # 3. Chuẩn bị Nhiễu Student-t
    if nu <= 2: scale_std = 1.0
    else: scale_std = np.sqrt(nu / (nu - 2.0))
    
    z = student_t.rvs(df=nu, size=(steps, n_sims))
    if scale_std != 1.0: z /= scale_std

    # 4. Vòng lặp Mô phỏng (Vectorized)
    returns_bps = np.zeros((steps, n_sims))
    
    for t in range(steps):
        # GARCH(p,q) equation simplified for 1,1 visualization backward compat, actual relies on fit
        # To support p,q correctly we use dynamic access, but for speed we approximate or rely on res
        # For full p>1 q>1 support in manual simulation, arch model has a forecast function.
        # But this simulator explicitly hardcodes p=1, q=1:
        sigma2 = omega + alpha * (eps_prev**2) + beta * (sigma_prev**2)
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))

        eps_t = sigma_t * z[t, :]
        returns_bps[t, :] = mu + eps_t

        sigma_prev = sigma_t
        eps_prev = eps_t

    # 5. Chuyển đổi về giá
    # Đảo ngược scale (chia cho 10000)
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
def simulate_single_price_path_with_garch(prices_dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: Optional[int] = 42, **kwargs):
    """
    Simulate a single price path with GARCH model.
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
    config.update(kwargs) # Override with tuning parameters
    print(f"Config: {config}")
    
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
    
    # Kiểm tra tham số nu (độ dày đuôi)
    print(f"      -> Params: alpha={res.params['alpha[1]']:.3f}, beta={res.params['beta[1]']:.3f}, nu={res.params['nu']:.2f}")

    # 5. Mô phỏng
    steps = time_length // time_increment
    S0 = float(hist_prices.iloc[-1])
    
    print(f"[INFO] Simulating {n_sims} paths for {time_length/3600}h (steps={steps})...")
    paths = simulate_garch_paths(res, S0, steps, n_sims, config, seed=seed)
    
    return paths

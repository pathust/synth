import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import SkewStudent, StudentsT, Normal
from typing import List, Optional
import warnings

# Tắt cảnh báo hội tụ để log sạch hơn
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ 1. CONFIGURATION (ENSEMBLE STRATEGY)
# ==========================================
def get_ensemble_configs(asset: str, time_increment: int) -> List[dict]:
    """
    Trả về danh sách cấu hình (Short-term & Long-term).
    """
    is_crypto = asset.lower() not in ["xau", "gold"]
    is_high_freq = time_increment <= 60 
    
    configs = []

    # --- Strategy A: Short-term / Fast React (Bắt biến động tức thời) ---
    cfg_short = {
        "name": "Short_Term",
        "mean_model": "Constant", # Bắt drift nhẹ ngắn hạn
        "dist": "skewstudent",    # Bắt lệch (Skewness)
        "scale": 10000.0,
        # Crypto biến động nhanh nên lookback ngắn hơn Vàng
        "lookback_days": 5 if (is_crypto and is_high_freq) else 14, 
        "weight": 0.4             # 40% số paths
    }
    configs.append(cfg_short)

    # --- Strategy B: Long-term / Structural (Định hình biên độ chuẩn) ---
    cfg_long = {
        "name": "Long_Term",
        "mean_model": "Zero",     # Mean Reversion dài hạn về 0
        "dist": "skewstudent",
        "scale": 10000.0,
        "lookback_days": 30 if (is_crypto and is_high_freq) else 60,
        "weight": 0.6             # 60% số paths
    }
    configs.append(cfg_long)
    
    return configs

# ==========================================
# 🛠️ 2. CORE FUNCTIONS
# ==========================================

def compute_log_returns(prices: pd.Series) -> pd.Series:
    # Fill gap nếu có trước khi tính returns để tránh shock ảo
    return np.log(prices.ffill()).diff().dropna()

def fit_garch_robust(returns: pd.Series, config: dict):
    """Fit model với cơ chế Fallback nếu lỗi."""
    scaled_returns = returns * config["scale"]
    
    # Định nghĩa Model
    model = arch_model(scaled_returns,
                       mean=config["mean_model"],
                       vol="GARCH",
                       p=1, q=1,
                       dist=config["dist"])
    
    # Chiến lược Fit: Thử chuẩn -> Thử rescale -> Fallback về Normal
    try:
        res = model.fit(disp="off", options={"maxiter": 200})
    except:
        try:
            # Thử rescale tự động của thư viện
            res = model.fit(disp="off", rescale=True)
        except:
            # Nếu SkewStudent lỗi (khó hội tụ), fallback về Normal
            print(f"[WARN] SkewStudent failed for {config['name']}, falling back to Normal.")
            model = arch_model(scaled_returns, mean=config["mean_model"], vol="GARCH", p=1, q=1, dist="Normal")
            res = model.fit(disp="off")
            
    return res

def simulate_paths_vectorized(fitted_res, S0, steps, n_sims, scale, dist_name):
    """
    Mô phỏng GARCH(1,1) Vectorized sử dụng PPF (Inverse CDF) để tối ưu tốc độ.
    """
    params = fitted_res.params
    
    # 1. Trích xuất tham số GARCH
    mu = params.get("mu", params.get("Const", 0.0))
    omega = params.get("omega", 0.01)
    alpha = params.get("alpha[1]", 0.05)
    beta = params.get("beta[1]", 0.90)
    
    # Tham số phân phối (Shape parameters)
    nu = params.get("nu", 8.0)
    lambda_skew = params.get("lambda", 0.0)

    # 2. Khởi tạo trạng thái ban đầu (Variance Targeting)
    last_vol = fitted_res.conditional_volatility.iloc[-1]
    last_resid = fitted_res.resid.iloc[-1]
    
    sigma_prev = np.full(n_sims, last_vol, dtype=np.float64)
    eps_prev = np.full(n_sims, last_resid, dtype=np.float64)

    # 3. Sinh Nhiễu (Noise Generation) - SỬ DỤNG PPF ĐỂ FIX LỖI & TĂNG TỐC
    # Sinh mảng ngẫu nhiên Uniform [0, 1] cho toàn bộ ma trận
    total_draws = steps * n_sims
    random_uniform = np.random.random(total_draws) 
    
    print(f"dist_name: {dist_name}")
    dn = dist_name.lower()
    # Kiểm tra bằng từ khóa chứa trong tên (nhận diện: "Standardized Skew Student's t")
    if "skew" in dn:
        dist_model = SkewStudent()
        z_flat = dist_model.ppf(random_uniform, [nu, lambda_skew])
        print("Using SkewStudent")
    elif "student" in dn: # Nhận diện: "Standardized Student's t"
        dist_model = StudentsT()
        z_flat = dist_model.ppf(random_uniform, [nu])
        print("Using StudentsT")
    else: # Normal hoặc bất kỳ loại nào khác
        z_flat = np.random.standard_normal(total_draws)
        print("Using Normal")
    # Reshape về (steps, n_sims)
    z = z_flat.reshape(steps, n_sims)

    # 4. Vòng lặp Mô phỏng (Vectorized over n_sims)
    returns_bps = np.zeros((steps, n_sims))
    
    for t in range(steps):
        # GARCH(1,1) Equation
        sigma2 = omega + alpha * (eps_prev**2) + beta * (sigma_prev**2)
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))
        
        # Innovation
        eps_t = sigma_t * z[t, :]
        returns_bps[t, :] = mu + eps_t
        
        # Update trạng thái
        sigma_prev = sigma_t
        eps_prev = eps_t

    # 5. Chuyển đổi về Giá
    log_ret = returns_bps / scale
    prices = np.zeros((n_sims, steps + 1))
    prices[:, 0] = S0
    cum_ret = np.cumsum(log_ret, axis=0)
    prices[:, 1:] = S0 * np.exp(cum_ret).T
    
    return prices
# ==========================================
# 🚀 3. MAIN CONTROLLER
# ==========================================
def simulate_single_price_path_with_garch(prices_dict, asset: str, time_increment: int, time_length: int, n_sims: int = 1000, seed: Optional[int] = 42):
    if seed is not None:
        np.random.seed(seed)

    # 1. Prepare Data
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    S0 = float(full_prices.iloc[-1])
    steps = time_length // time_increment
    
    # 2. Get Ensemble Configs
    configs = get_ensemble_configs(asset, time_increment)
    
    final_paths_list = []
    total_sims_generated = 0
    
    print(f"[INFO] Starting Ensemble Simulation for {asset} (S0={S0:.2f})...")

    for i, cfg in enumerate(configs):
        # Tính số lượng path cho config này
        if i == len(configs) - 1:
            n_sub_sims = n_sims - total_sims_generated # Đảm bảo tổng chẵn 1000
        else:
            n_sub_sims = int(n_sims * cfg["weight"])
        
        total_sims_generated += n_sub_sims
        
        # 3. Lookback Slicing (Cắt dữ liệu theo config)
        points_per_day = 86400 // time_increment
        needed_points = int(cfg["lookback_days"] * points_per_day)
        
        if len(full_prices) > needed_points:
            hist_prices = full_prices.tail(needed_points)
        else:
            hist_prices = full_prices
        
        # 4. Fit Model & Simulate
        returns = compute_log_returns(hist_prices)
        res = fit_garch_robust(returns, cfg)
        
        # Lấy tên distribution thực tế model đã dùng (phòng trường hợp fallback)
        actual_dist = res.model.distribution.name
        skew_param = res.params.get('lambda', 0)
        
        print(f"   -> [{cfg['name']}] Fit {actual_dist} (Skew={skew_param:.2f}). Simulating {n_sub_sims} paths...")
        
        paths = simulate_paths_vectorized(res, S0, steps, n_sub_sims, cfg["scale"], actual_dist)
        final_paths_list.append(paths)

    # 5. Combine & Shuffle
    # Gộp list các mảng (n_sub_sims, steps+1) thành (n_sims, steps+1)
    ensemble_paths = np.vstack(final_paths_list)
    
    # Xáo trộn thứ tự để ngẫu nhiên hóa vị trí (quan trọng nếu visualize hoặc lấy mẫu)
    # np.random.shuffle(ensemble_paths)
    
    return ensemble_paths
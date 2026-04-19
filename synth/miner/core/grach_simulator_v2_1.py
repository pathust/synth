import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from arch import arch_model
from typing import Optional, Dict
from synth.miner.regime import (
    REGIME_TYPE, 
    detect_market_regime_with_er, 
    detect_market_regime_with_bbw
)
from synth.miner.constants import STOCK_ASSETS

# ==========================================
# ⚙️ 1. HYPER-PARAMETERS CONFIGURATION
# ==========================================
def get_optimal_config(asset: str, time_increment: int, regime_info: dict) -> dict:
    """
    Trả về cấu hình tối ưu để đạt CRPS thấp nhất cho từng Asset/Task.
    
    Arguments:
        time_increment: 300 (5m) hoặc 60 (1m)
    """

    

    is_crypto = asset.lower() not in ["xau"]

    regime_er = regime_info["er"]
    regime_bbw = regime_info["bbw"]
    if asset.upper() in STOCK_ASSETS:
        return get_stock_refined_config(asset, time_increment, regime_bbw)

    regime_for_asset = regime_er if is_crypto else regime_bbw

    asset_upper = asset.upper()
    config = {
        # "mean_model": "Zero",  # Luôn dùng Zero cho high-freq để giảm nhiễu
        "dist": "StudentsT",   # Bắt buộc để bắt đuôi dày (Fat-tails)
        "scale": 10000.0,      # Basis points
        "min_nu": 6.0,          # Sàn của bậc tự do
        "vol_multiplier": 1.0,  # Hệ số co giãn volatility
        "mean_model": "Constant",
        "grach_o": 0
    }

    # Cửa sổ lookback theo từng đồng (đồng bộ garch_simulator_v2_2.get_optimal_config)
    ti = time_increment
    if asset_upper == "BTC":
        config["lookback_days"] = 3.0 if ti <= 60 else 30.0
    elif asset_upper == "ETH":
        config["lookback_days"] = 3.0 if ti <= 60 else 30.0
    elif asset_upper == "HYPE":
        config["lookback_days"] = 3.0 if ti <= 60 else 30.0
    elif asset_upper == "XRP":
        config["lookback_days"] = 3.0 if ti <= 60 else 30.0
    elif asset_upper == "SOL":
        config["lookback_days"] = 2.5 if ti <= 60 else 20.0
    elif asset_upper == "XAU":
        config["lookback_days"] = 3.9 if ti <= 60 else 30.0
    else:
        config["lookback_days"] = 3.0

    if asset.lower() in ["xau"]:
        config["min_nu"] = 10.0
        if regime_er["type"] == REGIME_TYPE.SIDEWAYS:
            config["mean_model"] = "Zero" # Ép về 0 tuyệt đối khi đi ngang
            config["vol_multiplier"] = 0.90 # Thu hẹp dải dự báo 10% để ăn điểm Sharpness

    
        # if regime_bbw["is_squeeze"] and not regime_bbw["is_trending"]:
        #     # Sideways cực đoan -> Ép mạnh để ăn điểm Sharpness
        #     config["mean_model"] = "Zero" # Drift = 0
        #     config["min_nu"] = 15.0       # Ép phân phối gần như Normal
        #     config["vol_multiplier"] = 0.85 # Giảm 8% vol dự báo
        
        # elif regime_bbw["is_trending"]:
        #     # Đang có trend -> Dùng Constant để bắt Drift, thả lỏng vol
        #     config["mean_model"] = "Constant"
        #     config["min_nu"] = 8.0 # Cho phép đuôi dày hơn chút để bắt shock
        #     config["vol_multiplier"] = 1.0
            
        # elif regime_er["type"] == REGIME_TYPE.SIDEWAYS:
        #     config["mean_model"] = "Zero"
        #     config["vol_multiplier"] = 0.9


    elif asset.lower() in ["sol"]:
        config["grach_o"] = 1
        config["min_nu"] = 4.5       # Cho phép đuôi dày (fat tails)
        
        # if regime_for_asset["type"] == REGIME_TYPE.TRENDING:
        #     config["vol_multiplier"] = 1.05 # Nới rộng vol khi có trend mạnh
        # else:
        #     config["mean_model"] = "Zero"

        if regime_bbw["is_trending"]:
            # Khi SOL vào trend, nó chạy rất điên -> Nới lỏng Vol
            config["vol_multiplier"] = 1.05 
            config["mean_model"] = "Constant"
        elif regime_bbw["is_squeeze"]:
            # Khi SOL nén, nó nén rất chặt trước khi nổ -> Giảm vol vừa phải
            config["vol_multiplier"] = 0.98
            config["mean_model"] = "Zero"

    elif asset.lower() in ["btc", "eth"]:
        config["grach_o"] = 1
        config["min_nu"] = 4.0
        if ti <= 60:
            config["lookback_days"] = 5.0
        if regime_for_asset["type"] == REGIME_TYPE.SIDEWAYS:
            config["vol_multiplier"] = 0.90
            config["mean_model"] = "Zero"
        elif regime_for_asset["type"] == REGIME_TYPE.TRENDING:
            config["vol_multiplier"] = 0.95
            config["mean_model"] = "Zero"
        else:
            config["vol_multiplier"] = 0.85
            config["mean_model"] = "Zero"

    elif asset.lower() == "hype":
        config["grach_o"] = 1
        config["min_nu"] = 4.0
        if ti <= 60:
            config["lookback_days"] = 5.0
        if regime_for_asset["type"] == REGIME_TYPE.SIDEWAYS:
            config["vol_multiplier"] = 0.90
            config["mean_model"] = "Zero"
        elif regime_for_asset["type"] == REGIME_TYPE.TRENDING:
            config["vol_multiplier"] = 0.95
            config["mean_model"] = "Zero"
        else:
            config["vol_multiplier"] = 0.85
            config["mean_model"] = "Zero"

    elif asset.lower() == "xrp":
        config["grach_o"] = 1
        config["min_nu"] = 4.0
        if ti <= 60:
            config["lookback_days"] = 5.0
        if regime_for_asset["type"] == REGIME_TYPE.SIDEWAYS:
            config["vol_multiplier"] = 0.90
            config["mean_model"] = "Zero"
        elif regime_for_asset["type"] == REGIME_TYPE.TRENDING:
            config["vol_multiplier"] = 0.95
            config["mean_model"] = "Zero"
        else:
            config["vol_multiplier"] = 0.85
            config["mean_model"] = "Zero"

    return config

def get_stock_refined_config(asset: str, time_increment: int, regime: dict) -> dict:
    """
    Cấu hình tối ưu cho Chứng khoán Mỹ (Tokenized Stocks).
    Tập trung vào GJR-GARCH và hiệu ứng giờ giao dịch.
    """
    is_high_freq = time_increment <= 60
    
    # Cấu hình chung cho Stock: Luôn dùng GJR-GARCH (o=1)
    config = {
        "mean_model": "Constant",
        "dist": "studentst", 
        "scale": 10000.0,
        "vol_multiplier": 1.0,
        "grach_o": 1,   # BẮT BUỘC: GJR-GARCH để bắt Leverage Effect
        "min_nu": 8.0   # Stock thường có đuôi mỏng hơn Crypto
    }

    # ==========================================
    # 🏛️ 1. SPYX (S&P 500 Index) - Low Volatility
    # ==========================================
    if asset == "SPYX":
        # Đồng bộ v2_2: SPYX low — lookback 15 ngày
        config["lookback_days"] = 15.0
        
        # Nếu thị trường đang nén (hoặc ngoài giờ giao dịch)
        if regime["is_squeeze"]:
            config["mean_model"] = "Zero" # Index đi ngang quanh tham chiếu
            config["vol_multiplier"] = 0.85 # Giảm vol cực mạnh
            config["min_nu"] = 15.0 # Ép về Normal Distribution
        else:
            config["vol_multiplier"] = 0.95 # Mặc định vẫn giảm nhẹ vol
            config["min_nu"] = 10.0

    # ==========================================
    # 🚀 2. NVDAX, TSLAX (High Growth/Vol)
    # ==========================================
    elif asset in ["NVDAX", "TSLAX"]:
        config["lookback_days"] = 20.0
        
        # Chấp nhận đuôi dày (Fat tails)
        config["min_nu"] = 2.0 
        
        if regime["is_trending"]:
            config["vol_multiplier"] = 1.05 # Nới lỏng khi vào sóng
            config["mean_model"] = "Constant"
        elif regime["is_squeeze"]: 
            # Off-hours của NVDA/TSLA vẫn có thể biến động do tin tức
            config["vol_multiplier"] = 0.95 
            config["mean_model"] = "Zero"
            config["min_nu"] = 8.0

    # ==========================================
    # 🏢 3. AAPLX, GOOGLX (Big Tech Bluechip)
    # ==========================================
    elif asset in ["AAPLX", "GOOGLX"]:
        # v2_2: AAPLX 30 ngày, GOOGLX 20 ngày
        config["lookback_days"] = 30.0 if asset == "AAPLX" else 20.0
        config["min_nu"] = 8.0
        
        if regime["is_squeeze"]:
            config["vol_multiplier"] = 0.90
            config["mean_model"] = "Zero"
            config["min_nu"] = 12.0 # Khá an toàn
        else:
            config["vol_multiplier"] = 1.0

    return config


def get_optimal_param_grid(asset: str, time_increment: int) -> Dict[str, list]:
    """
    Hyperparameter grid for tuning (merged on top of ``get_optimal_config`` at runtime).
    Keys must match fields used in ``fit_garch_optimized`` / ``simulate_garch_paths``.
    """
    asset_upper = asset.upper()
    is_high = time_increment <= 60
    
    # 🌟 Reduced from combinatorial explosion to essential pairs
    grid: Dict[str, list] = {
        "min_nu": [6.0, 8.0],
        "vol_multiplier": [0.95, 1.0],
        "grach_o": [1],  # Always use GJR-GARCH asymmetric effect 
        "mean_model": ["Constant", "Zero"],
    }
    
    if asset_upper in ("BTC", "ETH"):
        grid["lookback_days"] = [3.0, 5.0] if is_high else [25.0, 35.0]
    elif asset_upper == "HYPE":
        grid["lookback_days"] = [3.0, 5.0] if is_high else [25.0, 35.0]
    elif asset_upper == "XRP":
        grid["lookback_days"] = [3.0, 5.0] if is_high else [25.0, 35.0]
    elif asset_upper == "SOL":
        grid["lookback_days"] = [2.5, 5.0] if is_high else [12.0, 20.0]
        grid["vol_multiplier"] = [1.0, 1.05]
    elif asset_upper in ("XAU", "GOLD"):
        grid["lookback_days"] = [3.0, 5.0] if is_high else [30.0, 45.0]
    elif asset_upper in STOCK_ASSETS:
        grid["lookback_days"] = [5.0, 10.0] if is_high else [10.0, 20.0]
        grid["min_nu"] = [8.0, 12.0]
    else:
        grid["lookback_days"] = [5.0] if is_high else [20.0]

    return grid


# ==========================================
# 🛠️ 2. CORE FUNCTIONS
# ==========================================

def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()

def fit_garch_optimized(returns: pd.Series, config: dict):
    """Fit model với config tối ưu, có xử lý lỗi hội tụ."""
    print(f"Fitting GARCH({config['dist']} - config: {config['mean_model']})...")
    # Thử fit lần 1
    model = arch_model(returns * config["scale"],
                       mean=config["mean_model"],
                       vol="GARCH",
                       p=1, q=1, o=config["grach_o"],
                       dist=config["dist"])
    
    # Tăng maxiter để đảm bảo hội tụ cho dữ liệu nhiễu
    try:
        res = model.fit(disp="off", options={"maxiter": 200})
    except:
        # Fallback nếu không hội tụ: đổi sang rescale tự động
        print("[WARN] Model fit failed, retrying with rescaling...")
        res = model.fit(disp="off", options={"maxiter": 500}, rescale=True)
    return res

def simulate_garch_paths(fitted_res, S0, steps, n_sims, config, seed=None):
    if seed is not None:
        np.random.seed(seed)

    params = fitted_res.params
    print("PARAMS: ", params)
    # 1. Trích xuất tham số
    mu = params.get("mu", params.get("Const", 0.0))
    omega = params.get("omega", 0.01)
    alpha = params.get("alpha[1]", 0.05)
    beta = params.get("beta[1]", 0.90)

    # --- QUAN TRỌNG: Ràng buộc bậc tự do (Degrees of Freedom) ---
    # Model thường ước lượng nu quá thấp (đuôi quá dày) làm hỏng CRPS
    estimated_nu = params.get("nu", 8.0)
    nu = max(estimated_nu, config["min_nu"])
    print(f"Config: {config}, Estimated nu: {estimated_nu}, Final nu: {nu}")
 
    if config["grach_o"] == 1:
        # Tham số bất đối xứng (Gamma) - Chỉ có khi o=1
        gamma = params.get("gamma[1]", 0.0)

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
        if config["grach_o"] == 1:
            # Tính thành phần bất đối xứng: I[eps < 0] * eps^2
            # Nếu shock âm (eps < 0) -> return 1, ngược lại 0
            indicator = (eps_prev < 0).astype(float)
            term_shock = (alpha + gamma * indicator) * (eps_prev**2)
            sigma2 = omega + term_shock + beta * (sigma_prev**2)
        else:
            # GARCH(1,1) equation with o=0
            sigma2 = omega + alpha * (eps_prev**2) + beta * (sigma_prev**2)
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))
        eps_t = sigma_t * z[t, :]
        returns_bps[t, :] = mu + eps_t
        sigma_prev = sigma_t
        eps_prev = eps_t

    # 5. Chuyển đổi về giá
    # Đảo ngược scale (chia cho 10000)
    log_ret = returns_bps / config["scale"]
    
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
def simulate_single_price_path_with_garch(
    prices_dict,
    asset: str,
    time_increment: int,
    time_length: int,
    n_sims: int,
    seed: Optional[int] = 42,
    **kwargs,
):
    """
    Simulate a single price path with GARCH(1,1) model.
    - prices_dict: dictionary of prices {"timestamp": "price"}
    - time_increment: time increment in seconds
    - time_length: time length in seconds
    - n_sims: number of simulation paths
    - seed: seed for the random number generator
    - kwargs: optional overrides merged into the regime-based config (tuning).
    Returns:
      prices: ndarray shape (n_sims, steps+1) including initial price at index 0
    """
    # 1. Chuẩn bị dữ liệu
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    
    # 2. Lấy cấu hình tối ưu
    # Detect Regime
    regime_er = detect_market_regime_with_er(full_prices, lookback=20)
    regime_bbw = detect_market_regime_with_bbw(full_prices)
    regime_info = {
        "er": regime_er,
        "bbw": regime_bbw
    }
    config = get_optimal_config(asset, time_increment, regime_info)
    if kwargs:
        config.update(kwargs)
    print(f"Regime: {regime_info} \n Config: {config}")
    
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

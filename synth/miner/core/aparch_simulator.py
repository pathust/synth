import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import StudentsT
from typing import Optional

from synth.miner.core.regime_detection import detect_market_regime_with_bbw, detect_market_regime_with_er



# ==========================================
# ⚙️ 1. CONFIGURATION FOR APARCH
# ==========================================
def get_aparch_config(asset: str, time_increment: int, regime: dict) -> dict:
    """
    Cấu hình chuyên biệt cho APARCH.
    """
    is_high_freq = time_increment <= 60
    config = {
        "mean_model": "Constant",
        "dist": "studentst", 
        "scale": 10000.0,
        "vol_multiplier": 1.0,
        "min_nu": 2.5 if is_high_freq else 4.0
    }
    # --- TINH CHỈNH LOOKBACK ---
    # APARCH cần nhiều dữ liệu hơn chút để fit tham số Delta
    if asset == "BTC":
        config["lookback_days"] = 10 if is_high_freq else 45
        if regime["is_squeeze"]:
            # BTC đi ngang (Sideways) -> Đây là lúc ăn điểm Sharpness dễ nhất
            # Giảm Vol mạnh tay hơn, Mean về 0, ép Nu cao lên (vì đang ít biến động)
            config["vol_multiplier"] = 0.95 
            config["mean_model"] = "Zero"
            # Khi squeeze, BTC chạy như Normal distribution -> Ép nu cao
            config["min_nu"] = 6.0 
            
        elif regime["is_trending"]:
            # BTC vào Trend -> Nó chạy rất lì -> Vol chuẩn, Nu chuẩn (để bắt đuôi)
            config["vol_multiplier"] = 1.05 
            config["mean_model"] = "Constant"
            # Trend mạnh thì đuôi dày là bình thường, trả về min_nu gốc (4.5/6.0)
        
        else:
            # Trạng thái bình thường -> Giảm nhẹ vol để an toàn
            config["vol_multiplier"] = 0.96

    elif asset == "SOL":
        config["lookback_days"] = 14 if is_high_freq else 40
        config["min_nu"] = 4.5 # Cho phép đuôi dày
        # Nếu SOL đang trend mạnh, nới lỏng vol
        if regime.get("is_trending", False):
            config["vol_multiplier"] = 1.05
    else: # XAU / ETH
        config["lookback_days"] = 30 if is_high_freq else 90
        if asset == "XAU": 
            config["min_nu"] = 10.0
            config["vol_multiplier"] = 0.95 # XAU ổn định
    print("Config: ", config)
    return config

# ==========================================
# 🚀 2. SIMULATION CORE (APARCH VECTORIZED)
# ==========================================
def simulate_aparch_optimized(prices_dict, asset, time_increment, time_length, n_sims=1000, seed: Optional[int] = 42):
    if seed is not None:
        np.random.seed(seed)
    # 1. Prepare Data
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    S0 = float(full_prices.iloc[-1])
    steps = time_length // time_increment
    
    # 2. Simple Regime Detect (Tích hợp nhanh để lấy config)
    regime = detect_market_regime_with_bbw(full_prices)
    
    cfg = get_aparch_config(asset, time_increment, regime)
    
    # 3. Fit APARCH Model
    points = int(cfg["lookback_days"] * 86400 // time_increment)
    hist_prices = full_prices.tail(points)
    returns = np.log(hist_prices.ffill()).diff().dropna() * cfg["scale"]

    # vol='APARCH' (Thay vì Garch)
    # p=1, q=1 is standard
    model = arch_model(returns, mean=cfg["mean_model"], vol='APARCH', p=1, o=1, q=1, dist='studentst')
    
    try:
        res = model.fit(disp="off", show_warning=False)
    except:
        # Fallback về GARCH thường nếu APARCH không hội tụ (ít khi xảy ra nhưng cần an toàn)
        print(f"[WARN] APARCH failed for {asset}, using GJR-GARCH.")
        model = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='studentst')
        res = model.fit(disp="off", rescale=True)
        # Gán delta = 2 để logic mô phỏng bên dưới hoạt động như GJR
        res.params["delta"] = 2.0 

    # 4. Extract Params
    mu = res.params.get("mu", 0.0)
    omega = res.params.get("omega", 0.01)
    alpha = res.params.get("alpha[1]", 0.05)
    beta = res.params.get("beta[1]", 0.90)
    gamma = res.params.get("gamma[1]", 0.0) # Leverage parameter
    delta = res.params.get("delta", 2.0)    # Power parameter (Lũy thừa)
    
    nu = max(res.params.get("nu", 8.0), cfg["min_nu"])
    print(f"Params: {res.params}, nu: {nu}")
    # 5. Initialize Simulation
    # APARCH làm việc với sigma^delta, không phải sigma^2
    last_vol = res.conditional_volatility.iloc[-1] * cfg["vol_multiplier"]
    last_resid = res.resid.iloc[-1]
    
    # Lưu trữ sigma (độ lệch chuẩn) để dễ tính toán, khi vào công thức thì mũ delta lên
    sigma_prev = np.full(n_sims, last_vol)
    eps_prev = np.full(n_sims, last_resid)

    # 6. Generate Noise
    dist_sampler = StudentsT()
    z = dist_sampler.ppf(np.random.random((steps, n_sims)), [nu])

    returns_bps = np.zeros((steps, n_sims))

    # 7. APARCH Simulation Loop
    for t in range(steps):
        # Công thức APARCH:
        # sigma_t^delta = omega + alpha * (|eps_{t-1}| - gamma * eps_{t-1})^delta + beta * sigma_{t-1}^delta
        
        # Term 1: Shock component (Asymmetric)
        # abs(eps) - gamma * eps
        shock_val = np.abs(eps_prev) - gamma * eps_prev
        
        # Mũ delta
        shock_term = alpha * (shock_val ** delta)
        
        # Persistence component
        persist_term = beta * (sigma_prev ** delta)
        
        # Tổng hợp sigma^delta
        sigma_delta = omega + shock_term + persist_term
        
        # Chuyển về sigma (căn bậc delta)
        # Cần maximum để tránh số âm/NaN (dù lý thuyết APARCH luôn dương)
        sigma_t = np.maximum(sigma_delta, 1e-12) ** (1.0 / delta)
        
        # Tính Returns
        eps_t = sigma_t * z[t, :]
        returns_bps[t, :] = mu + eps_t
        
        # Update
        sigma_prev = sigma_t
        eps_prev = eps_t

    # 8. Convert Result
    log_ret = returns_bps / cfg["scale"]
    paths = S0 * np.exp(np.cumsum(log_ret, axis=0)).T
    
    return np.column_stack([np.full(n_sims, S0), paths])
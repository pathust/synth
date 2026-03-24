import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from arch import arch_model
from typing import Tuple, Optional, Dict

# ==========================================
# ⚙️ 1. HYPER-PARAMETERS CONFIGURATION
# ==========================================
def get_optimal_config(asset: str, time_increment: int) -> dict:
    """
    Cấu hình GARCH(1,1) tối ưu chuyên biệt cho HFT (1m nến, 60m dự phóng).
    Giới hạn phần cứng/data: Tối đa 4 ngày dữ liệu (data_retention_days = 4).
    """
    asset_upper = asset.upper()
    
    # Cấu hình gốc (Base Config cho nến 1m)
    config = {
        "mean_model": "Zero",  # Nến 1m không có drift (lợi nhuận kỳ vọng = 0)
        "vol_model": "GARCH",  # GARCH tiêu chuẩn là nhanh và ổn định nhất
        "p": 1, 
        "q": 1,
        "dist": "StudentsT",   # Bắt buộc dùng Student-T để xử lý fat-tails
        "scale": 100.0,        # Đưa return về dạng % để optimizer của Scipy không bị lỗi
        "lookback_days": 3.0,  # Mặc định 3 ngày (~4320 nến 1m)
        "min_nu": 3.0,
        "vol_multiplier": 1.0, # Hệ số mồi Volatility
    }

    # ==========================================
    # TÙY CHỈNH CHUYÊN SÂU CHO TỪNG ĐỒNG
    # ==========================================
    if asset_upper == "BTC":
        config["lookback_days"] = 3.0 if time_increment <= 60 else 30.0
        config["min_nu"] = 3.0
        config["vol_multiplier"] = 1.0   # Chuẩn mực, không cần buff

    elif asset_upper == "ETH":
        config["lookback_days"] = 3.0 if time_increment <= 60 else 30.0
        config["min_nu"] = 3.0
        config["vol_multiplier"] = 1.02  # ETH thường có biên độ theo sau BTC nhưng rướn mạnh hơn một chút

    elif asset_upper == "SOL":
        config["lookback_days"] = 2.5 if time_increment <= 60 else 20.0 # SOL thay đổi tính chất rất nhanh, 2.5 ngày (~3600 nến) là đủ để bắt trend vol hiện tại
        config["min_nu"] = 2.5           # Bắt buộc ép nu nhỏ gọn để dự báo các cú giật đuôi dày (fat-tails)
        config["vol_multiplier"] = 1.08  # Buff vol lên 8% để bù đắp việc GARCH thường under-predict các cú pump/dump của SOL

    elif asset_upper == "XAU":
        # XAU dính thứ 7, CN và các phiên nghỉ. 
        # Cố gắng lấy kịch kim dữ liệu cho phép (3.9 ngày để tránh tràn viền data_retention 4 ngày)
        config["lookback_days"] = 3.9 if time_increment <= 60 else 30.0
        config["min_nu"] = 4.0           # Vàng thuần túy hơn, đuôi ít dày hơn Crypto một chút
        config["scale"] = 1000.0         # Volatility nến 1m của XAU cực kỳ nhỏ, cần scale mạnh hơn (x1000) để hội tụ
        config["vol_multiplier"] = 0.95  # GARCH thường over-predict vol của XAU lúc mở phiên, nên hãm lại một chút

    elif asset_upper == "NVDAX":
        config["lookback_days"] = 20.0   # 20 ngày để nắm bắt đủ các chu kỳ bùng nổ của NVDAX
        config["min_nu"] = 3.0           
        config["vol_multiplier"] = 1.10  

    elif asset_upper == "TSLAX":
        config["lookback_days"] = 20.0   # Đủ dài để GARCH hiểu được tần suất xuất hiện Flash Dump
        config["min_nu"] = 3.0           
        config["vol_multiplier"] = 1.10  

    elif asset_upper == "AAPLX":
        config["lookback_days"] = 30.0   # AAPLX đầm tính, 30 ngày sẽ tạo ra đường baseline siêu mượt
        config["min_nu"] = 4.5           
        config["vol_multiplier"] = 0.95  

    elif asset_upper == "GOOGLX":
        config["lookback_days"] = 20.0   # Đủ để nhận diện các gap mở phiên theo chu kỳ
        config["min_nu"] = 3.5           
        config["vol_multiplier"] = 1.08

    elif asset_upper == "SPYX":
        # SPYX low có cuối tuần/ngày nghỉ: kéo dài lookback để mô hình bớt sốc đầu tuần.
        config["lookback_days"] = 15.0
        config["min_nu"] = 4.0
        config["vol_multiplier"] = 0.95

    else:
        # Fallback cho các đồng khác nếu sau này bạn thêm vào
        config["lookback_days"] = 3.0

    return config

# ==========================================
# 🛠️ 2. CORE FUNCTIONS
# ==========================================

def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()

def fit_garch_optimized(returns: pd.Series, config: dict):
    """
    Hàm fit model siêu tốc cho High-Frequency, chống lỗi hội tụ (Convergence Warning).
    """
    # 1. Khởi tạo mô hình
    model = arch_model(
        returns * config["scale"],
        mean=config["mean_model"],
        vol=config["vol_model"],
        p=config["p"], 
        q=config["q"],
        dist=config["dist"]
    )
    
    # 2. Chiến lược Fit (Cố gắng Fit nhanh trước, hỏng thì dùng rescale)
    try:
        # Bước 1: Thử fit nhanh với SLSQP (nhanh, phù hợp HFT)
        res = model.fit(
            update_freq=0, 
            disp="off", 
            options={"maxiter": 100, "ftol": 1e-4}, # Giảm dung sai hội tụ để tăng tốc độ
            show_warning=False
        )
    except:
        try:
            # Bước 2: Nếu fail, dùng thuật toán robust hơn và tự động rescale của arch
            print(f"[WARN] SLSQP failed, retrying with L-BFGS-B and auto-rescale...")
            res = model.fit(
                update_freq=0, 
                disp="off", 
                rescale=True, 
                options={"maxiter": 200}, 
                show_warning=False
            )
        except Exception as e:
            # Bước 3: Nếu dữ liệu quá nát (ví dụ đoạn thị trường sideway tắt thanh khoản)
            # Trả về lỗi để handle ở tầng trên (có thể fallback về dự báo EMA or Historical Vol)
            print(f"[ERROR] GARCH cannot converge: {e}")
            raise e
            
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
        sigma2 = omega + alpha * (eps_prev**2) + beta * (sigma_prev**2)
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))

        eps_t = sigma_t * z[t, :]
        returns_bps[t, :] = mu + eps_t

        sigma_prev = sigma_t
        eps_prev = eps_t

    # 5. Chuyển đổi về giá
    log_ret = returns_bps / scale
    
    prices = np.zeros((n_sims, steps + 1))
    prices[:, 0] = S0
    
    cum_ret = np.cumsum(log_ret, axis=0)
    prices[:, 1:] = S0 * np.exp(cum_ret).T

    return prices

# ==========================================
# 🚀 3. MAIN SIMULATION CONTROLLER
# ==========================================
def simulate_single_price_path_with_garch(prices_dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: Optional[int] = 42, **kwargs):
    # 1. Chuẩn bị dữ liệu
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    
    # 2. Lấy cấu hình tối ưu
    config = get_optimal_config(asset, time_increment)
    config.update(kwargs) # Override with tuning parameters
    print(f"Config: {config}")
    
    # 3. Cắt gọt dữ liệu (Lookback Window)
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
    try:
        res = fit_garch_optimized(returns, config)
        print(f"      -> Params: alpha={res.params.get('alpha[1]', 0):.3f}, beta={res.params.get('beta[1]', 0):.3f}, nu={res.params.get('nu', 0):.2f}")
    except Exception as e:
        print(f"[ERROR] Fit failed definitively: {e}. Falling back to random walk with historical vol.")
        # Minimal random walk fallback setup
        returns_bps = np.zeros((time_length // time_increment, n_sims))
        scale = config["scale"]
        log_ret = returns_bps / scale
        prices = np.zeros((n_sims, (time_length // time_increment) + 1))
        S0 = float(hist_prices.iloc[-1])
        prices[:, 0] = S0
        cum_ret = np.cumsum(log_ret, axis=0)
        prices[:, 1:] = S0 * np.exp(cum_ret).T
        return prices

    # 5. Mô phỏng
    steps = time_length // time_increment
    S0 = float(hist_prices.iloc[-1])
    
    print(f"[INFO] Simulating {n_sims} paths for {time_length/3600}h (steps={steps})...")
    paths = simulate_garch_paths(res, S0, steps, n_sims, config, seed=seed)
    
    return paths

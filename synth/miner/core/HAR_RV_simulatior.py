### pip install statsmodels

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from typing import Tuple, Optional, Dict
from tqdm import trange
from synth.miner.core.garch_simulator import fit_garch_studentt, compute_log_returns

def compute_realized_variance(prices: pd.Series, freq: str = '1D') -> pd.Series:
    """
    Tính Realized Variance (RV) hàng ngày từ chuỗi giá tần suất cao.

    RV cho một khoảng thời gian được tính là tổng bình phương lợi suất log trong khoảng đó.
    prices: chuỗi giá được sắp xếp theo thời gian (ví dụ: tần suất 5 phút)
    freq: tần suất gộp lại (ví dụ: '1D' cho hàng ngày)
    """
    # 1. Tính lợi suất log tần suất cao
    returns = np.log(prices).diff().dropna()
    
    # 2. Bình phương lợi suất
    squared_returns = returns.pow(2)
    
    # 3. Gộp lại theo tần suất mong muốn (ví dụ: '1D') và tính tổng
    # Ví dụ: group by ngày và sum các returns^2 để có RV hàng ngày
    rv = squared_returns.resample(freq).sum().dropna()
    
    # RV thường được biểu diễn dưới dạng phần trăm (bps^2) hoặc nhân với 100
    # Ở đây chúng ta giữ ở dạng decimal để dễ dàng tích hợp.
    return rv

def compute_har_components(daily_rv: pd.Series, periods: Tuple[int, int] = (7, 30)) -> pd.DataFrame:
    """
    Tạo các thành phần trễ (lagged components) cho mô hình HAR-RV.
    periods: (Weekly days, Monthly days)
    """
    df = pd.DataFrame(daily_rv.rename('RV_Day'))
    
    # RV_Week (Trung bình 7 ngày trước)
    df['RV_Week'] = df['RV_Day'].rolling(window=periods[0]).mean().shift(1)
    
    # RV_Month (Trung bình 30 ngày trước)
    df['RV_Month'] = df['RV_Day'].rolling(window=periods[1]).mean().shift(1)
    
    # RV_Day (Ngày hôm trước)
    df['RV_Day_Lag'] = df['RV_Day'].shift(1)
    
    # Loại bỏ các hàng có giá trị NaN do rolling và shift
    df = df.dropna()
    
    # RV_Day là biến phụ thuộc, các biến còn lại là biến độc lập
    return df

def fit_and_forecast_har(daily_rv: pd.Series, forecast_steps: int = 1) -> float:
    """
    Ước tính mô hình HAR-RV và dự báo Realized Variance (RV) cho ngày tiếp theo.
    forecast_steps: số ngày để dự báo (mặc định 1 ngày)
    Returns: Dự báo RV cho ngày tiếp theo (dạng decimal)
    """
    har_data = compute_har_components(daily_rv)
    
    if len(har_data) < 40:
        print("[WARNING] Dữ liệu lịch sử quá ngắn để ước tính HAR ổn định. Sử dụng RV trung bình.")
        return daily_rv.mean()

    # Biến phụ thuộc (t+1)
    y = har_data['RV_Day']
    # Biến độc lập (t)
    X = har_data[['RV_Day_Lag', 'RV_Week', 'RV_Month']]
    # Thêm hằng số vào mô hình
    X = sm.add_constant(X)
    
    # Ước tính mô hình Hồi quy Tuyến tính (OLS)
    har_model = sm.OLS(y, X).fit()
    # print(har_model.summary()) # Có thể kiểm tra P-value và R^2
    
    # Dữ liệu mới nhất để dự báo
    last_data = har_data.iloc[-1]
    
    # X_forecast: Dữ liệu hiện tại để dự báo RV cho ngày tiếp theo
    X_forecast = pd.Series({
        'const': 1.0,
        'RV_Day_Lag': last_data['RV_Day'],  # RV hôm nay
        'RV_Week': har_data['RV_Day'].rolling(window=7).mean().iloc[-1],
        'RV_Month': har_data['RV_Day'].rolling(window=30).mean().iloc[-1]
    })
    
    # Dự báo RV ngày tiếp theo
    rv_forecast = har_model.predict(X_forecast).iloc[0]
    
    # Biến động không được là số âm
    return max(0, float(rv_forecast))

# Giả định fit_garch_studentt quay về GARCH(1,1) Student's t:
# def fit_garch_studentt(returns: pd.Series, mean: str = "Constant"):
#     model = arch_model(returns * 10000.0, mean=mean, vol="GARCH", p=1, q=1, dist="StudentsT")
#     res = model.fit(disp="off")
#     return res

def simulate_garch_paths_har_normalized(fitted_res,
                                         S0: float,
                                         steps: int,
                                         n_sims: int,
                                         har_rv_forecast: float, # <-- Đầu vào mới
                                         seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    
    # [Giữ nguyên logic khởi tạo và trích xuất tham số mu, omega, alpha, beta, nu]
    if seed is not None:
        np.random.seed(seed)
        
    params = fitted_res.params
    mu_candidates = [k for k in params.index if k.lower() in ("mu", "const", "mean")]
    mu = float(params[mu_candidates[0]]) if mu_candidates else 0.0

    omega = float(params.get("omega", 1e-6))
    alpha = float(params.get("alpha[1]", 0.05))
    beta = float(params.get("beta[1]", 0.9))
    nu = float(params.get("nu", 8.0))

    sigma0 = float(fitted_res.conditional_volatility.iloc[-1]) # Conditional volatility cuối cùng
    
    if nu <= 2:
        scale_std = 1.0
    else:
        scale_std = np.sqrt(nu / (nu - 2.0))
        
    returns_bps = np.zeros((steps, n_sims), dtype=np.float64)
    sigma2_sim = np.zeros((steps, n_sims), dtype=np.float64) # Lưu trữ phương sai GARCH thô

    sigma_prev = np.full(n_sims, sigma0 * 10000.0, dtype=np.float64) # Scale BPS
    eps_prev = fitted_res.resid.iloc[-1] # Sốc lợi suất cuối cùng (BPS scale)
    eps_prev = np.full(n_sims, eps_prev, dtype=np.float64) 
    
    z = student_t.rvs(df=nu, size=(steps, n_sims))
    if scale_std != 1.0:
        z = z / scale_std

    # --- Bước Mô phỏng GARCH THÔ (Chưa chuẩn hóa) ---
    for t in trange(steps, desc="Simulating GARCH raw paths"):
        sigma2 = omega + alpha * (eps_prev ** 2) + beta * (sigma_prev ** 2)
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))
        
        eps_t = sigma_t * z[t, :]
        returns_bps[t, :] = mu + eps_t
        sigma2_sim[t, :] = sigma2 # Lưu phương sai thô

        sigma_prev = sigma_t
        eps_prev = eps_t
        
    # --- Bước Chuẩn hóa Phương sai (Variance Targeting) ---
    
    # 1. Tính tổng phương sai GARCH THÔ dự báo (cho N_sims đường)
    # Tổng phương sai GARCH dự kiến (ở BPS scale):
    total_garch_variance_sims = sigma2_sim.sum(axis=0) 
    
    # 2. Tính tỷ lệ chuẩn hóa (Normalization Ratio)
    # har_rv_forecast cần phải được chuyển thành BPS^2 scale (nhân 10000^2)
    har_rv_forecast_bps2 = har_rv_forecast * (10000.0 ** 2)
    
    # Tỷ lệ: (Biến động HAR Mục tiêu) / (Biến động GARCH Dự báo)
    # Sử dụng np.mean(total_garch_variance_sims) để lấy giá trị trung bình từ 1000 sims
    mean_garch_variance = np.mean(total_garch_variance_sims)
    
    if mean_garch_variance <= 1e-12:
        # Tránh chia cho 0, dùng tỷ lệ mặc định 1
        scale_factor = 1.0
    else:
        # Tỷ lệ chuẩn hóa (áp dụng cho toàn bộ đường mô phỏng)
        scale_factor = har_rv_forecast_bps2 / mean_garch_variance

    # 3. Áp dụng chuẩn hóa cho Lợi suất (returns_bps)
    # Lợi suất ~ sigma * z. Lợi suất^2 ~ sigma^2 * z^2. 
    # Nếu scale sigma^2 lên X lần, thì lợi suất cần scale lên sqrt(X) lần.
    scale_multiplier = np.sqrt(scale_factor)
    
    # Lợi suất đã chuẩn hóa
    logret = (returns_bps / 10000.0) * scale_multiplier

    # Build price paths (Giữ nguyên logic)
    prices = np.zeros((n_sims, steps + 1), dtype=np.float64)
    prices[:, 0] = S0
    for t in range(steps):
        prices[:, t + 1] = prices[:, t] * np.exp(logret[t, :])

    meta = {
        # ... (params)
        "HAR_RV_Target_bps2": har_rv_forecast_bps2,
        "GARCH_RV_Mean_bps2": mean_garch_variance,
        "Scaling_Multiplier": scale_multiplier
    }
    return prices, meta

def simulate_single_price_path_with_har_garch(prices_dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: Optional[int] = 42, **kwargs):
    
    # Sắp xếp và xử lý dữ liệu lịch sử
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    hist_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    
    # 1. Dự báo Biến động Cấp độ Lớn bằng HAR-RV
    daily_rv = compute_realized_variance(hist_prices, freq='1D')
    har_rv_forecast = fit_and_forecast_har(daily_rv) # RV dự báo cho 24h tới (dạng decimal)
    
    if har_rv_forecast == 0:
        print("[ERROR] Dự báo HAR-RV bằng 0. Sử dụng GARCH thuần.")
        har_rv_forecast = daily_rv.mean()
    
    # 2. Ước tính mô hình GARCH(1,1) trên lợi suất 5 phút
    returns = compute_log_returns(hist_prices)
    res = fit_garch_studentt(returns, mean="Constant") # Sử dụng GARCH(1,1) Student's t cơ bản
    
    # 3. Mô phỏng và Chuẩn hóa
    steps = time_length // time_increment
    S0 = float(hist_prices.iloc[-1])
    
    prices, meta = simulate_garch_paths_har_normalized(
        res, S0=S0, steps=steps, n_sims=n_sims, 
        har_rv_forecast=har_rv_forecast, # <-- Truyền HAR RV vào
        seed=seed
    )
    
    print(f"\n[INFO] HAR-GARCH Simulation Summary:")
    print(f"  HAR RV Target (24h, decimal): {har_rv_forecast:.6f}")
    print(f"  Scaling Multiplier applied: {meta['Scaling_Multiplier']:.4f}")
    print(f"  Example Path (Sim 0, end price): {prices[0, -1]:.2f}")
    
    return prices
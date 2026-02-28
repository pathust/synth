import numpy as np
import pandas as pd
from typing import Optional
from arch import arch_model
from arch.univariate import StudentsT

# ==========================================
# 📅 1. WEEKLY SEASONALITY ENGINE (7-DAY MAP)
# ==========================================
def compute_weekly_profile(prices: pd.Series, time_increment: int = 300) -> dict:
    """
    Tạo Profile biến động theo TUẦN (Weekday + Hour + Minute).
    Khắc phục lỗi dự báo vào cuối tuần (Weekend).
    """
    # 1. Tính Absolute Returns
    log_ret = np.log(prices).diff().dropna()
    abs_ret = log_ret.abs()
    
    # 2. Tạo DataFrame chứa thông tin thời gian
    df = pd.DataFrame({'abs_ret': abs_ret.values}, index=abs_ret.index)
    df['weekday'] = df.index.weekday # 0=Monday, 6=Sunday
    df['time_key'] = df.index.time   # HH:MM:SS
    
    # 3. Group by (Weekday, Time)
    # Đây là bước quan trọng nhất: Tách biệt Thứ 2 với Thứ 7
    profile_raw = df.groupby(['weekday', 'time_key'])['abs_ret'].mean()
    
    # 4. Convert sang Dictionary để lookup siêu tốc
    # Key: (weekday, time_object) -> Value: Multiplier
    profile_map = profile_raw.to_dict()
    
    # 5. Xử lý Missing Values & Weekends
    # Nếu dữ liệu lịch sử không có T7/CN (do sàn nghỉ), profile sẽ thiếu key.
    # Ta cần fill giá trị mặc định cực nhỏ cho các khung giờ thiếu này.
    
    # Tính trung bình toàn bộ để chuẩn hóa
    global_mean = df['abs_ret'].mean()
    
    # Chuẩn hóa về hệ số (Multiplier)
    # Value = 1.0 nghĩa là biến động bình thường
    # Value = 3.0 nghĩa là biến động gấp 3 (Giờ mở cửa)
    # Value = 0.1 nghĩa là biến động tắt (Đêm/Cuối tuần)
    final_profile = {k: (v / global_mean) for k, v in profile_map.items()}
    
    return final_profile, global_mean

def get_future_weekly_factors(start_time, steps, time_increment, profile_map):
    """
    Tra cứu hệ số biến động cho tương lai dựa trên Lịch Tuần.
    """
    factors = []
    current_t = start_time
    
    # Hệ số mặc định cho giờ "Chết" (những giờ không có trong lịch sử)
    # Ví dụ: Thứ 7 với chứng khoán Mỹ. Ta set cực nhỏ để dải dự báo co lại tối đa.
    DEAD_ZONE_MULTIPLIER = 0.05 
    
    for _ in range(steps):
        current_t += pd.Timedelta(seconds=time_increment)
        
        # Tạo key lookup
        wd = current_t.weekday()
        tm = current_t.time()
        key = (wd, tm)
        
        # Lookup
        val = profile_map.get(key, DEAD_ZONE_MULTIPLIER)
        factors.append(val)
        
    return np.array(factors)


def simulate_weekly_seasonal_optimized(prices_dict, asset, time_increment, time_length, n_sims=1000, seed: Optional[int] = 42):
    if seed is not None:
        np.random.seed(seed)
    # 1. Prepare Data
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    S0 = float(full_prices.iloc[-1])
    steps = time_length // time_increment
    
    # 2. Compute Weekly Profile
    # Lưu ý: Cần lookback ĐỦ DÀI để bắt được quy luật tuần
    # Stock/XAU cần ít nhất 6-8 tuần dữ liệu (40-60 ngày)
    profile_map, global_mean = compute_weekly_profile(full_prices, time_increment)
    
    # 3. Deseasonalize (Khử mùa để fit GARCH)
    lookback_days = 60 # Lấy 60 ngày gần nhất để fit
    points = int(lookback_days * 86400 // time_increment)
    hist_prices = full_prices.tail(points)
    
    raw_returns = np.log(hist_prices.ffill()).diff().dropna()
    
    # Tạo mảng multiplier tương ứng với lịch sử để chia
    hist_factors = []
    for t in raw_returns.index:
        key = (t.weekday(), t.time())
        # Nếu lịch sử có nến đó mà profile ko có (rất hiếm), dùng 1.0
        hist_factors.append(profile_map.get(key, 1.0))
    
    hist_factors = np.array(hist_factors)
    # Tránh chia cho 0
    hist_factors = np.maximum(hist_factors, 0.1)
    
    # Returns đã lọc sạch yếu tố giờ giấc -> Chỉ còn biến động ngẫu nhiên thuần túy
    filtered_returns = (raw_returns / hist_factors) * 10000.0

    # 4. Fit GJR-GARCH (Chuẩn cho Stock/XAU)
    # Với Stock: o=1 (Bắt đòn bẩy). Với XAU: o=0 (Thường ko cần, nhưng o=1 cũng ko hại)
    use_o = 1 if asset != "XAU" else 0
    model = arch_model(filtered_returns, vol='Garch', p=1, o=use_o, q=1, dist='studentst', mean='Constant')
    
    try:
        res = model.fit(disp="off", show_warning=False)
    except:
        res = model.fit(disp="off", rescale=True)

    # 5. Get Future Factors (Tương lai 24h)
    last_timestamp = full_prices.index[-1]
    future_factors = get_future_weekly_factors(last_timestamp, steps, time_increment, profile_map)

    # 6. Init Simulation
    last_vol = res.conditional_volatility.iloc[-1]
    last_resid = res.resid.iloc[-1]
    
    mu = res.params.get("mu", 0.0)
    omega = res.params.get("omega", 0.01)
    alpha = res.params.get("alpha[1]", 0.05)
    beta = res.params.get("beta[1]", 0.90)
    gamma = res.params.get("gamma[1]", 0.0) if use_o else 0.0
    
    # Tối ưu Nu cho XAU/Stock
    target_nu = 8.0 if asset != "XAU" else 10.0
    nu = max(res.params.get("nu", 8.0), target_nu)

    sigma_prev = np.full(n_sims, last_vol)
    eps_prev = np.full(n_sims, last_resid)

    # Noise Generation
    dist_sampler = StudentsT()
    z = dist_sampler.ppf(np.random.random((steps, n_sims)), [nu])

    returns_bps = np.zeros((steps, n_sims))

    # 7. Seasonal Simulation Loop
    for t in range(steps):
        # A. GARCH Variance (Core Noise)
        indicator = (eps_prev < 0).astype(float)
        term_shock = (alpha + gamma * indicator) * (eps_prev**2)
        sigma2 = omega + term_shock + beta * (sigma_prev**2)
        sigma_t_garch = np.sqrt(np.maximum(sigma2, 1e-12))
        
        # B. Apply Seasonality
        seasonal_factor = future_factors[t]
        
        # Volatility thực = GARCH Vol * Seasonal Factor
        # Nếu là đêm/cuối tuần, seasonal_factor ~ 0.05 -> Vol tắt ngóm -> CRPS cực tốt
        sigma_t_final = sigma_t_garch * seasonal_factor
        
        # C. Return Calculation
        eps_t = sigma_t_final * z[t, :]
        
        # Drift Scaling: Giờ nghỉ thì Drift cũng phải nghỉ
        scaled_mu = mu * seasonal_factor 
        
        returns_bps[t, :] = scaled_mu + eps_t
        
        # D. Update State (Feed filtered residual back to GARCH)
        # Để GARCH học tiếp, phải trả về eps đã khử mùa
        sigma_prev = sigma_t_garch
        eps_prev = eps_t / max(seasonal_factor, 0.1)

    # 8. Output
    log_ret = returns_bps / 10000.0
    paths = S0 * np.exp(np.cumsum(log_ret, axis=0)).T
    
    return np.column_stack([np.full(n_sims, S0), paths])
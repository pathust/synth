import numpy as np
import pandas as pd
from typing import Optional
from arch import arch_model
from arch.univariate import StudentsT, Normal

# ==========================================
# 📊 1. SEASONALITY ENGINE
# ==========================================
def compute_intraday_profile(prices: pd.Series, time_increment: int = 300) -> pd.Series:
    """
    Tạo Profile biến động theo từng khung giờ trong ngày (Intraday Seasonality).
    Trả về: Series index là 'HH:MM', value là hệ số multiplier.
    """
    # 1. Tính Absolute Returns (Độ lớn biến động)
    # Log returns
    log_ret = np.log(prices).diff().dropna()
    abs_ret = log_ret.abs()
    
    # 2. Group theo giờ trong ngày (Time of Day)
    # Tạo cột Time dạng chuỗi "HH:MM" hoặc integer minute of day
    # time_increment = 300 (5 phút)
    
    # Chuyển index về múi giờ gốc hoặc UTC (tùy data input), lấy giờ:phút
    # Giả sử index là datetime
    time_index = abs_ret.index.time
    
    # Tạo DataFrame để group
    df_vol = pd.DataFrame({'abs_ret': abs_ret.values}, index=time_index)
    
    # Tính trung bình biến động cho từng khung giờ (Mean Absolute Deviation)
    # Dùng rolling mean nhẹ để làm mượt profile
    profile = df_vol.groupby(df_vol.index).mean()
    
    # Làm mượt profile (Smoothing) để tránh gai nhọn do nhiễu
    # Rolling 3 slot (15 phút)
    # Cần convert sang series để rolling, sau đó chuẩn hóa
    s_profile = pd.Series(profile['abs_ret'], index=profile.index)
    
    # Xử lý missing values hoặc giờ hiếm giao dịch (gán giá trị nhỏ nhất)
    s_profile = s_profile.replace(0, np.nan).ffill().bfill()
    
    # Chuẩn hóa: Chia cho trung bình tổng thể để ra hệ số (Multiplier)
    # Trung bình hệ số phải bằng 1
    s_profile = s_profile / s_profile.mean()
    
    return s_profile

def get_future_seasonality(start_time, steps, time_increment, profile):
    """
    Lấy chuỗi hệ số mùa vụ cho tương lai 24h.
    """
    future_multipliers = []
    current_t = start_time
    
    # Convert profile keys to simple string/time objects for lookup
    # Cách nhanh: Map time -> value dict
    profile_dict = profile.to_dict()
    
    # Fallback value
    avg_mult = 1.0
    
    for _ in range(steps):
        current_t += pd.Timedelta(seconds=time_increment)
        t_key = current_t.time()
        
        # Lookup
        val = profile_dict.get(t_key, avg_mult)
        future_multipliers.append(val)
        
    return np.array(future_multipliers)

# ==========================================
# 🚀 2. SEASONAL GARCH SIMULATION
# ==========================================
def simulate_seasonal_stock(prices_dict, asset, time_increment, time_length, n_sims=1000, seed: Optional[int] = 42):
    if seed is not None:
        np.random.seed(seed)
    """
    Dự báo Stock với xử lý Mùa vụ trong ngày (Intraday Seasonality).
    Khắc phục lỗi dự báo sai khi thị trường đóng/mở cửa.
    """
    # 1. Prepare Data
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    S0 = float(full_prices.iloc[-1])
    steps = time_length // time_increment
    
    # 2. Compute Seasonality Profile (Dùng toàn bộ dữ liệu hiện có để học thói quen giờ giấc)
    # Stock cần lookback dài để profile giờ giấc chính xác
    # Nhưng fit GARCH chỉ cần lookback ngắn
    seasonality_profile = compute_intraday_profile(full_prices, time_increment)
    
    # 3. Deseasonalize Data (Khử mùa)
    # Lấy dữ liệu lookback để fit
    lookback_days = 20 # 20 ngày gần nhất
    points = int(lookback_days * 86400 // time_increment)
    hist_prices = full_prices.tail(points)
    
    # Tính returns
    raw_returns = np.log(hist_prices.ffill()).diff().dropna()
    
    # Lấy hệ số mùa vụ tương ứng với lịch sử
    hist_times = raw_returns.index.time
    hist_multipliers = np.array([seasonality_profile.get(t, 1.0) for t in hist_times])
    
    # Chia returns cho multiplier -> Returns đã khử (chỉ còn GARCH noise)
    # Tránh chia cho 0 hoặc số quá nhỏ
    hist_multipliers = np.maximum(hist_multipliers, 0.1) 
    filtered_returns = (raw_returns / hist_multipliers) * 10000.0 # Scale lên

    # 4. Fit GJR-GARCH trên dữ liệu đã khử
    # Lúc này GARCH sẽ học được bản chất biến động gốc mà không bị nhiễu bởi giờ đóng/mở cửa
    model = arch_model(filtered_returns, vol='Garch', p=1, o=1, q=1, dist='studentst', mean='Constant')
    
    try:
        res = model.fit(disp="off", show_warning=False)
    except:
        res = model.fit(disp="off", rescale=True)

    # 5. Get Future Seasonality Multipliers
    # Đây là chìa khóa: Lấy hệ số cho 24h TIẾP THEO
    last_timestamp = full_prices.index[-1]
    future_seasonal_factors = get_future_seasonality(last_timestamp, steps, time_increment, seasonality_profile)

    # 6. Simulation Init
    last_vol = res.conditional_volatility.iloc[-1]
    last_resid = res.resid.iloc[-1]
    
    # Params
    mu = res.params.get("mu", 0.0)
    omega = res.params.get("omega", 0.01)
    alpha = res.params.get("alpha[1]", 0.05)
    beta = res.params.get("beta[1]", 0.90)
    gamma = res.params.get("gamma[1]", 0.0)
    nu = max(res.params.get("nu", 8.0), 5.0) # TSLA/NVDA chấp nhận 5.0

    sigma_prev = np.full(n_sims, last_vol)
    eps_prev = np.full(n_sims, last_resid)

    # Noise
    dist_sampler = StudentsT()
    z = dist_sampler.ppf(np.random.random((steps, n_sims)), [nu])

    returns_bps = np.zeros((steps, n_sims))

    # 7. Seasonal Simulation Loop
    for t in range(steps):
        # A. GARCH Volatility (Của thành phần ngẫu nhiên)
        indicator = (eps_prev < 0).astype(float)
        term_shock = (alpha + gamma * indicator) * (eps_prev**2)
        sigma2 = omega + term_shock + beta * (sigma_prev**2)
        sigma_t_garch = np.sqrt(np.maximum(sigma2, 1e-12))
        
        # B. Apply Seasonality (Reseasonalize)
        # Vol thực tế = GARCH Vol * Seasonal Multiplier của tương lai
        # Nếu t rơi vào giờ mở cửa (9:30), multiplier sẽ > 2.0 -> Vol bùng nổ
        # Nếu t rơi vào giờ nghỉ, multiplier < 0.5 -> Vol tắt ngóm
        seasonal_factor = future_seasonal_factors[t]
        sigma_t_final = sigma_t_garch * seasonal_factor
        
        # C. Calculate Return
        eps_t = sigma_t_final * z[t, :]
        returns_bps[t, :] = (mu * seasonal_factor) + eps_t # Scale cả drift theo mùa (tùy chọn)
        
        # D. Update State (Lưu ý: State cho GARCH phải là đã khử mùa)
        # Để update GARCH step sau, ta cần đưa eps về dạng "filtered"
        # eps_filtered = eps_t / seasonal_factor
        sigma_prev = sigma_t_garch # Gữ lại sigma gốc của GARCH
        eps_prev = eps_t / max(seasonal_factor, 0.1) 

    # 8. Output
    log_ret = returns_bps / 10000.0
    paths = S0 * np.exp(np.cumsum(log_ret, axis=0)).T
    
    return np.column_stack([np.full(n_sims, S0), paths])

# import numpy as np
# import pandas as pd
# from arch import arch_model
# from arch.univariate import StudentsT, Normal
# from typing import Optional

# # ==========================================
# # 📊 1. STOCK INDICATORS (RSI & VOLATILITY)
# # ==========================================
# def calculate_stock_indicators(prices: pd.Series):
#     """
#     Tính RSI và trạng thái biến động để điều chỉnh tham số.
#     """
#     # 1. Tính RSI (14 period)
#     delta = prices.diff()
#     gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
#     loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
#     # 2. Tính Volatility Ratio (Hiện tại vs Trung bình 50 nến)
#     # Để xem vol đang nở ra hay co vào
#     returns = np.log(prices).diff()
#     curr_vol = returns.tail(10).std()
#     avg_vol = returns.tail(50).std()
#     vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
    
#     return rsi, vol_ratio

# # ==========================================
# # ⚙️ 2. ASSET-CLASS CONFIGURATION
# # ==========================================
# def get_stock_specialist_config(asset: str, rsi: float, vol_ratio: float, time_increment: int):
#     config = {
#         "scale": 10000.0,
#         "mean_model": "Constant",
#         "garch_o": 1, # Luôn dùng GJR-GARCH cho stock
#     }
    
#     is_high_freq = time_increment <= 60
    
#     # --- NHÓM 1: TSLAX, NVDAX (Biến động mạnh - High Beta) ---
#     if asset in ["TSLAX", "NVDAX"]:
#         # Lookback ngắn để bắt nhịp nhanh
#         config["lookback_days"] = 10 if is_high_freq else 20
        
#         # Chấp nhận đuôi dày (Fat tails) vì 2 mã này hay chạy điên
#         config["min_nu"] = 5.0 
        
#         # Logic RSI cho Volatility Multiplier
#         # RSI 40-60: Thị trường lưỡng lự/đi ngang -> Giảm Vol để ăn điểm Sharpness
#         if 40 <= rsi <= 60:
#             config["vol_multiplier"] = 0.90 # Giảm 10%
#             config["mean_model"] = "Zero"   # Bỏ drift
#         # RSI Quá mua (>70) hoặc Quá bán (<30): Volatility thường tăng mạnh -> Giữ nguyên hoặc tăng nhẹ
#         elif rsi > 70 or rsi < 30:
#             config["vol_multiplier"] = 1.05 
#         else:
#             config["vol_multiplier"] = 1.0

#     # --- NHÓM 2: AAPLX, GOOGLX (Ổn định - Low Beta) ---
#     elif asset in ["AAPLX", "GOOGLX"]:
#         # Lookback dài hơn để ổn định
#         config["lookback_days"] = 14 if is_high_freq else 45
        
#         # Ép đuôi mỏng hơn (gần Normal) để dải dự báo gọn
#         config["min_nu"] = 4.0 
        
#         # Nếu Vol đang thấp hơn trung bình (vol_ratio < 1) -> Đang Squeeze
#         if vol_ratio < 0.9:
#             config["vol_multiplier"] = 0.85 # Ép mạnh tay
#             config["mean_model"] = "Zero"
#             config["min_nu"] = 12.0 # Ép phân phối gần Normal
#         elif 40 <= rsi <= 60:
#             config["vol_multiplier"] = 0.95
#         else:
#             config["vol_multiplier"] = 1.0
            
#     return config

# # ==========================================
# # 🚀 3. SIMULATION FUNCTION
# # ==========================================
# def simulate_stock_optimized(prices_dict, asset, time_increment, time_length, n_sims=1000, seed: Optional[int] = 42):
#     if seed is not None:
#         np.random.seed(seed)
#     # 1. Prepare Data
#     timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
#     full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
#     S0 = float(full_prices.iloc[-1])
#     steps = time_length // time_increment
    
#     # 2. Get Indicators & Config
#     rsi, vol_ratio = calculate_stock_indicators(full_prices)
#     cfg = get_stock_specialist_config(asset, rsi, vol_ratio, time_increment)

#     print(f"RSi: {rsi}, Vol Ratio: {vol_ratio}, Config: {cfg}")
    
#     # 3. Fit GJR-GARCH
#     points = int(cfg["lookback_days"] * 86400 // time_increment)
#     hist_prices = full_prices.tail(points)
#     returns = np.log(hist_prices.ffill()).diff().dropna() * cfg["scale"]

#     # Fit với phân phối StudentT
#     model = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='studentst', mean=cfg["mean_model"])
#     try:
#         res = model.fit(disp="off", show_warning=True, options={"maxiter": 500})
#     except Exception as e:
#         print(f"[WARNING] Fit GJR-GARCH failed: {e}; retrying with rescaling...")
#         res = model.fit(disp="off", rescale=True, options={"maxiter": 500})

#     print(f"Params: {res.params}")

#     # 4. Params & Constraints
#     mu = res.params.get("mu", 0.0)
#     omega = res.params.get("omega", 0.01)
#     alpha = res.params.get("alpha[1]", 0.05)
#     beta = res.params.get("beta[1]", 0.90)
#     gamma = res.params.get("gamma[1]", 0.0)
    
#     # Ép Nu theo config (TSLA/NVDA thấp, AAPL/GOOGL cao)
#     nu = max(res.params.get("nu", 8.0), cfg["min_nu"])

#     # 5. Init Simulation (Volatility Targeting)
#     last_vol = res.conditional_volatility.iloc[-1] * cfg["vol_multiplier"]
#     last_resid = res.resid.iloc[-1]
    
#     sigma_prev = np.full(n_sims, last_vol)
#     eps_prev = np.full(n_sims, last_resid)

#     # 6. Generate Noise (StudentT PPF)
#     dist_sampler = StudentsT()
#     z = dist_sampler.ppf(np.random.random((steps, n_sims)), [nu])

#     returns_bps = np.zeros((steps, n_sims))

#     # 7. Loop (GJR-GARCH)
#     for t in range(steps):
#         # GJR Variance Term
#         indicator = (eps_prev < 0).astype(float)
#         term_shock = (alpha + gamma * indicator) * (eps_prev**2)
        
#         sigma2 = omega + term_shock + beta * (sigma_prev**2)
#         sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))
        
#         eps_t = sigma_t * z[t, :]
#         returns_bps[t, :] = mu + eps_t 
        
#         sigma_prev = sigma_t
#         eps_prev = eps_t

#     # 8. Output
#     log_ret = returns_bps / cfg["scale"]
#     paths = S0 * np.exp(np.cumsum(log_ret, axis=0)).T
    
#     return np.column_stack([np.full(n_sims, S0), paths])

# def simulate_fhs_bootstrap(prices_dict, asset, time_increment, time_length, n_sims=1000, seed: Optional[int] = 42):
#     if seed is not None:
#         np.random.seed(seed)
#     """
#     FHS (Filtered Historical Simulation) chuyên sâu.
#     Sử dụng Bootstrapping từ phần dư chuẩn hóa (Standardized Residuals).
#     """
#     # 1. Prepare Data
#     timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
#     full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
#     S0 = float(full_prices.iloc[-1])
#     steps = time_length // time_increment
#     is_high_freq = time_increment <= 60

#     # 2. Config Lookback (Quan trọng cho FHS)
#     # Cần đủ dữ liệu để kho mẫu (Pool) đa dạng, nhưng không quá xa để bị lỗi thời
#     if asset in ["TSLAX", "NVDAX"]:
#         lookback_days = 30 # High volatility cần mẫu gần
#         pool_size = 1000   # Chỉ lấy 1000 residuals gần nhất để bootstrap
#     elif asset == "SPYX":
#         lookback_days = 90
#         pool_size = 2000   # SPYX cần mẫu lớn để mượt
#     else: # BTC, ETH, AAPL...
#         lookback_days = 60
#         pool_size = 1500

#     # 3. Fit GARCH để lọc Volatility (Filter)
#     points = int(lookback_days * 86400 // time_increment)
#     hist_prices = full_prices.tail(points)
#     returns = np.log(hist_prices.ffill()).diff().dropna() * 10000.0

#     # Dùng SkewStudent hoặc StudentT để GARCH lọc vol tốt nhất có thể
#     # Lưu ý: Ta KHÔNG dùng distribution này để dự báo, chỉ để tính residuals
#     model = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='skewstudent', mean='Constant')
    
#     try:
#         res = model.fit(disp="off", show_warning=True, options={"maxiter": 500})
#     except Exception as e:
#         print(f"[WARNING] Fit GJR-GARCH failed: {e}; retrying with rescaling...")
#         # Fallback nếu skewstudent lỗi
#         model = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='studentst', mean='Constant')
#         res = model.fit(disp="off", rescale=True)

#     # 4. TẠO KHO MẪU (BOOTSTRAP POOL)
#     # Lấy phần dư chuẩn hóa: z = returns / sigma
#     std_resid = res.std_resid.dropna()
    
#     # Chỉ lấy N mẫu gần nhất (Recent Bias)
#     if len(std_resid) > pool_size:
#         bootstrap_pool = std_resid.tail(pool_size).values
#     else:
#         bootstrap_pool = std_resid.values

#     # --- Xử lý Outliers cực đoan (Tùy chọn) ---
#     # Nếu trong quá khứ có cú sập flash crash do lỗi sàn, ta nên loại bỏ
#     # Giữ lại 99.8% dữ liệu giữa
#     lower = np.percentile(bootstrap_pool, 0.1)
#     upper = np.percentile(bootstrap_pool, 99.9)
#     bootstrap_pool = bootstrap_pool[(bootstrap_pool >= lower) & (bootstrap_pool <= upper)]

#     # 5. Initialization
#     last_vol = res.conditional_volatility.iloc[-1]
#     last_resid = res.resid.iloc[-1]
    
#     # Params
#     mu = res.params.get("mu", 0.0)
#     omega = res.params.get("omega", 0.01)
#     alpha = res.params.get("alpha[1]", 0.05)
#     beta = res.params.get("beta[1]", 0.90)
#     gamma = res.params.get("gamma[1]", 0.0)

#     # Chia đôi simulation để dùng Antithetic
#     n_half = n_sims // 2
    
#     sigma_prev = np.full(n_half, last_vol)
#     eps_prev_raw = np.full(n_half, last_resid)
    
#     # Arrays chứa kết quả
#     paths_pos = np.zeros((steps, n_half))
#     paths_neg = np.zeros((steps, n_half))
    
#     # State cho nhánh đối xứng
#     sigma_prev_neg = sigma_prev.copy()
#     eps_prev_neg = eps_prev_raw.copy()

#     # 6. SIMULATION LOOP (BOOTSTRAP)
#     for t in range(steps):
#         # A. BỐC MẪU (SAMPLING)
#         # Thay vì dùng StudentT.rvs(), ta bốc ngẫu nhiên từ quá khứ
#         # Đây chính là lúc ta bắt được nu=2.5 một cách tự nhiên nhất
#         z_random = np.random.choice(bootstrap_pool, size=n_half, replace=True)
        
#         # --- NHÁNH 1: Gốc ---
#         # GJR-GARCH Variance
#         ind_pos = (eps_prev_raw < 0).astype(float)
#         sig2_pos = omega + (alpha + gamma * ind_pos) * (eps_prev_raw**2) + beta * (sigma_prev**2)
#         sig_t_pos = np.sqrt(np.maximum(sig2_pos, 1e-12))
        
#         eps_t_pos = sig_t_pos * z_random
#         paths_pos[t, :] = mu + eps_t_pos # Return BPS
        
#         # --- NHÁNH 2: Antithetic (Đối xứng) ---
#         # Dùng -z_random. 
#         # Nếu z_random là một cú sập (đuôi trái), thì -z_random là cú tăng (đuôi phải)
#         # Điều này giúp trung hòa sai số mẫu, giữ Mean ổn định
#         z_anti = -z_random 
        
#         ind_neg = (eps_prev_neg < 0).astype(float)
#         sig2_neg = omega + (alpha + gamma * ind_neg) * (eps_prev_neg**2) + beta * (sigma_prev_neg**2)
#         sig_t_neg = np.sqrt(np.maximum(sig2_neg, 1e-12))
        
#         eps_t_neg = sig_t_neg * z_anti
#         paths_neg[t, :] = mu + eps_t_neg

#         # Update States
#         sigma_prev = sig_t_pos
#         eps_prev_raw = eps_t_pos
        
#         sigma_prev_neg = sig_t_neg
#         eps_prev_neg = eps_t_neg

#     # 7. Combine & Output
#     # Ghép 2 nhánh
#     all_returns_bps = np.hstack([paths_pos, paths_neg])
    
#     # Shuffle để trộn đều (quan trọng)
#     np.random.shuffle(all_returns_bps.T)
    
#     # Tính giá
#     log_ret = all_returns_bps / 10000.0
#     cum_ret = np.cumsum(log_ret, axis=0)
#     paths = S0 * np.exp(cum_ret).T
    
#     return np.column_stack([np.full(n_sims, S0), paths])
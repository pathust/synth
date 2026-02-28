import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import Normal, StudentsT
from typing import Optional

def simulate_spyx_sniper(prices_dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: Optional[int] = 42):
    """
    SPYX Sniper Pro: Tận dụng Regime để chỉnh cả Volatility và Mean Reversion Speed.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Prepare Data
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    S0 = float(full_prices.iloc[-1])
    steps = time_length // time_increment
    
    # 2. Analyze Regime (Logic BBW)
    window = 20
    sma = full_prices.rolling(window).mean()
    std = full_prices.rolling(window).std()
    
    # BB Width hiện tại và trung bình
    bb_width = (sma + 2*std - (sma - 2*std)) / sma
    current_bbw = bb_width.iloc[-1]
    avg_bbw = bb_width.rolling(60).mean().iloc[-1]
    
    # 3. Dynamic Configuration based on Regime (DÙNG TRIỆT ĐỂ Ở ĐÂY)
    if current_bbw < avg_bbw:
        # --- REGIME: SQUEEZE (Thị trường nén/ngủ) ---
        # Hành động: Ép chặt mọi thứ
        vol_multiplier = 0.70  # Giảm 30% Vol (Rất mạnh)
        
        # Mean Reversion cực mạnh: Kéo giá về trung bình ngay lập tức
        # Task 1h (60s) -> 0.2, Task 24h -> 0.05
        theta = 0.2 if time_increment <= 60 else 0.05 
        
        # GARCH Mean: Ép về 0 tuyệt đối
        garch_mean_type = 'Zero' 
    else:
        # --- REGIME: NORMAL/EXPANSION (Thị trường chạy) ---
        # Hành động: Nới lỏng
        vol_multiplier = 0.85  # Giảm 15% Vol (Vẫn giảm vì SPYX luôn ổn định)
        
        # Mean Reversion yếu hơn: Cho phép giá trôi đi một chút
        theta = 0.05 if time_increment <= 60 else 0.01
        
        # GARCH Mean: Cho phép hằng số nhỏ (nếu có trend)
        garch_mean_type = 'Constant'

    # 4. Fit Model (GJR-GARCH Normal)
    lookback_points = 5000 
    hist_prices = full_prices.tail(lookback_points)
    returns = np.log(hist_prices.ffill()).diff().dropna() * 10000.0

    model = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='Normal', mean=garch_mean_type)
    
    try:
        res = model.fit(disp="off", show_warning=False)
    except:
        # Fallback
        model = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal', mean='Zero')
        res = model.fit(disp="off", rescale=True)

    # 5. Extract Params
    omega = res.params.get("omega", 0.01)
    alpha = res.params.get("alpha[1]", 0.05)
    beta = res.params.get("beta[1]", 0.90)
    gamma = res.params.get("gamma[1]", 0.0)
    mu = res.params.get("mu", 0.0) # Lấy mu nếu dùng mean='Constant'

    # 6. Initialize
    last_vol = res.conditional_volatility.iloc[-1] * vol_multiplier
    last_resid = res.resid.iloc[-1]
    
    sigma_prev = np.full(n_sims, last_vol)
    eps_prev = np.full(n_sims, last_resid)
    current_sim_prices = np.full(n_sims, S0)

    # 7. Generate Noise (Standard Normal)
    z = np.random.standard_normal((steps, n_sims))

    # Lấy đường trung bình dài hạn để làm nam châm hút giá
    long_term_mean = full_prices.rolling(50).mean().iloc[-1]
    
    returns_bps = np.zeros((steps, n_sims))

    # 8. Simulation Loop
    for t in range(steps):
        # --- GJR Variance ---
        indicator = (eps_prev < 0).astype(float)
        term_shock = (alpha + gamma * indicator) * (eps_prev**2)
        sigma2 = omega + term_shock + beta * (sigma_prev**2)
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))
        
        # --- Dynamic Drift (Tận dụng Theta từ Regime) ---
        # Lực hồi quy về trung bình
        reversion_drift = -theta * (current_sim_prices - long_term_mean) / current_sim_prices * 10000.0
        
        # Nếu model là Constant Mean, ta cộng thêm mu vào (Drift = Reversion + Trend)
        # Nếu model là Zero Mean, mu = 0
        total_drift = reversion_drift + mu
        
        # --- Total Return ---
        eps_t = sigma_t * z[t, :]
        total_ret_bps = total_drift + eps_t
        
        returns_bps[t, :] = total_ret_bps
        
        # Update State
        sigma_prev = sigma_t
        eps_prev = eps_t
        current_sim_prices = current_sim_prices * np.exp(total_ret_bps / 10000.0)

    # 9. Output
    log_ret = returns_bps / 10000.0
    paths = S0 * np.exp(np.cumsum(log_ret, axis=0)).T
    
    return np.column_stack([np.full(n_sims, S0), paths])


def simulate_spyx_robust(prices_dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: Optional[int] = 42):
    """
    SPYX Balanced: Cân bằng giữa độ an toàn (StudentT) và độ hẹp (Nu cao).
    Sử dụng GJR-GARCH để bắt đòn bẩy giảm giá.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Prepare Data
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    S0 = float(full_prices.iloc[-1])
    steps = time_length // time_increment
    
    is_high_freq = time_increment <= 60

    # 2. Analyze Regime (Logic BBW đơn giản)
    # Xác định xem thị trường đang nén hay đang chạy
    window = 20
    sma = full_prices.rolling(window).mean()
    std = full_prices.rolling(window).std()
    bb_width = (sma + 2*std - (sma - 2*std)) / sma
    
    current_bbw = bb_width.iloc[-1]
    avg_bbw = bb_width.rolling(60).mean().iloc[-1]
    
    is_squeeze = current_bbw < (avg_bbw * 0.95) # Nén nhẹ
    
    # 3. Config (Conservative Optimization)
    # Lookback: 60 ngày cho task 24h, 14 ngày cho task 1h
    lookback_days = 14 if is_high_freq else 60
    
    if is_squeeze:
        # Khi nén: Tăng Nu lên cao (đuôi mỏng), giảm nhẹ vol
        target_nu = 15.0 
        vol_multiplier = 0.90 
        mean_type = 'Constant' # Vẫn giữ drift để bắt trend nhẹ
    else:
        # Bình thường: Nu chuẩn cho stock, giữ nguyên vol
        target_nu = 8.0
        vol_multiplier = 1.0
        mean_type = 'Constant'

    # 4. Fit GJR-GARCH (Bắt buộc cho Stock)
    points = int(lookback_days * 86400 // time_increment)
    hist_prices = full_prices.tail(points)
    returns = np.log(hist_prices.ffill()).diff().dropna() * 10000.0

    # GJR-GARCH (o=1)
    model = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='studentst', mean=mean_type)
    
    try:
        res = model.fit(disp="off", show_warning=False)
    except:
        # Fallback
        model = arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='studentst', mean='Constant')
        res = model.fit(disp="off", rescale=True)

    # 5. Extract Params
    mu = res.params.get("mu", 0.0)
    omega = res.params.get("omega", 0.01)
    alpha = res.params.get("alpha[1]", 0.05)
    beta = res.params.get("beta[1]", 0.90)
    gamma = res.params.get("gamma[1]", 0.0) # Leverage param
    
    # Quan trọng: Ép Nu theo Regime đã định nghĩa
    # Không để model tự quyết định nu quá thấp (đuôi quá dày)
    estimated_nu = res.params.get("nu", 8.0)
    nu = max(estimated_nu, target_nu)

    # 6. Init Simulation
    last_vol = res.conditional_volatility.iloc[-1] * vol_multiplier
    last_resid = res.resid.iloc[-1]
    
    sigma_prev = np.full(n_sims, last_vol)
    eps_prev = np.full(n_sims, last_resid)

    # 7. Generate Noise (StudentT)
    # Dùng PPF để đảm bảo phân phối chính xác
    dist_sampler = StudentsT()
    z = dist_sampler.ppf(np.random.random((steps, n_sims)), [nu])

    returns_bps = np.zeros((steps, n_sims))

    # 8. Loop (GJR logic chuẩn)
    for t in range(steps):
        # GJR Variance Term
        indicator = (eps_prev < 0).astype(float)
        term_shock = (alpha + gamma * indicator) * (eps_prev**2)
        
        sigma2 = omega + term_shock + beta * (sigma_prev**2)
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))
        
        # Return Calculation
        eps_t = sigma_t * z[t, :]
        returns_bps[t, :] = mu + eps_t # Chỉ dùng Mu của GARCH, bỏ Drift tự chế
        
        # Update
        sigma_prev = sigma_t
        eps_prev = eps_t

    # 9. Output
    log_ret = returns_bps / 10000.0
    paths = S0 * np.exp(np.cumsum(log_ret, axis=0)).T
    
    return np.column_stack([np.full(n_sims, S0), paths])

def simulate_fhs_antithetic(prices_dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: Optional[int] = 42):
    """
    Phương pháp: Filtered Historical Simulation (FHS) + Antithetic Variates.
    Tối ưu cho CRPS bằng cách dùng phân phối thực nghiệm và triệt tiêu sai số mô phỏng.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Prepare Data
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    S0 = float(full_prices.iloc[-1])
    steps = time_length // time_increment
    is_high_freq = time_increment <= 60
    
    # 2. Config & Regime (Giữ lại cái này để scale Volatility)
    # Lấy thông tin cơ bản để scale vol, nhưng KHÔNG dùng tham số phân phối (nu, dist) nữa
    window = 20
    sma = full_prices.rolling(window).mean()
    bb_width = (sma + 2*full_prices.rolling(window).std() - (sma - 2*full_prices.rolling(window).std())) / sma
    current_bbw = bb_width.iloc[-1]
    avg_bbw = bb_width.rolling(60).mean().iloc[-1]
    is_squeeze = current_bbw < avg_bbw

    # Cấu hình Lookback dài để có kho mẫu lớn
    if asset == "SPYX":
        lookback_days = 90 # Khoảng 2000 nến 1h, đủ để bootstrap
        vol_multiplier = 0.85 if is_squeeze else 1.0
    elif asset in ["BTC", "ETH"]:
        lookback_days = 30
        vol_multiplier = 0.95 if is_squeeze else 1.0
    elif asset == "SOL":
        lookback_days = 20
        vol_multiplier = 0.98 if is_squeeze else 1.05
    else: # XAU
        lookback_days = 60
        vol_multiplier = 0.90 if is_squeeze else 1.0

    # 3. Fit GARCH để lọc nhiễu (Filter)
    # Mục đích: Chuẩn hóa returns để lấy phần dư (Standardized Residuals)
    points = int(lookback_days * 86400 // time_increment)
    hist_prices = full_prices.tail(points)
    returns = np.log(hist_prices.ffill()).diff().dropna() * 10000.0 # Basis points

    # Dùng GJR-GARCH StudentT để lọc vol tốt nhất
    # Lưu ý: Ta chỉ dùng model để lấy residuals, không dùng nó để sinh số ngẫu nhiên
    model = arch_model(returns, vol='Garch', p=1, o=1, q=1, dist='studentst', mean='Constant')
    try:
        res = model.fit(disp="off", show_warning=False)
    except:
        res = model.fit(disp="off", rescale=True)

    # 4. Lấy kho mẫu (Standardized Residuals Pool)
    # Đây là "DNA" thực tế của thị trường
    std_resid_pool = res.std_resid.dropna().values
    
    # Lọc bớt outliers quá cực đoan (nếu muốn an toàn cho CRPS)
    # Giữ lại 99.5% dữ liệu, bỏ 0.25% đuôi trái phải nếu quá xa (ví dụ flash crash do lỗi data)
    lower = np.percentile(std_resid_pool, 0.25)
    upper = np.percentile(std_resid_pool, 99.75)
    std_resid_pool = std_resid_pool[(std_resid_pool >= lower) & (std_resid_pool <= upper)]

    # 5. Simulation Init
    last_vol = res.conditional_volatility.iloc[-1] * vol_multiplier
    
    # Antithetic Sampling: Chia đôi số lượng paths
    # Ta sẽ sinh n_sims/2 path gốc, và tạo n_sims/2 path đối xứng
    n_half = n_sims // 2
    
    sigma_prev = np.full(n_half, last_vol)
    eps_prev_raw = np.full(n_half, res.resid.iloc[-1]) # Lấy resid thực tế cuối cùng
    
    # Params
    omega = res.params.get("omega", 0.01)
    alpha = res.params.get("alpha[1]", 0.05)
    beta = res.params.get("beta[1]", 0.90)
    gamma = res.params.get("gamma[1]", 0.0)
    mu = res.params.get("mu", 0.0)

    # Output arrays cho 2 nửa
    returns_bps_pos = np.zeros((steps, n_half))
    returns_bps_neg = np.zeros((steps, n_half))
    
    # Trạng thái cho nửa âm (Negative paths)
    sigma_prev_neg = sigma_prev.copy()
    eps_prev_neg = eps_prev_raw.copy()

    # 6. Bootstrap Loop (FHS)
    for t in range(steps):
        # --- A. Lấy mẫu từ quá khứ (Bootstrap) ---
        # Chọn ngẫu nhiên từ kho mẫu residuals thực tế
        z_boot = np.random.choice(std_resid_pool, size=n_half, replace=True)
        
        # --- B. Tính toán nhánh Positive (+Z) ---
        # GJR Variance
        indicator_pos = (eps_prev_raw < 0).astype(float)
        term_shock_pos = (alpha + gamma * indicator_pos) * (eps_prev_raw**2)
        sigma2_pos = omega + term_shock_pos + beta * (sigma_prev**2)
        sigma_t_pos = np.sqrt(np.maximum(sigma2_pos, 1e-12))
        
        # New Return
        eps_t_pos = sigma_t_pos * z_boot
        returns_bps_pos[t, :] = mu + eps_t_pos
        
        # --- C. Tính toán nhánh Negative (-Z) -> Antithetic ---
        # Dùng -z_boot để cân bằng sai số lấy mẫu
        z_anti = -z_boot 
        
        # GJR Variance (Nhánh âm có lịch sử riêng)
        indicator_neg = (eps_prev_neg < 0).astype(float)
        term_shock_neg = (alpha + gamma * indicator_neg) * (eps_prev_neg**2)
        sigma2_neg = omega + term_shock_neg + beta * (sigma_prev_neg**2)
        sigma_t_neg = np.sqrt(np.maximum(sigma2_neg, 1e-12))
        
        eps_t_neg = sigma_t_neg * z_anti
        returns_bps_neg[t, :] = mu + eps_t_neg

        # Update States
        sigma_prev = sigma_t_pos
        eps_prev_raw = eps_t_pos
        
        sigma_prev_neg = sigma_t_neg
        eps_prev_neg = eps_t_neg

    # 7. Combine & Output
    # Ghép 2 nửa lại thành 1000 paths
    all_returns_bps = np.hstack([returns_bps_pos, returns_bps_neg])
    
    # Shuffle để không bị quy luật nửa đầu nửa sau
    # (Quan trọng nếu hệ thống chấm điểm lấy mẫu 1 phần)
    np.random.shuffle(all_returns_bps.T) 

    log_ret = all_returns_bps / 10000.0
    paths = S0 * np.exp(np.cumsum(log_ret, axis=0)).T
    
    return np.column_stack([np.full(n_sims, S0), paths])
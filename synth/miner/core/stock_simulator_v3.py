import numpy as np
import pandas as pd
from typing import Optional
from arch import arch_model
from arch.univariate import StudentsT, SkewStudent

def compute_weekly_profile(prices: pd.Series, time_increment: int = 300) -> dict:
    log_ret = np.log(prices).diff().dropna()
    abs_ret = log_ret.abs()
    df = pd.DataFrame({'abs_ret': abs_ret.values}, index=abs_ret.index)
    df['weekday'] = df.index.weekday
    df['time_key'] = df.index.time
    profile_raw = df.groupby(['weekday', 'time_key'])['abs_ret'].mean()
    profile_map = profile_raw.to_dict()
    global_mean = df['abs_ret'].mean()
    final_profile = {k: (v / global_mean) for k, v in profile_map.items()}
    return final_profile, global_mean

def get_future_weekly_factors(start_time, steps, time_increment, profile_map):
    factors = []
    current_t = start_time
    DEAD_ZONE_MULTIPLIER = 0.0001
    for _ in range(steps):
        current_t += pd.Timedelta(seconds=time_increment)
        wd = current_t.weekday()
        tm = current_t.time()
        key = (wd, tm)
        val = profile_map.get(key, DEAD_ZONE_MULTIPLIER)
        factors.append(val)
    return np.array(factors)

def simulate_weekly_garch_v4(
    prices_dict,
    asset,
    time_increment,
    time_length,
    n_sims=1000,
    seed: Optional[int] = 42,
    **kwargs,
):
    if seed is not None:
        np.random.seed(seed)

    lookback_days = int(kwargs.pop("lookback_days", 90))
    # Absorb any other tuning keys so grid search can pass a fixed superset.
    kwargs.clear()

    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    S0 = float(full_prices.iloc[-1])
    steps = time_length // time_increment
    
    profile_map, global_mean = compute_weekly_profile(full_prices, time_increment)
    points = int(lookback_days * 86400 // time_increment)
    hist_prices = full_prices.tail(points)
    
    raw_returns = np.log(hist_prices.ffill()).diff().dropna()
    
    hist_factors = []
    for t in raw_returns.index:
        key = (t.weekday(), t.time())
        hist_factors.append(profile_map.get(key, 1.0))
    hist_factors = np.maximum(np.array(hist_factors), 0.05)
    filtered_returns = (raw_returns / hist_factors) * 10000.0

    # Advanced GARCH V4 characteristics (SkewStudent, o=1, p=1, q=1)
    use_o = 1 if asset != "XAU" else 0
    model = arch_model(filtered_returns, vol='Garch', p=1, o=use_o, q=1, dist='skewstudent', mean='Constant')
    
    try:
        res = model.fit(disp="off", show_warning=False)
    except:
        model = arch_model(filtered_returns, vol='Garch', p=1, o=use_o, q=1, dist='studentst', mean='Constant')
        res = model.fit(disp="off", rescale=True)

    future_factors = get_future_weekly_factors(full_prices.index[-1], steps, time_increment, profile_map)

    last_vol = res.conditional_volatility.iloc[-1]
    last_resid = res.resid.iloc[-1]
    
    mu = res.params.get("mu", 0.0)
    omega = res.params.get("omega", 0.01)
    alpha = res.params.get("alpha[1]", 0.05)
    beta = res.params.get("beta[1]", 0.90)
    gamma = res.params.get("gamma[1]", 0.0) if use_o else 0.0
    
    nu = max(res.params.get("nu", 6.0), 3.0)

    sigma_prev = np.full(n_sims, last_vol)
    eps_prev = np.full(n_sims, last_resid)

    # Use basic Student's T to save complexity
    dist_sampler = StudentsT()
    z = dist_sampler.ppf(np.random.random((steps, n_sims)), [nu])
    returns_bps = np.zeros((steps, n_sims))

    for t in range(steps):
        indicator = (eps_prev < 0).astype(float)
        term_shock = (alpha + gamma * indicator) * (eps_prev**2)
        sigma2 = omega + term_shock + beta * (sigma_prev**2)
        sigma_t_garch = np.sqrt(np.maximum(sigma2, 1e-12))
        
        seasonal_factor = future_factors[t]
        sigma_t_final = sigma_t_garch * seasonal_factor
        
        eps_t = sigma_t_final * z[t, :]
        scaled_mu = mu * seasonal_factor 
        returns_bps[t, :] = scaled_mu + eps_t
        
        sigma_prev = sigma_t_garch
        eps_prev = eps_t / max(seasonal_factor, 0.05)

    log_ret = returns_bps / 10000.0
    paths = S0 * np.exp(np.cumsum(log_ret, axis=0)).T
    return np.column_stack([np.full(n_sims, S0), paths])

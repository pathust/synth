import numpy as np
import pandas as pd
from typing import Optional

def _estimate_jump_params(returns: np.ndarray, dt: float) -> dict:
    mu = np.mean(returns) / dt
    total_var = np.var(returns) / dt
    kurt = float(pd.Series(returns).kurtosis())

    if kurt < 1.0:
        jump_intensity = 0.5
        jump_var = total_var * 0.05
    else:
        jump_intensity = min(kurt / 10.0, 5.0)
        jump_var = total_var * min(kurt / 20.0, 0.3)

    diffusion_var = max(total_var - jump_intensity * jump_var, total_var * 0.5)
    sigma = np.sqrt(diffusion_var)

    jump_std = np.sqrt(jump_var) if jump_var > 0 else 0.001
    jump_mean = -0.5 * jump_var

    return {
        "mu": mu,
        "sigma": sigma,
        "jump_intensity": jump_intensity,
        "jump_mean": jump_mean,
        "jump_std": jump_std,
    }

def simulate_jump_diffusion(
    prices_dict: dict,
    asset: str,
    time_increment: int,
    time_length: int,
    n_sims: int,
    seed: Optional[int] = 42,
    **params
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    lookback_days = params.get("lookback_days", 30)

    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit="s")
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()

    points_per_day = 86400 // time_increment
    needed = int(lookback_days * points_per_day)
    hist_prices = full_prices.tail(needed) if len(full_prices) > needed else full_prices

    log_returns = np.log(hist_prices).diff().dropna().values
    dt = time_increment / 86400.0

    jp = _estimate_jump_params(log_returns, dt)

    mu = jp["mu"]
    sigma = jp["sigma"]
    lam = params.get("jump_intensity_override") or jp["jump_intensity"]
    jm = params.get("jump_mean_override") or jp["jump_mean"]
    js = params.get("jump_std_override") or jp["jump_std"]
    jv_scale = float(params.get("jump_vol_scale", 1.0))
    js = float(js) * jv_scale

    steps = time_length // time_increment
    S0 = float(hist_prices.iloc[-1])

    k = np.exp(jm + 0.5 * js**2) - 1.0

    drift = (mu - lam * k - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    dW = np.random.randn(steps, n_sims)
    N_jumps = np.random.poisson(lam * dt, size=(steps, n_sims))
    J_total = np.zeros((steps, n_sims))
    
    for t in range(steps):
        for s in range(n_sims):
            if N_jumps[t, s] > 0:
                J_total[t, s] = np.sum(np.random.normal(jm, js, size=N_jumps[t, s]))

    log_ret = drift + diffusion * dW + J_total

    prices = np.zeros((n_sims, steps + 1))
    prices[:, 0] = S0
    cum_ret = np.cumsum(log_ret, axis=0)
    prices[:, 1:] = S0 * np.exp(cum_ret).T

    return prices

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import StudentsT as ArchStudentsT
from typing import Optional

def simulate_gjr_garch(
    prices_dict: dict,
    asset: str,
    time_increment: int,
    time_length: int,
    n_sims: int,
    seed: Optional[int] = 42,
    **params
) -> np.ndarray:
    
    if asset.upper() == "ETH" and time_increment <= 60:
        if "lookback_days" not in params:
            params["lookback_days"] = 7
        if "mean_model" not in params:
            params["mean_model"] = "Zero"

    lookback_days = params.get("lookback_days", 45)
    mean_model = params.get("mean_model", "Constant")
    scale = params.get("scale", 10000.0)
    vol_multiplier = params.get("vol_multiplier", 1.0)

    if seed is not None:
        np.random.seed(seed)

    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit="s")
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()

    points_per_day = 86400 // time_increment
    needed = int(lookback_days * points_per_day)
    hist_prices = full_prices.tail(needed) if len(full_prices) > needed else full_prices

    returns = np.log(hist_prices.ffill()).diff().dropna() * scale

    model = arch_model(
        returns,
        vol="Garch",
        p=1, o=1, q=1,
        dist="studentst",
        mean=mean_model,
    )
    try:
        res = model.fit(disp="off", show_warning=False)
    except Exception:
        res = model.fit(disp="off", show_warning=False, rescale=True, options={"maxiter": 500})

    mu = float(res.params.get("mu", res.params.get("Const", 0.0)))
    omega = float(res.params.get("omega", 0.01))
    alpha = float(res.params.get("alpha[1]", 0.05))
    beta = float(res.params.get("beta[1]", 0.90))
    gamma = float(res.params.get("gamma[1]", 0.0))
    nu = max(float(res.params.get("nu", 8.0)), 3.0)

    steps = time_length // time_increment
    S0 = float(hist_prices.iloc[-1])

    last_vol = float(res.conditional_volatility.iloc[-1]) * vol_multiplier
    last_resid = float(res.resid.iloc[-1])

    sigma_prev = np.full(n_sims, last_vol)
    eps_prev = np.full(n_sims, last_resid)

    dist_sampler = ArchStudentsT()
    z = dist_sampler.ppf(np.random.random((steps, n_sims)), [nu])
    returns_bps = np.zeros((steps, n_sims))

    for t in range(steps):
        indicator = (eps_prev < 0).astype(float)
        term_shock = (alpha + gamma * indicator) * (eps_prev**2)
        sigma2 = omega + term_shock + beta * (sigma_prev**2)
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))

        eps_t = sigma_t * z[t, :]
        returns_bps[t, :] = mu + eps_t

        sigma_prev = sigma_t
        eps_prev = eps_t

    log_ret = returns_bps / scale
    cum_ret = np.cumsum(log_ret, axis=0)
    prices = np.zeros((n_sims, steps + 1))
    prices[:, 0] = S0
    prices[:, 1:] = S0 * np.exp(cum_ret).T

    return prices

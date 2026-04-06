import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t as student_t
from typing import Optional

def simulate_egarch(
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
    mean_model = params.get("mean_model", "Zero")
    scale = params.get("scale", 10000.0)
    p = params.get("p", 1)
    q = params.get("q", 1)
    o = params.get("o", 1)

    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit="s")
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()

    points_per_day = 86400 // time_increment
    needed = int(lookback_days * points_per_day)
    hist_prices = full_prices.tail(needed) if len(full_prices) > needed else full_prices

    returns = np.log(hist_prices).diff().dropna()
    scaled_returns = returns * scale

    model = arch_model(
        scaled_returns,
        mean=mean_model,
        vol="EGARCH",
        p=p, o=o, q=q,
        dist="StudentsT",
    )
    try:
        res = model.fit(disp="off", show_warning=False)
    except Exception:
        res = model.fit(disp="off", show_warning=False, options={"maxiter": 500}, rescale=True)

    nu = max(float(res.params.get("nu", 8.0)), 3.0)
    mu = float(res.params.get("mu", res.params.get("Const", 0.0)))

    steps = time_length // time_increment
    S0 = float(hist_prices.iloc[-1])

    scale_std = np.sqrt(nu / (nu - 2.0)) if nu > 2 else 1.0
    z = student_t.rvs(df=nu, size=(steps, n_sims)) / scale_std

    last_vol = float(res.conditional_volatility.iloc[-1])
    omega = float(res.params.get("omega", 0.0))
    alpha_params = [float(res.params.get(f"alpha[{i+1}]", 0.0)) for i in range(p)]
    gamma_params = [float(res.params.get(f"gamma[{i+1}]", 0.0)) for i in range(o)]
    beta_params = [float(res.params.get(f"beta[{i+1}]", 0.0)) for i in range(q)]

    alpha1 = alpha_params[0] if alpha_params else 0.0
    gamma1 = gamma_params[0] if gamma_params else 0.0
    beta1 = min(beta_params[0] if beta_params else 0.9, 0.98)

    ln_sigma2_prev = np.full(n_sims, np.log(last_vol**2))
    z_prev = np.zeros(n_sims)
    returns_bps = np.zeros((steps, n_sims))
    expected_abs_z = np.sqrt(2 / np.pi)

    for t in range(steps):
        ln_sigma2 = (
            omega
            + alpha1 * (np.abs(z_prev) - expected_abs_z)
            + gamma1 * z_prev
            + beta1 * ln_sigma2_prev
        )
        ln_sigma2 = np.clip(ln_sigma2, -15, 12)
        sigma_t = np.sqrt(np.exp(ln_sigma2))

        z_t = np.clip(z[t, :], -5.0, 5.0)
        eps_t = sigma_t * z_t
        returns_bps[t, :] = mu + eps_t

        z_prev = z_t
        ln_sigma2_prev = ln_sigma2

    log_ret = returns_bps / scale
    cum_ret = np.clip(np.cumsum(log_ret, axis=0), -3.0, 3.0)
    prices = np.zeros((n_sims, steps + 1))
    prices[:, 0] = S0
    prices[:, 1:] = S0 * np.exp(cum_ret).T

    return prices

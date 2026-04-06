import numpy as np
import pandas as pd
from typing import Optional

def _fit_ou_params(prices: pd.Series, dt: float) -> dict:
    X = np.log(prices.values)
    dX = np.diff(X)
    X_lag = X[:-1]

    A = np.column_stack([np.ones_like(X_lag), X_lag])
    result = np.linalg.lstsq(A, dX, rcond=None)
    a, b = result[0]

    residuals = dX - (a + b * X_lag)

    theta = max(-b / dt, 0.001)
    mu_log = -a / b if abs(b) > 1e-10 else np.mean(X)
    sigma = np.std(residuals) / np.sqrt(dt)

    return {
        "theta": theta,
        "mu_log": mu_log,
        "sigma": sigma,
    }

def simulate_mean_reversion(
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

    dt = time_increment
    dt_norm = dt / 86400.0
    ou_params = _fit_ou_params(hist_prices, dt_norm)

    theta = ou_params["theta"]
    mu_log = ou_params["mu_log"]
    sigma = ou_params["sigma"]

    if params.get("half_life_override") is not None:
        theta = np.log(2) / params["half_life_override"]

    steps = time_length // time_increment
    S0 = float(hist_prices.iloc[-1])
    X0 = np.log(S0)

    dW = np.random.randn(steps, n_sims) * np.sqrt(dt_norm)

    X = np.zeros((n_sims, steps + 1))
    X[:, 0] = X0

    for t in range(steps):
        drift = theta * (mu_log - X[:, t]) * dt_norm
        diffusion = sigma * dW[t, :]
        X[:, t + 1] = X[:, t] + drift + diffusion

    prices = np.exp(X)
    return prices

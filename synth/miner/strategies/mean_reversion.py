"""
mean_reversion.py — Ornstein-Uhlenbeck (OU) mean-reversion strategy.

Models prices that tend to revert to a long-run mean. The OU process is:
    dX_t = θ(μ - X_t)dt + σ dW_t

Parameters estimated via MLE from historical log-prices.

Best for: XAU, SPYX — lower volatility, mean-reverting behavior.
"""

from typing import Optional
import numpy as np
import pandas as pd

from synth.miner.strategies.base import BaseStrategy


def _fit_ou_params(prices: pd.Series, dt: float) -> dict:
    """
    Estimate OU parameters (theta, mu, sigma) via OLS on discretized OU:
        X_{t+1} - X_t = θ(μ - X_t)·dt + σ·√dt·ε_t
    
    Equivalently linear regression: ΔX = a + b·X + noise
        → θ = -b/dt, μ = -a/b, σ = std(residuals)/√dt
    """
    X = np.log(prices.values)
    dX = np.diff(X)
    X_lag = X[:-1]

    # OLS: dX = a + b*X_lag
    A = np.column_stack([np.ones_like(X_lag), X_lag])
    result = np.linalg.lstsq(A, dX, rcond=None)
    a, b = result[0]

    residuals = dX - (a + b * X_lag)

    # Extract params
    theta = max(-b / dt, 0.001)  # mean-reversion speed (prevent negative)
    mu_log = -a / b if abs(b) > 1e-10 else np.mean(X)
    sigma = np.std(residuals) / np.sqrt(dt)

    return {
        "theta": theta,
        "mu_log": mu_log,
        "sigma": sigma,
    }


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"
    description = (
        "Ornstein-Uhlenbeck mean-reversion process — prices revert toward "
        "a long-run mean, good for range-bound assets"
    )
    supported_asset_types = ["gold"]
    supported_regimes = ["mean_reverting"]
    default_params = {
        "lookback_days": 30,
        "half_life_override": None,  # if set, overrides fitted θ
    }
    param_grid = {
        "lookback_days": [14, 30, 45, 60],
    }

    def simulate(
        self,
        prices_dict: dict,
        asset: str,
        time_increment: int,
        time_length: int,
        n_sims: int,
        seed: Optional[int] = 42,
        **kwargs,
    ) -> np.ndarray:
        params = self.get_default_params()
        params.update(kwargs)

        if seed is not None:
            np.random.seed(seed)

        # ── 1. Prepare Data ──
        timestamps = pd.to_datetime(
            [int(ts) for ts in prices_dict.keys()], unit="s"
        )
        full_prices = pd.Series(
            list(prices_dict.values()), index=timestamps
        ).sort_index()

        points_per_day = 86400 // time_increment
        needed = int(params["lookback_days"] * points_per_day)
        hist_prices = (
            full_prices.tail(needed) if len(full_prices) > needed else full_prices
        )

        # ── 2. Fit OU Parameters ──
        dt = time_increment  # in seconds, but we normalize
        dt_norm = dt / 86400.0  # express in days
        ou_params = _fit_ou_params(hist_prices, dt_norm)

        theta = ou_params["theta"]
        mu_log = ou_params["mu_log"]
        sigma = ou_params["sigma"]

        # Override half-life if requested
        if params["half_life_override"] is not None:
            theta = np.log(2) / params["half_life_override"]

        # ── 3. Simulate OU Process in Log-Space ──
        steps = time_length // time_increment
        S0 = float(hist_prices.iloc[-1])
        X0 = np.log(S0)

        # Pre-generate noise
        dW = np.random.randn(steps, n_sims) * np.sqrt(dt_norm)

        # Euler-Maruyama discretization of OU
        X = np.zeros((n_sims, steps + 1))
        X[:, 0] = X0

        for t in range(steps):
            drift = theta * (mu_log - X[:, t]) * dt_norm
            diffusion = sigma * dW[t, :]
            X[:, t + 1] = X[:, t] + drift + diffusion

        # ── 4. Convert to Prices ──
        prices = np.exp(X)

        return prices


strategy = MeanReversionStrategy()

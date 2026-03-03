"""
jump_diffusion.py — Merton Jump-Diffusion strategy.

Extends GBM with Poisson jumps to capture sudden price discontinuities:
    dS/S = (μ - λk)dt + σ dW + J dN

where J ~ N(jump_mean, jump_std) and N is a Poisson process with
intensity λ.

Parameters estimated from historical kurtosis and skewness of returns.

Best for: BTC, ETH, SOL (sudden crashes/rallies), TSLAX, NVDAX (earnings jumps).
"""

from typing import Optional
import numpy as np
import pandas as pd

from synth.miner.strategies.base import BaseStrategy


def _estimate_jump_params(returns: np.ndarray, dt: float) -> dict:
    """
    Estimate jump parameters from excess kurtosis and variance decomposition.
    
    Method: Use the observed kurtosis vs normal kurtosis (3) to infer
    jump intensity and jump variance.
    """
    mu = np.mean(returns) / dt
    total_var = np.var(returns) / dt
    kurt = float(pd.Series(returns).kurtosis())  # excess kurtosis

    # If excess kurtosis is low, jumps are minor
    if kurt < 1.0:
        jump_intensity = 0.5  # about 1 jump per 2 time units
        jump_var = total_var * 0.05
    else:
        # Higher kurtosis → more frequent or larger jumps
        jump_intensity = min(kurt / 10.0, 5.0)  # cap at 5 jumps per unit time
        jump_var = total_var * min(kurt / 20.0, 0.3)

    diffusion_var = max(total_var - jump_intensity * jump_var, total_var * 0.5)
    sigma = np.sqrt(diffusion_var)

    jump_std = np.sqrt(jump_var) if jump_var > 0 else 0.001
    # Jumps are slightly negative on average for crypto (crash bias)
    jump_mean = -0.5 * jump_var

    return {
        "mu": mu,
        "sigma": sigma,
        "jump_intensity": jump_intensity,
        "jump_mean": jump_mean,
        "jump_std": jump_std,
    }


class JumpDiffusionStrategy(BaseStrategy):
    name = "jump_diffusion"
    description = (
        "Merton Jump-Diffusion — GBM with Poisson jumps for capturing "
        "sudden price shocks (flash crashes, earnings, macro events)"
    )
    supported_assets = ["BTC", "ETH", "SOL", "TSLAX", "NVDAX"]
    supported_frequencies = ["high", "low"]
    default_params = {
        "lookback_days": 30,
        "jump_intensity_override": None,
        "jump_mean_override": None,
        "jump_std_override": None,
    }
    param_grid = {
        "lookback_days": [14, 30, 45],
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

        # Log returns
        log_returns = np.log(hist_prices).diff().dropna().values
        dt = time_increment / 86400.0  # in days

        # ── 2. Estimate Parameters ──
        jp = _estimate_jump_params(log_returns, dt)

        mu = jp["mu"]
        sigma = jp["sigma"]
        lam = params["jump_intensity_override"] or jp["jump_intensity"]
        jm = params["jump_mean_override"] or jp["jump_mean"]
        js = params["jump_std_override"] or jp["jump_std"]

        # ── 3. Simulate ──
        steps = time_length // time_increment
        S0 = float(hist_prices.iloc[-1])

        # Expected jump compensation: k = E[e^J - 1]
        k = np.exp(jm + 0.5 * js**2) - 1.0

        # GBM + Poisson jumps
        # dS/S = (mu - lam*k)*dt + sigma*dW + J*dN
        drift = (mu - lam * k - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        # Pre-generate: Brownian, Poisson jumps, jump sizes
        dW = np.random.randn(steps, n_sims)
        N_jumps = np.random.poisson(lam * dt, size=(steps, n_sims))
        # Sum of J for each step (each jump has size ~ N(jm, js))
        J_total = np.zeros((steps, n_sims))
        for t in range(steps):
            for s in range(n_sims):
                if N_jumps[t, s] > 0:
                    J_total[t, s] = np.sum(
                        np.random.normal(jm, js, size=N_jumps[t, s])
                    )

        # Log-return per step
        log_ret = drift + diffusion * dW + J_total

        # ── 4. Build Prices ──
        prices = np.zeros((n_sims, steps + 1))
        prices[:, 0] = S0
        cum_ret = np.cumsum(log_ret, axis=0)
        prices[:, 1:] = S0 * np.exp(cum_ret).T

        return prices


strategy = JumpDiffusionStrategy()

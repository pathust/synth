"""
egarch.py — EGARCH (Exponential GARCH) strategy.

Models volatility in log-space, naturally handling asymmetric effects
(negative shocks increase volatility more than positive ones). No
positivity constraints needed unlike standard GARCH.

Best for: BTC, ETH, SOL — high volatility with asymmetric reactions.
"""

from typing import Optional
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t as student_t

from synth.miner.strategies.base import BaseStrategy


class EgarchStrategy(BaseStrategy):
    name = "egarch"
    description = (
        "Exponential GARCH — models asymmetric volatility in log-space, "
        "capturing leverage effects where drops cause higher vol than rallies"
    )
    supported_assets = ["BTC", "ETH", "SOL", "XAU"]
    supported_frequencies = ["high", "low"]
    default_params = {
        "p": 1,
        "q": 1,
        "o": 1,
        "lookback_days": 30,
        "mean_model": "Zero",
        "scale": 10000.0,
    }
    param_grid = {
        "lookback_days": [14, 30, 45],
        "mean_model": ["Zero", "Constant"],
        "p": [1, 2],
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

        # Lookback window
        points_per_day = 86400 // time_increment
        needed = int(params["lookback_days"] * points_per_day)
        hist_prices = (
            full_prices.tail(needed) if len(full_prices) > needed else full_prices
        )

        returns = np.log(hist_prices).diff().dropna()
        scaled_returns = returns * params["scale"]

        # ── 2. Fit EGARCH ──
        model = arch_model(
            scaled_returns,
            mean=params["mean_model"],
            vol="EGARCH",
            p=params["p"],
            o=params["o"],
            q=params["q"],
            dist="StudentsT",
        )
        try:
            res = model.fit(disp="off", show_warning=False)
        except Exception:
            # Fallback with rescaling
            res = model.fit(
                disp="off", show_warning=False, options={"maxiter": 500},
                rescale=True,
            )

        # ── 3. Extract Parameters ──
        nu = max(float(res.params.get("nu", 8.0)), 3.0)
        mu = float(res.params.get("mu", res.params.get("Const", 0.0)))

        # ── 4. Simulate using arch forecast simulations ──
        steps = time_length // time_increment
        S0 = float(hist_prices.iloc[-1])

        # Use arch's built-in simulation from the fitted model
        sim = res.model.simulate(
            res.params,
            nobs=steps,
            initial_value=scaled_returns.iloc[-1],
            initial_variance=res.conditional_volatility.iloc[-1] ** 2,
        )

        # Generate multiple paths manually using Student-t innovation
        scale_std = np.sqrt(nu / (nu - 2.0)) if nu > 2 else 1.0
        z = student_t.rvs(df=nu, size=(steps, n_sims)) / scale_std

        # Reconstruct vol path from EGARCH equation
        last_vol = float(res.conditional_volatility.iloc[-1])

        # Use the fitted omega, alpha, gamma, beta for EGARCH log-vol recursion
        omega = float(res.params.get("omega", 0.0))
        alpha_params = [
            float(res.params.get(f"alpha[{i+1}]", 0.0))
            for i in range(params["p"])
        ]
        gamma_params = [
            float(res.params.get(f"gamma[{i+1}]", 0.0))
            for i in range(params["o"])
        ]
        beta_params = [
            float(res.params.get(f"beta[{i+1}]", 0.0))
            for i in range(params["q"])
        ]

        # Simplified EGARCH(1,1,1) simulation
        alpha1 = alpha_params[0] if alpha_params else 0.0
        gamma1 = gamma_params[0] if gamma_params else 0.0
        beta1 = beta_params[0] if beta_params else 0.9

        ln_sigma2_prev = np.full(n_sims, np.log(last_vol**2))
        z_prev = np.zeros(n_sims)

        returns_bps = np.zeros((steps, n_sims))

        for t in range(steps):
            # EGARCH: ln(σ²_t) = ω + α(|z_{t-1}| - E|z|) + γ*z_{t-1} + β*ln(σ²_{t-1})
            expected_abs_z = np.sqrt(2 / np.pi)  # E[|z|] for standard normal
            ln_sigma2 = (
                omega
                + alpha1 * (np.abs(z_prev) - expected_abs_z)
                + gamma1 * z_prev
                + beta1 * ln_sigma2_prev
            )
            sigma_t = np.sqrt(np.exp(np.clip(ln_sigma2, -20, 20)))

            eps_t = sigma_t * z[t, :]
            returns_bps[t, :] = mu + eps_t

            z_prev = z[t, :]
            ln_sigma2_prev = ln_sigma2

        # ── 5. Build Prices ──
        log_ret = returns_bps / params["scale"]
        cum_ret = np.cumsum(log_ret, axis=0)
        prices = np.zeros((n_sims, steps + 1))
        prices[:, 0] = S0
        prices[:, 1:] = S0 * np.exp(cum_ret).T

        return prices


strategy = EgarchStrategy()

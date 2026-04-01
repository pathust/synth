"""
gjr_garch.py — GJR-GARCH (Glosten-Jagannathan-Runkle) strategy.

Threshold GARCH that adds an indicator for negative shocks, allowing
different volatility responses to positive vs negative returns.

Best for: Stocks (NVDAX, TSLAX, AAPLX, GOOGLX, SPYX) and XAU.
"""

from typing import Optional
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import StudentsT as ArchStudentsT

from synth.miner.strategies.base import BaseStrategy


class GjrGarchStrategy(BaseStrategy):
    name = "gjr_garch"
    description = (
        "GJR-GARCH — threshold volatility model with leverage indicator "
        "for asymmetric volatility response to positive/negative shocks"
    )
    supported_assets = []  # all assets
    supported_frequencies = ["high", "low"]
    default_params = {
        "lookback_days": 45,
        "mean_model": "Constant",
        "scale": 10000.0,
        "vol_multiplier": 1.0,
    }
    param_grid = {
        "lookback_days": [14, 30, 45, 60],
        "mean_model": ["Zero", "Constant"],
        "vol_multiplier": [0.85, 0.95, 1.0, 1.05],
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

        # ETH high-frequency (1m): rút lookback để phản ứng nhanh với flash dump/pump.
        # Cho phép kwargs tiếp tục ghi đè nếu caller truyền tường minh.
        if asset.upper() == "ETH" and time_increment <= 60:
            params["lookback_days"] = 7
            params["mean_model"] = "Zero"
            if "lookback_days" in kwargs:
                params["lookback_days"] = kwargs["lookback_days"]
            if "mean_model" in kwargs:
                params["mean_model"] = kwargs["mean_model"]

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

        returns = np.log(hist_prices.ffill()).diff().dropna() * params["scale"]

        # ── 2. Fit GJR-GARCH(1,1,1) ──
        model = arch_model(
            returns,
            vol="Garch",
            p=1,
            o=1,  # GJR leverage term
            q=1,
            dist="studentst",
            mean=params["mean_model"],
        )
        try:
            res = model.fit(disp="off", show_warning=False)
        except Exception:
            res = model.fit(
                disp="off", show_warning=False, rescale=True,
                options={"maxiter": 500},
            )

        # ── 3. Extract Parameters ──
        mu = float(res.params.get("mu", res.params.get("Const", 0.0)))
        omega = float(res.params.get("omega", 0.01))
        alpha = float(res.params.get("alpha[1]", 0.05))
        beta = float(res.params.get("beta[1]", 0.90))
        gamma = float(res.params.get("gamma[1]", 0.0))
        nu = max(float(res.params.get("nu", 8.0)), 3.0)

        # ── 4. Initialize Simulation State ──
        steps = time_length // time_increment
        S0 = float(hist_prices.iloc[-1])

        last_vol = float(res.conditional_volatility.iloc[-1]) * params["vol_multiplier"]
        last_resid = float(res.resid.iloc[-1])

        sigma_prev = np.full(n_sims, last_vol)
        eps_prev = np.full(n_sims, last_resid)

        # Student-t innovations
        dist_sampler = ArchStudentsT()
        z = dist_sampler.ppf(np.random.random((steps, n_sims)), [nu])

        returns_bps = np.zeros((steps, n_sims))

        # ── 5. Simulation Loop ──
        for t in range(steps):
            indicator = (eps_prev < 0).astype(float)
            term_shock = (alpha + gamma * indicator) * (eps_prev**2)
            sigma2 = omega + term_shock + beta * (sigma_prev**2)
            sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))

            eps_t = sigma_t * z[t, :]
            returns_bps[t, :] = mu + eps_t

            sigma_prev = sigma_t
            eps_prev = eps_t

        # ── 6. Build Prices ──
        log_ret = returns_bps / params["scale"]
        cum_ret = np.cumsum(log_ret, axis=0)
        prices = np.zeros((n_sims, steps + 1))
        prices[:, 0] = S0
        prices[:, 1:] = S0 * np.exp(cum_ret).T

        return prices


strategy = GjrGarchStrategy()

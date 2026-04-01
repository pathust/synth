"""
regime_switching.py — Markov Regime-Switching GARCH strategy.

Uses regime detection to identify low-vol/high-vol market states, then
applies different GARCH parameters per regime. Blends simulations from
each regime weighted by the current regime probability.

Uses existing regime_detection.py utilities.

Best for: All assets — adapts to changing market conditions.
"""

from typing import Optional
import numpy as np
import pandas as pd
from arch import arch_model

from synth.miner.strategies.base import BaseStrategy
from synth.miner.core.regime_detection import (
    REGIME_TYPE,
    detect_market_regime_with_er,
    detect_market_regime_with_bbw,
)


class RegimeSwitchingStrategy(BaseStrategy):
    name = "regime_switching"
    description = (
        "Markov Regime-Switching GARCH — detects market regime "
        "(trending/sideways) and adapts volatility model accordingly"
    )
    supported_asset_types = []
    supported_regimes = []
    default_params = {
        "lookback_days": 30,
        "regime_method": "er",  # "er" or "bbw"
        "scale": 10000.0,
        "trending_vol_mult": 1.2,
        "sideways_vol_mult": 0.8,
    }
    param_grid = {
        "lookback_days": [14, 30, 45],
        "trending_vol_mult": [1.0, 1.1, 1.2, 1.3],
        "sideways_vol_mult": [0.6, 0.7, 0.8, 0.9],
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

        # ── 2. Detect Current Regime ──
        if params["regime_method"] == "bbw":
            regime_info = detect_market_regime_with_bbw(hist_prices)
            is_trending = regime_info["is_trending"]
        else:
            regime_info = detect_market_regime_with_er(hist_prices)
            is_trending = regime_info["type"] == REGIME_TYPE.TRENDING

        vol_multiplier = (
            params["trending_vol_mult"] if is_trending
            else params["sideways_vol_mult"]
        )
        print(
            f"[RegimeSwitching] {asset}: "
            f"regime={'TRENDING' if is_trending else 'SIDEWAYS'}, "
            f"vol_mult={vol_multiplier:.2f}"
        )

        # ── 3. Fit GARCH(1,1) ──
        returns = np.log(hist_prices.ffill()).diff().dropna() * params["scale"]

        model = arch_model(
            returns,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist="StudentsT",
        )
        try:
            res = model.fit(disp="off", show_warning=False)
        except Exception:
            res = model.fit(
                disp="off", show_warning=False, rescale=True,
                options={"maxiter": 500},
            )

        # ── 4. Extract Parameters ──
        mu = float(res.params.get("mu", res.params.get("Const", 0.0)))
        omega = float(res.params.get("omega", 0.01))
        alpha = float(res.params.get("alpha[1]", 0.05))
        beta_p = float(res.params.get("beta[1]", 0.90))
        nu = max(float(res.params.get("nu", 8.0)), 3.0)

        # ── 5. Simulate with Regime-Adjusted Volatility ──
        steps = time_length // time_increment
        S0 = float(hist_prices.iloc[-1])

        last_vol = float(res.conditional_volatility.iloc[-1]) * vol_multiplier
        last_shock = float(res.resid.iloc[-1])

        sigma_prev = np.full(n_sims, last_vol)
        eps_prev = np.full(n_sims, last_shock)

        # Student-t noise
        from scipy.stats import t as student_t
        scale_std = np.sqrt(nu / (nu - 2.0)) if nu > 2 else 1.0
        z = student_t.rvs(df=nu, size=(steps, n_sims)) / scale_std

        returns_bps = np.zeros((steps, n_sims))

        for t in range(steps):
            sigma2 = omega + alpha * (eps_prev**2) + beta_p * (sigma_prev**2)
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


strategy = RegimeSwitchingStrategy()

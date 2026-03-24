import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import SkewStudent, StudentsT, Normal
from typing import List, Optional, Dict
import warnings
from arch.utility.exceptions import ConvergenceWarning

from synth.miner.core.garch_simulator_v2_2 import (
    simulate_single_price_path_with_garch as _fallback_garch_v2_2,
)

warnings.filterwarnings("ignore")


# ==========================================
# ⚙️ 1. CONFIGURATION (ENSEMBLE STRATEGY)
# ==========================================
def get_ensemble_configs(asset: str, time_increment: int) -> List[dict]:
    is_crypto = asset.lower() not in ["xau", "gold"]
    is_high_freq = time_increment <= 60

    configs = []

    cfg_short = {
        "name": "Short_Term",
        "mean_model": "Constant",
        "dist": "skewstudent",
        "scale": 10000.0,
        "lookback_days": 5 if (is_crypto and is_high_freq) else 14,
        "weight": 0.4,
    }
    configs.append(cfg_short)

    cfg_long = {
        "name": "Long_Term",
        "mean_model": "Zero",
        "dist": "skewstudent",
        "scale": 10000.0,
        "lookback_days": 30 if (is_crypto and is_high_freq) else 60,
        "weight": 0.6,
    }
    configs.append(cfg_long)

    return configs


# ==========================================
# 🛠️ 2. CORE FUNCTIONS
# ==========================================
def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices.ffill()).diff().dropna()


def fit_garch_robust(returns: pd.Series, config: dict):
    scaled_returns = returns * config["scale"]
    # arch 8.x: fit() không có tham số rescale
    attempts = [
        {"dist": config["dist"], "maxiter": 300},
        {"dist": "studentst", "maxiter": 500},
        {"dist": "normal", "maxiter": 500},
    ]

    last_exc = None
    for i, a in enumerate(attempts, start=1):
        model = arch_model(
            scaled_returns,
            mean=config["mean_model"],
            vol="GARCH",
            p=1, q=1,
            dist=a["dist"],
        )
        try:
            # Silence noisy optimizer convergence warnings; we validate with convergence_flag.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                res = model.fit(disp="off", options={"maxiter": a["maxiter"]})
            conv_flag = int(getattr(res, "convergence_flag", 0))
            if conv_flag == 0:
                return res
            print(f"[WARN] {config['name']} fit attempt {i} not converged (flag={conv_flag}), retrying...")
        except Exception as e:
            last_exc = e

    if last_exc is not None:
        print(f"[WARN] GARCH fit failed for {config['name']}: {last_exc}")
    raise RuntimeError(f"GARCH fit did not converge for {config['name']}")


def simulate_paths_vectorized(fitted_res, S0, steps, n_sims, scale, dist_name):
    params = fitted_res.params
    mu = float(params.get("mu", params.get("Const", 0.0)))
    omega = float(params.get("omega", 0.01))
    alpha = float(params.get("alpha[1]", 0.05))
    beta = float(params.get("beta[1]", 0.90))
    nu = float(params.get("nu", 8.0))
    lambda_skew = float(params.get("lambda", 0.0))

    last_vol = float(fitted_res.conditional_volatility.iloc[-1])
    last_resid = float(fitted_res.resid.iloc[-1])

    sigma_prev = np.full(n_sims, last_vol, dtype=np.float64)
    eps_prev = np.full(n_sims, last_resid, dtype=np.float64)

    total_draws = steps * n_sims
    random_uniform = np.random.random(total_draws)

    dn = dist_name.lower() if hasattr(dist_name, "lower") else str(dist_name).lower()
    if "skew" in dn:
        dist_model = SkewStudent()
        z_flat = dist_model.ppf(random_uniform, [nu, lambda_skew])
    elif "student" in dn:
        dist_model = StudentsT()
        z_flat = dist_model.ppf(random_uniform, [nu])
    else:
        z_flat = np.random.standard_normal(total_draws)
    z = z_flat.reshape(steps, n_sims)

    returns_bps = np.zeros((steps, n_sims))
    for t in range(steps):
        sigma2 = omega + alpha * (eps_prev ** 2) + beta * (sigma_prev ** 2)
        sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))
        eps_t = sigma_t * z[t, :]
        returns_bps[t, :] = mu + eps_t
        sigma_prev = sigma_t
        eps_prev = eps_t

    log_ret = returns_bps / scale
    prices = np.zeros((n_sims, steps + 1))
    prices[:, 0] = S0
    cum_ret = np.cumsum(log_ret, axis=0)
    prices[:, 1:] = S0 * np.exp(cum_ret).T

    return prices


# ==========================================
# 🚀 3. MAIN CONTROLLER
# ==========================================
def simulate_single_price_path_with_garch(
    prices_dict: Dict,
    asset: str,
    time_increment: int,
    time_length: int,
    n_sims: int = 1000,
    seed: Optional[int] = 42,
    **kwargs,
):
    if seed is not None:
        np.random.seed(seed)

    try:
        timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit="s")
        full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
        S0 = float(full_prices.iloc[-1])
        steps = time_length // time_increment

        configs = get_ensemble_configs(asset, time_increment)
        final_paths_list = []
        total_sims_generated = 0

        print(f"[INFO] Starting Ensemble Simulation for {asset} (S0={S0:.2f})...")

        for i, cfg in enumerate(configs):
            if i == len(configs) - 1:
                n_sub_sims = n_sims - total_sims_generated
            else:
                n_sub_sims = int(n_sims * cfg["weight"])
            total_sims_generated += n_sub_sims

            points_per_day = 86400 // time_increment
            needed_points = int(cfg["lookback_days"] * points_per_day)
            hist_prices = full_prices.tail(needed_points) if len(full_prices) > needed_points else full_prices

            returns = compute_log_returns(hist_prices)
            res = fit_garch_robust(returns, cfg)
            actual_dist = res.model.distribution.name
            skew_param = res.params.get("lambda", 0)

            print(f"   -> [{cfg['name']}] Fit {actual_dist} (Skew={skew_param:.2f}). Simulating {n_sub_sims} paths...")

            paths = simulate_paths_vectorized(res, S0, steps, n_sub_sims, cfg["scale"], actual_dist)
            final_paths_list.append(paths)

        ensemble_paths = np.vstack(final_paths_list)
        return ensemble_paths
    except Exception as e:
        print(
            f"[WARN] grach_simulator_v4_2 ensemble failed ({e!r}); "
            f"falling back to garch_simulator_v2_2",
            flush=True,
        )
        return _fallback_garch_v2_2(
            prices_dict,
            asset,
            time_increment,
            time_length,
            n_sims,
            seed=seed,
            **kwargs,
        )

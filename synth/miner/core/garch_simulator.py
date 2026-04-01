"""
garch_simulator.py

Fit GARCH(1,1) with Student-t innovations and simulate multiple paths.

Requirements:
    pip install numpy pandas scipy arch tqdm

Usage example at bottom shows how to simulate 1000 paths for 3 assets.
"""
import json
from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from arch import arch_model
from tqdm import trange

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from price series (aligned)."""
    return np.log(prices).diff().dropna()

def fit_garch_studentt(returns: pd.Series, mean: str = "Constant", p: int = 1, q: int = 1):
    """
    Fit GARCH(p,q) with Student-t innovations.
    returns: pandas Series of log returns
    mean: "Zero" or "Constant" or "AR"
    Returns the fitted result object (ARCHModelResult).
    """
    # arch_model accepts mean='Zero'|'Constant'|'AR'
    model = arch_model(returns * 10000.0,  # scale to bps to stabilize numerics
                       mean=mean,
                       vol="GARCH",
                       p=p, q=q,
                       dist="StudentsT")
    res = model.fit(disp="off", show_warning=False)
    return res

def simulate_garch_paths(fitted_res,
                         S0: float,
                         steps: int,
                         n_sims: int,
                         seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    """
    Simulate paths from fitted GARCH(1,1) Student-t model.
    - fitted_res: result object from fit_garch_studentt
    - S0: starting price (float)
    - steps: number of future steps (e.g., 288 for 24h with 5-min steps)
    - n_sims: number of simulation paths
    Returns:
      prices: ndarray shape (n_sims, steps+1) including initial price at index 0
      meta: dict with model params used
    Notes:
      We model log-returns in basis points (bps) internally, then convert back to prices.
    """
    if seed is not None:
        np.random.seed(seed)

    params = fitted_res.params  # pandas Series
    # Parameter names in arch: 'mu' (or 'Const'), 'omega', 'alpha[1]', 'beta[1]', 'nu' (for StudentsT)
    # Depending on 'mean' arg, the param for mean could be 'mu' or 'Const' or 'AR.1', adjust:
    mu_candidates = [k for k in params.index if k.lower() in ("mu", "const", "mean")]
    mu = float(params[mu_candidates[0]]) if mu_candidates else 0.0

    omega = float(params.get("omega", params.get("omega[1]", 1e-6)))
    alpha = float(params.get("alpha[1]", params.get("alpha", 0.05)))
    beta = float(params.get("beta[1]", params.get("beta", 0.9)))
    nu = float(params.get("nu", params.get("student_t", 8.0)))  # degrees of freedom

    # Unconditional variance for GARCH(1,1): omega / (1 - alpha - beta)
    denom = 1.0 - alpha - beta
    if denom <= 0 or denom is None:
        # fallback safe starting sigma
        sigma0 = np.sqrt(max(omega, 1e-6))
    else:
        sigma0 = np.sqrt(omega / denom)

    # We'll simulate log-returns in bps scale (same scale as fitting)
    # We want z_t standardised to unit variance. For Student-t with df=nu,
    # variance = nu/(nu-2) for nu>2. So standardize by sqrt(nu/(nu-2)).
    if nu <= 2:
        scale_std = 1.0
    else:
        scale_std = np.sqrt(nu / (nu - 2.0))

    # generate all shocks z at once: Student-t, shape (steps, n_sims)
    z = student_t.rvs(df=nu, size=(steps, n_sims))
    if scale_std != 1.0:
        z = z / scale_std

    # SAFETY 1: clip shocks from fat tails
    z = np.clip(z, -5.0, 5.0)

    # pre-allocate arrays (returns in bps)
    returns_bps = np.zeros((steps, n_sims), dtype=np.float64)

    # initialize sigma_0
    sigma_prev = np.full(n_sims, sigma0, dtype=np.float64)
    eps_prev = np.zeros(n_sims, dtype=np.float64)

    # SAFETY 1.5: cap persistence to keep process mean-reverting
    beta_safe = min(beta, 0.98)

    # iterate time steps (vectorized over sims)
    for t in range(steps):
        sigma2 = omega + alpha * (eps_prev ** 2) + beta_safe * (sigma_prev ** 2)

        # SAFETY 2: cap variance to avoid numeric explosion
        sigma2 = np.clip(sigma2, 1e-12, 250000.0)
        sigma_t = np.sqrt(sigma2)

        eps_t = sigma_t * z[t, :]
        returns_bps[t, :] = mu + eps_t

        sigma_prev = sigma_t
        eps_prev = eps_t

    # Convert returns (bps) back to log-returns in decimal
    logret = returns_bps / 10000.0  # shape (steps, n_sims)

    # Build price paths (vectorized) with final guard
    prices = np.zeros((n_sims, steps + 1), dtype=np.float64)
    prices[:, 0] = S0
    cum_ret = np.cumsum(logret, axis=0)

    # SAFETY 3: clip cumulative returns before exp()
    cum_ret = np.clip(cum_ret, -4.0, 4.0)
    prices[:, 1:] = S0 * np.exp(cum_ret).T

    meta = {
        "mu": float(mu),
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "beta_safe": float(beta_safe),
        "nu": float(nu),
        "sigma0": float(sigma0),
        "n_sims": int(n_sims),
        "steps": int(steps),
    }
    return prices, meta

def simulate_single_price_path_with_garch(prices_dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: Optional[int] = 42, **kwargs):
    """
    Simulate a single price path with GARCH model.
    - prices_dict: dictionary of prices {"timestamp": "price"}
    - time_increment: time increment in seconds
    - time_length: time length in seconds
    - n_sims: number of simulation paths
    - seed: seed for the random number generator
    Returns:
      prices: ndarray shape (n_sims, steps+1) including initial price at index 0
      meta: dict with model params used
    """
    steps = time_length // time_increment

    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    prices_values = list(prices_dict.values())
    hist_prices = pd.Series(prices_values, index=timestamps)
    
    # Sort by timestamp to ensure chronological order
    hist_prices = hist_prices.sort_index()
    print(f"Date range: {hist_prices.index[0]} to {hist_prices.index[-1]}")
    print(f"First few prices: {hist_prices.head()}")
    returns = compute_log_returns(hist_prices)

    mean_model = kwargs.get("mean", "Constant")
    p_param = kwargs.get("p", 1)
    q_param = kwargs.get("q", 1)

    print(f"[INFO] Fitting GARCH model on historical price data, length: {len(returns)}")
    res = fit_garch_studentt(returns, mean=mean_model, p=p_param, q=q_param)
    print(res.summary().as_text()[:400])  # truncated summary

    # simulate n_sims paths
    S0 = float(hist_prices.iloc[-1])
    print(f"Simulating {n_sims} paths, steps={steps}, start price S0={S0:.2f}")
    prices, meta = simulate_garch_paths(res, S0=S0, steps=steps, n_sims=n_sims, seed=seed)
    print(f"[INFO] Simulation done. Example first path (first 10 points and last 10 points): {prices[0, :10]} ... {prices[0, -10:]}")

    return prices


if __name__ == "__main__":
    # Example parameters
    time_increment = 300            # 5 minutes
    time_length = 86400             # 24 hours
    n_sims = 10

    # Load historical prices from JSON file
    json_path = "synth/miner/data/BTC_5m_prices.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Extract 5m price data
    prices_dict = data["5m"]
    
    simulate_single_price_path_with_garch(prices_dict, time_increment, time_length, n_sims)

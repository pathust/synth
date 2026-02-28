import numpy as np


def simulate_heston_with_jumps(
    prices_dict: dict,
    time_increment: int,
    time_length: int,
    n_sims: int,
):
    """
    Heston + Jump Diffusion simulator

    Inputs:
        prices_dict: {"timestamp": "price"}
        time_increment: seconds per step (e.g. 300 for 5m)
        time_length: forecast horizon in seconds (e.g. 86400 for 1 day)
        n_sims: number of Monte Carlo paths

    Returns:
        np.ndarray shape: (n_sims, n_steps+1)
    """

    # ---- Extract historical prices ----
    prices_values = list(prices_dict.values())
    prices = np.array(prices_values, dtype=np.float64)
    log_prices = np.log(prices)
    log_returns = np.diff(log_prices)
    log_returns = log_returns[np.isfinite(log_returns)]

    # ---- Estimate drift and vol ----
    mu = np.mean(log_returns)
    var = np.var(log_returns)

    # ---- Heston parameters (simple calibration) ----
    kappa = 3.0                  # mean reversion speed
    theta = var                  # long-run variance
    sigma = 0.5 * np.sqrt(var)   # vol-of-vol

    # ---- Jump parameters (crypto-friendly defaults) ----
    lamb = 0.1                   # expected jumps per step
    mu_j = -0.02                 # average jump size (log)
    sigma_j = 0.05              # jump volatility

    # ---- Time grid ----
    steps = int(time_length // time_increment)
    dt = time_increment / 86400  # normalize to days

    # ---- Init arrays ----
    S0 = prices[-1]
    prices_sim = np.zeros((n_sims, steps + 1))
    prices_sim[:, 0] = S0

    v = np.ones(n_sims) * theta

    # ---- Monte Carlo simulation ----
    for t in range(1, steps + 1):
        # Variance process (Heston)
        z_v = np.random.randn(n_sims)
        dv = kappa * (theta - v) * dt + sigma * np.sqrt(np.maximum(v, 1e-10)) * np.sqrt(dt) * z_v
        v = np.maximum(v + dv, 1e-10)

        # Poisson jumps
        jump_occurs = np.random.poisson(lamb * dt, size=n_sims)
        jump_sizes = np.exp(mu_j + sigma_j * np.random.randn(n_sims)) - 1.0

        # Price process
        z_s = np.random.randn(n_sims)
        diffusion = (mu - 0.5 * v) * dt + np.sqrt(v * dt) * z_s
        jump_component = jump_occurs * jump_sizes

        prices_sim[:, t] = prices_sim[:, t - 1] * np.exp(diffusion) * (1.0 + jump_component)

    return prices_sim

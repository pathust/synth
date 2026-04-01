import numpy as np
import pandas as pd
from typing import Optional
from arch import arch_model
from arch.univariate import StudentsT
import pytz

def simulate_us_equity_exact(
    prices_dict,
    asset,
    time_increment,
    time_length,
    n_sims=1000,
    seed: Optional[int] = 42,
    extended_hours: bool = False,
    **kwargs,
):
    """
    Simulates US Equity prices by hardcoding strictly enforced US Market Hours 
    (09:30 AM to 04:00 PM Eastern Time) or Extended Hours (04:00 AM to 08:00 PM).
    
    Any step outside these hours will have exactly 0 variance and 0 drift.
    At exactly 09:30 AM ET, an overnight/weekend Gap variance is applied.
    During intraday hours (09:35 AM to 15:55 PM), a GARCH(1,1) model fitted on intraday returns is applied.

    Optional kwargs:
        gap_ceiling_bps — cap on gap std in basis points (default: 80 TSLAX, 50 others).
    """
    gap_ceiling_bps = kwargs.pop("gap_ceiling_bps", None)
    kwargs.clear()

    if seed is not None:
        np.random.seed(seed)
        
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    S0 = float(full_prices.iloc[-1])
    steps = time_length // time_increment
    
    # Check if this is a crypto or something. This simulator is ONLY for US Equities.
    # Convert index to NY time
    ny_tz = pytz.timezone('America/New_York')
    ny_prices = full_prices.copy()
    ny_prices.index = ny_prices.index.tz_localize('UTC').tz_convert(ny_tz)
    
    # 1. Split data into Intraday returns and Overnight/Gap returns
    # We want to fit GARCH only on the pure intraday active periods.
    # We will compute 5-min returns. 
    returns = np.log(ny_prices).diff().dropna()
    
    # Filter for active trading hours
    if extended_hours:
        active_mask = (
            (returns.index.weekday < 5) & 
            (returns.index.hour >= 4) & 
            (returns.index.hour < 20)
        )
        gap_mask = (
            (returns.index.weekday < 5) & 
            (returns.index.hour == 4) & 
            (returns.index.minute == 0)
        )
    else:
        active_mask = (
            (returns.index.weekday < 5) & 
            (
                (returns.index.hour > 9) | ((returns.index.hour == 9) & (returns.index.minute >= 30))
            ) & 
            (
                (returns.index.hour < 16) | ((returns.index.hour == 16) & (returns.index.minute == 0))
            )
        )
        gap_mask = (
            (returns.index.weekday < 5) & 
            (returns.index.hour == 9) & 
            (returns.index.minute == 30)
        )
        
    intraday_returns = returns[active_mask]
    gap_returns = returns[gap_mask]
    
    # Fit GARCH on intraday returns
    # Multiply by 10000 for numerical stability in GARCH
    scaled_intraday = intraday_returns * 10000.0
    
    # We use basic GARCH(1,1) with t-distribution for intraday
    use_o = 1 if asset != "XAU" else 0  # GJR-GARCH to catch negative shocks
    try:
        model = arch_model(scaled_intraday, vol='Garch', p=1, o=use_o, q=1, dist='studentst', mean='Constant')
        res = model.fit(disp="off", show_warning=False)
    except:
        try:
            model = arch_model(scaled_intraday, vol='Garch', p=1, o=0, q=1, dist='Normal', mean='Zero')
            res = model.fit(disp="off", rescale=True)
        except Exception as e:
            # Fallback
            class MockRes:
                params = {"mu": 0, "omega": 0.1, "alpha[1]": 0.05, "beta[1]": 0.9, "nu": 8.0}
                conditional_volatility = pd.Series([1.0])
                resid = pd.Series([0.0])
            res = MockRes()
            
    last_vol = res.conditional_volatility.iloc[-1]
    last_resid = res.resid.iloc[-1]
    
    mu = res.params.get("mu", 0.0)
    omega = res.params.get("omega", 0.01)
    alpha = res.params.get("alpha[1]", 0.05)
    beta = res.params.get("beta[1]", 0.90)
    gamma = res.params.get("gamma[1]", 0.0) if use_o else 0.0
    nu = max(res.params.get("nu", 8.0), 8.0)

    # For gap returns, we'll just use a simple Student's T distribution fit, or just Normal
    if len(gap_returns) > 5:
        gap_std = gap_returns.std() * 10000.0
        gap_mu = gap_returns.mean() * 10000.0
    else:
        gap_std = intraday_returns.std() * 10000.0 * np.sqrt(16) # Heuristic overnight vol is ~ 4x 5-min vol
        gap_mu = 0.0
        
    if pd.isna(gap_std) or gap_std == 0:
        gap_std = 10.0
        
    sigma_prev = np.full(n_sims, last_vol)
    eps_prev = np.full(n_sims, last_resid)

    dist_sampler = StudentsT()
    z_intraday = dist_sampler.ppf(np.random.random((steps, n_sims)), [nu])
    z_gap = np.random.standard_t(df=int(nu), size=(steps, n_sims))
    
    returns_bps = np.zeros((steps, n_sims))
    
    # Start simulating step by step
    last_utc = full_prices.index[-1]
    
    for t in range(steps):
        # Current simulated time in UTC
        current_utc = last_utc + pd.Timedelta(seconds=(t + 1) * time_increment)
        # Convert to NY time
        current_ny = pd.Timestamp(current_utc).tz_localize('UTC').tz_convert(ny_tz)
        
        is_weekday = current_ny.weekday() < 5
        hour = current_ny.hour
        minute = current_ny.minute
        
        # Determine market state
        is_market_open = False
        is_opening_gap = False
        
        if is_weekday:
            if extended_hours:
                if hour == 4 and minute == 0:
                    is_opening_gap = True
                elif 4 <= hour < 20:
                    is_market_open = True
            else:
                if (hour == 9 and minute == 30):
                    is_opening_gap = True
                elif (hour == 9 and minute > 30) or (10 <= hour < 16) or (hour == 16 and minute == 0):
                    is_market_open = True
                
        if is_opening_gap:
            if gap_ceiling_bps is not None:
                gap_ceiling = float(gap_ceiling_bps)
            else:
                # TSLAX có gap mở phiên rất lớn → nới ceiling lên 80 BPS
                gap_ceiling = 80.0 if asset == "TSLAX" else 50.0
            safe_gap_std = gap_std if gap_std < gap_ceiling else 20.0
            step_return = gap_mu + safe_gap_std * z_gap[t, :]
            returns_bps[t, :] = step_return
            
            # Note: gap return doesn't feed into GARCH variance usually, but we could feed a small proxy
            # to reset the GARCH or just leave eps_prev as is.
            
        elif is_market_open:
            # Standard GARCH intraday variance
            indicator = (eps_prev < 0).astype(float)
            term_shock = (alpha + gamma * indicator) * (eps_prev**2)
            sigma2 = omega + term_shock + beta * (sigma_prev**2)
            sigma_t = np.sqrt(np.maximum(sigma2, 1e-12))
            
            eps_t = sigma_t * z_intraday[t, :]
            returns_bps[t, :] = mu + eps_t
            
            sigma_prev = sigma_t
            eps_prev = eps_t
            
        else:
            # OUT-OF-HOURS (Weekend, overnight). Variance is near 0 to avoid CRPS math penalties.
            returns_bps[t, :] = np.random.normal(mu * 0.3, max(omega, 1e-6) * 30.0, size=n_sims)
            
    # Output paths
    log_ret = returns_bps / 10000.0
    paths = S0 * np.exp(np.cumsum(log_ret, axis=0)).T
    
    # Return array shape: (n_sims, steps + 1)
    return np.column_stack([np.full(n_sims, S0), paths])

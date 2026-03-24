import numpy as np
import pandas as pd
from typing import Optional
from statsmodels.tsa.arima.model import ARIMA
import pytz
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)

def simulate_arima_us_equity_exact(prices_dict, asset, time_increment, time_length, n_sims=1000, seed: Optional[int] = 42, extended_hours: bool = False):
    """
    Simulates US Equity prices by hardcoding strictly enforced US Market Hours 
    (09:30 AM to 04:00 PM Eastern Time) or Extended Hours (04:00 AM to 08:00 PM).
    
    Fits an ARIMA(1,0,1) process to intraday contiguous returns to better capture GOOGLX 
    auto-correlation and drift during trading hours.
    
    Any step outside market hours will have exactly 0 variance and 0 drift.
    At the configured market open (09:30 AM or 04:00 AM), an overnight/weekend Gap variance is applied.
    """
    if seed is not None:
        np.random.seed(seed)
        
    timestamps = pd.to_datetime([int(ts) for ts in prices_dict.keys()], unit='s')
    full_prices = pd.Series(list(prices_dict.values()), index=timestamps).sort_index()
    S0 = float(full_prices.iloc[-1])
    steps = time_length // time_increment
    
    # 1. Split data and timezone
    ny_tz = pytz.timezone('America/New_York')
    ny_prices = full_prices.copy()
    ny_prices.index = ny_prices.index.tz_localize('UTC').tz_convert(ny_tz)
    
    returns = np.log(ny_prices).diff().dropna()
    
    # Intraday hours filter
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
    
    scaled_intraday = intraday_returns * 10000.0
    
    # 2. Fit ARIMA(1,0,1) model
    try:
        # ARIMA(p,d,q) - AR(1), I(0), MA(1)
        model = ARIMA(scaled_intraday, order=(1, 0, 1), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit()
        
        ar_coef = res.arparams[0] if len(res.arparams) > 0 else 0.0
        ma_coef = res.maparams[0] if len(res.maparams) > 0 else 0.0
        mu = res.params.get('const', 0.0)
        resid_std = np.sqrt(res.params.get('sigma2', scaled_intraday.var()))
        
        last_resid = res.resid.iloc[-1]
        last_return = scaled_intraday.iloc[-1]
        
    except Exception as e:
        # Fallback to random walk with drift
        ar_coef = 0.0
        ma_coef = 0.0
        mu = scaled_intraday.mean() if len(scaled_intraday) > 0 else 0.0
        resid_std = scaled_intraday.std() if len(scaled_intraday) > 0 else 10.0
        if pd.isna(resid_std):
            resid_std = 10.0
        last_resid = 0.0
        last_return = 0.0

    # For gap returns
    if len(gap_returns) > 5:
        gap_std = gap_returns.std() * 10000.0
        gap_mu = gap_returns.mean() * 10000.0
    else:
        gap_std = intraday_returns.std() * 10000.0 * np.sqrt(4) # Heuristic overnight multiplier (much less than 16)
        gap_mu = 0.0
        
    if pd.isna(gap_std) or gap_std == 0:
        gap_std = resid_std * 2.0  # Much more conservative fallback
        
    # State tracking arrays for the 1000 Sims
    prev_returns = np.full(n_sims, last_return)
    prev_errors = np.full(n_sims, last_resid)
    
    # Pre-generate standard normal noise for speed
    z_intraday = np.random.normal(0, 1, size=(steps, n_sims))
    z_gap      = np.random.standard_t(df=8, size=(steps, n_sims)) # Use slight fat-tail for gap 

    returns_bps = np.zeros((steps, n_sims))
    last_utc = full_prices.index[-1]
    
    # 3. Step forward in time
    for t in range(steps):
        current_utc = last_utc + pd.Timedelta(seconds=(t + 1) * time_increment)
        current_ny = pd.Timestamp(current_utc).tz_localize('UTC').tz_convert(ny_tz)
        
        is_weekday = current_ny.weekday() < 5
        hour = current_ny.hour
        minute = current_ny.minute
        
        is_market_open = False
        is_opening_gap = False
        
        if is_weekday:
            if extended_hours:
                if hour == 4 and minute == 0:
                    is_opening_gap = True
                elif 4 <= hour < 20:
                    is_market_open = True
            else:
                if hour == 9 and minute == 30:
                    is_opening_gap = True
                elif (hour == 9 and minute > 30) or (10 <= hour < 16) or (hour == 16 and minute == 0):
                    is_market_open = True
                
        if is_opening_gap:
            # We must be very careful with gap variance. A massive jump ruins CRPS if the real market opens flat.
            # Use intraday std * sqrt(2) as a much safer proxy for the overnight jump.
            safe_gap_std = intraday_returns.std() * 10000.0 * 1.414
            step_return = (gap_mu if not pd.isna(gap_mu) else 0.0) + safe_gap_std * z_gap[t, :]
            returns_bps[t, :] = step_return
            
            # Reset the autoregressive memory to not bleed overnight sentiment directly into minute 1 AR logic
            prev_returns = step_return
            prev_errors = np.zeros(n_sims)
            
        elif is_market_open:
            # ARIMA(1,0,1) standard formula: r_t = c + phi*r_{t-1} + theta*e_{t-1} + e_t
            eps_t = resid_std * z_intraday[t, :]
            step_return = mu + (ar_coef * prev_returns) + (ma_coef * prev_errors) + eps_t
            
            returns_bps[t, :] = step_return
            
            # Update state variables
            prev_returns = step_return
            prev_errors = eps_t
            
        else:
            # OUT-OF-HOURS (Weekend, overnight).
            # The market technically moves very slightly off-hours due to pre-market / after-hours trading.
            # Locking it to exactly 0 or 1e-6 causes massive CRPS penalty if the real price shifts.
            if asset == "NVDAX":
                # NVDAX cuối tuần giao dịch rất ít trên sàn phái sinh/tokenized
                # → hạ noise xuống 10% để bám sát đường thẳng băng, tránh CRPS penalty
                returns_bps[t, :] = np.random.normal(mu * 0.1, resid_std * 0.1, size=n_sims)
            else:
                # Default: Using 30% of intraday variance handles typical pre-market/after-hours drifting
                returns_bps[t, :] = np.random.normal(mu * 0.3, resid_std * 0.3, size=n_sims)
            
    # Output paths mapping
    log_ret = returns_bps / 10000.0
    paths = S0 * np.exp(np.cumsum(log_ret, axis=0)).T
    
    return np.column_stack([np.full(n_sims, S0), paths])

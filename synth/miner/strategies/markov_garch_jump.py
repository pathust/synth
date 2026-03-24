"""
markov_garch_jump.py — Markov-Switching GARCH(1,1) + Jump-Diffusion

Chiến thuật ứng dụng Toán Tài chính "white-box" kết hợp:
1. Mô hình chuyển đổi trạng thái (Markov-Switching) 2 Regimes.
    - Regime 0: Thị trường đi ngang/thanh khoản thấp.
    - Regime 1: Thị trường biến động mạnh/tin tức.
2. GARCH(1,1) mô phỏng cụm biến động (volatility clustering) theo từng trạng thái.
3. Phân phối Student-t kết hợp Poisson Jumps để tái tạo hiện tượng "fat-tail" 
   (những cú giật flash dump / flash pump) vô cùng đặc trưng trên thị trường 5m của SPYX/Crypto.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import poisson

from synth.miner.strategies.base import BaseStrategy

class MarkovGarchJumpStrategy(BaseStrategy):
    name = "markov_garch_jump"
    description = (
        "Markov-Switching GARCH(1,1) + Jump-Diffusion with Asset-Specific Priors. "
        "Regime 0 (sideway): low vol, rare jumps. "
        "Regime 1 (volatile): high vol, frequent jumps."
    )
    supported_assets = ["SPYX", "BTC", "ETH", "SOL", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]
    supported_frequencies = ["low", "high"]

    # Bộ tham số mặc định (Dành cho Crypto hoặc SPYX)
    default_params = {
        "lookback_days": 15,
        "P00": 0.98, "P11": 0.95,
        "omega_0": 1e-09, "alpha_0": 0.03, "beta_0": 0.95, "lambda_0": 0.0002,
        "omega_1": 1e-08, "alpha_1": 0.10, "beta_1": 0.85, "lambda_1": 0.002,
        "mu_J": 0.0, "sigma_J": 0.0005, "nu": 5.0,
    }

    # BỘ THAM SỐ HARDCODED RIÊNG CHO TỪNG TÀI SẢN (Vô hiệu hóa Auto-fit bằng lookback_days = 0)
    ASSET_PRIORS = {
        "SPYX": {
            "lookback_days": 0,
            "P00": 0.98, "P11": 0.90,
            "omega_0": 1e-10, "alpha_0": 0.02, "beta_0": 0.96, "lambda_0": 0.0001,
            "omega_1": 1e-08, "alpha_1": 0.10, "beta_1": 0.85, "lambda_1": 0.005,
            "mu_J": 0.0, "sigma_J": 0.001, "nu": 5.0
        },
        "GOOGLX": {
            "lookback_days": 0, 
            "P00": 0.99, "P11": 0.90,
            "omega_0": 1e-12, "alpha_0": 0.01, "beta_0": 0.98, "lambda_0": 0.0,      
            "omega_1": 1e-08, "alpha_1": 0.15, "beta_1": 0.80, "lambda_1": 0.005,    
            "mu_J": 0.001, "sigma_J": 0.0005, "nu": 4.0
        },
        "NVDAX": {
            "lookback_days": 0, 
            "P00": 0.95, "P11": 0.85,
            "omega_0": 1e-09, "alpha_0": 0.05, "beta_0": 0.90, "lambda_0": 0.0001,   
            "omega_1": 5e-08, "alpha_1": 0.20, "beta_1": 0.70, "lambda_1": 0.01,     
            "mu_J": 0.0, "sigma_J": 0.002, "nu": 3.0           
        },
        "TSLAX": {
            "lookback_days": 0, 
            "P00": 0.97, "P11": 0.80,
            "omega_0": 1e-09, "alpha_0": 0.03, "beta_0": 0.95, "lambda_0": 0.0002,
            "omega_1": 8e-08, "alpha_1": 0.25, "beta_1": 0.60, "lambda_1": 0.015,    
            "mu_J": -0.001, "sigma_J": 0.0025, "nu": 3.0
        },
        "AAPLX": {
            "lookback_days": 0, 
            "P00": 0.99, "P11": 0.95,        
            "omega_0": 5e-10, "alpha_0": 0.02, "beta_0": 0.97, "lambda_0": 0.00001,  
            "omega_1": 5e-09, "alpha_1": 0.08, "beta_1": 0.90, "lambda_1": 0.001,    
            "mu_J": 0.0, "sigma_J": 0.0005, "nu": 6.0           
        }
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
        
        # 1. Nạp tham số ưu tiên theo Asset
        params = self.default_params.copy()
        if asset in self.ASSET_PRIORS:
            params.update(self.ASSET_PRIORS[asset])
            print(f"[MarkovGarchJump] Loaded dedicated hardcoded priors for {asset}")
        
        # Cho phép kwargs (từ grid search) ghi đè tham số nếu cần
        params.update(kwargs)

        # BTC/ETH: luôn ưu tiên Auto-Fit trên cửa sổ 15 ngày 5m.
        # Không áp dụng bộ hardcoded kiểu cổ phiếu Mỹ cho nhóm crypto này.
        if asset.upper() in {"BTC", "ETH"}:
            params["lookback_days"] = 15

        if seed is not None:
            np.random.seed(seed)
        
        # 2. Prepare Data
        if not prices_dict:
            raise ValueError("prices_dict is empty")
            
        sorted_keys = sorted([int(k) for k in prices_dict.keys()])
        S0 = float(prices_dict[str(sorted_keys[-1])])
        steps = time_length // time_increment
        
        # ==========================================================
        # 3. AUTO-FIT LOGIC (Thực nghiệm MSGARCH-JD)
        # ==========================================================
        lookback = params.get("lookback_days", 0)
        
        if lookback > 0 and len(sorted_keys) > 100:
            try:
                import pandas as pd
                from arch import arch_model
                
                needed_points = min(len(sorted_keys), int((lookback * 86400) // time_increment))
                hist_keys = sorted_keys[-needed_points:]
                hist_prices = pd.Series([float(prices_dict[str(k)]) for k in hist_keys])
                
                SCALE = 10000.0
                returns = np.log(hist_prices).diff().dropna()
                scaled_returns = returns * SCALE
                
                model = arch_model(scaled_returns, mean="Zero", vol="GARCH", p=1, q=1, dist="StudentsT")
                res = model.fit(disp="off", show_warning=False)
                
                nu_fit = max(float(res.params.get("nu", 5.0)), 3.0)
                omega_fit = float(res.params.get("omega", 1e-2)) / (SCALE ** 2)
                alpha_fit = float(res.params.get("alpha[1]", 0.1))
                beta_fit = float(res.params.get("beta[1]", 0.85))
                
                cond_vol_scaled = res.conditional_volatility
                cond_vol = cond_vol_scaled / SCALE
                median_vol = cond_vol.median()
                
                regimes = (cond_vol > median_vol).astype(int).values
                r_t = regimes[:-1]
                r_next = regimes[1:]
                
                mask_0 = (r_t == 0)
                P00_fit = np.mean(r_next[mask_0] == 0) if mask_0.sum() > 0 else 0.98
                
                mask_1 = (r_t == 1)
                P11_fit = np.mean(r_next[mask_1] == 1) if mask_1.sum() > 0 else 0.95
                    
                params["P00"] = np.clip(P00_fit, 0.85, 0.999)
                params["P11"] = np.clip(P11_fit, 0.85, 0.999)
                
                params["omega_0"] = omega_fit * 0.5
                params["alpha_0"] = np.clip(alpha_fit * 0.5, 0.01, 0.1)
                params["beta_0"]  = np.clip(beta_fit * 1.05, 0.5, 0.98)
                
                params["omega_1"] = omega_fit * 2.0
                params["alpha_1"] = np.clip(alpha_fit * 1.5, 0.1, 0.4)
                params["beta_1"]  = np.clip(beta_fit * 0.85, 0.5, 0.95)
                
                params["nu"] = nu_fit
                
                abs_ret = np.abs(returns.values[1:])
                jump_mask = abs_ret > (3.5 * cond_vol.values[:-1])
                jumps_val = returns.values[1:][jump_mask]
                
                lam_0 = np.sum(jump_mask[r_t == 0]) / max(len(r_t[r_t == 0]), 1)
                lam_1 = np.sum(jump_mask[r_t == 1]) / max(len(r_t[r_t == 1]), 1)
                
                params["lambda_0"] = np.clip(lam_0, 0.001, 0.1)
                params["lambda_1"] = np.clip(lam_1, 0.01, 0.2)
                
                if len(jumps_val) > 2:
                    params["mu_J"] = np.mean(jumps_val)
                    params["sigma_J"] = np.std(jumps_val)
                else:
                    params["mu_J"] = 0.0
                    params["sigma_J"] = 0.002
                    
                print(f"[Auto-Fit] {asset} MSGARCH: P00={params['P00']:.3f}, P11={params['P11']:.3f}, "
                      f"lam_0={params['lambda_0']:.4f}, lam_1={params['lambda_1']:.4f}, nu={nu_fit:.1f}")
                      
            except Exception as e:
                print(f"[Auto-Fit] MSGARCH Failed for {asset}: {e}. Falling back to default priors.")
        else:
            if asset not in self.ASSET_PRIORS:
                print(f"[Auto-Fit] Lookback disabled. Using default MSGARCH priors for {asset}.")

        # ==========================================================
        # 4. EXTRACTION
        # ==========================================================
        P00 = params["P00"]
        P11 = params["P11"]
        P01 = 1.0 - P00
        P10 = 1.0 - P11
        
        omega = np.array([params["omega_0"], params["omega_1"]])
        alpha = np.array([params["alpha_0"], params["alpha_1"]])
        beta  = np.array([params["beta_0"], params["beta_1"]])
        lam   = np.array([params["lambda_0"], params["lambda_1"]])
        
        mu_J = params["mu_J"]
        sigma_J = params["sigma_J"]
        nu = params["nu"]
        
        p0_steady = P10 / (P01 + P10) if (P01 + P10) > 0 else 1.0
        states = np.where(np.random.rand(n_sims) < p0_steady, 0, 1)
        
        var_0 = omega[0] / max(1.0 - alpha[0] - beta[0], 1e-6)
        var_1 = omega[1] / max(1.0 - alpha[1] - beta[1], 1e-6)
        h = np.where(states == 0, var_0, var_1)
        
        returns_m = np.zeros((steps, n_sims))
        
        rand_transition = np.random.rand(steps, n_sims)
        scale_std = np.sqrt(nu / (nu - 2.0)) if nu > 2 else 1.0
        z = student_t.rvs(df=nu, size=(steps, n_sims)) / scale_std
        
        eps_prev = np.zeros(n_sims) 

        # ==========================================================
        # 5. MONTE CARLO LOOP
        # ==========================================================
        for t in range(steps):
            
            # --- A. Regime Switching (Markov) ---
            rng_t = rand_transition[t]
            is_0 = (states == 0)
            is_1 = (states == 1)
            
            next_states = states.copy()
            next_states[is_0 & (rng_t > P00)] = 1 
            next_states[is_1 & (rng_t > P11)] = 0 
            states = next_states
            
            # --- B. GARCH(1,1) Equation ---
            omega_t = omega[states]
            alpha_t = alpha[states]
            beta_t  = beta[states]
            
            h = omega_t + alpha_t * (eps_prev ** 2) + beta_t * h
            h = np.clip(h, 1e-12, 1e-2) 
            
            # --- C. Student-t Innovation ---
            eps_t = np.sqrt(h) * z[t]
            eps_prev = eps_t
            
            # --- D. Jump Component (Poisson-Normal) ---
            lam_t = lam[states]
            N_jumps = np.random.poisson(lam_t)
            
            jump_totals = np.zeros(n_sims)
            has_jumps = N_jumps > 0
            if np.any(has_jumps):
                total_jumps = np.sum(N_jumps[has_jumps])
                if total_jumps > 0:
                    jump_magnitudes = np.random.normal(mu_J, sigma_J, size=total_jumps)
                    path_indices = np.repeat(np.arange(n_sims)[has_jumps], N_jumps[has_jumps])
                    np.add.at(jump_totals, path_indices, jump_magnitudes)
            
            # --- E. Total Return ---
            r_t = eps_t + jump_totals
            returns_m[t, :] = r_t
            
        # 6. Build Prices
        cum_ret = np.cumsum(returns_m, axis=0)
        prices = np.zeros((n_sims, steps + 1))
        prices[:, 0] = S0
        prices[:, 1:] = S0 * np.exp(cum_ret).T

        return prices
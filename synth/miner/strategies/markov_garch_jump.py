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

from synth.miner.strategies.base import BaseStrategy

class MarkovGarchJumpStrategy(BaseStrategy):
    name = "markov_garch_jump"
    description = (
        "Markov-Switching GARCH(1,1) + Jump-Diffusion with Asset-Specific Priors. "
        "Regime 0 (sideway): low vol, rare jumps. "
        "Regime 1 (volatile): high vol, frequent jumps."
    )
    supported_asset_types = ["crypto", "equity"]
    supported_regimes = []
    
    default_params = {
        "lookback_days": 15,
        "P00": 0.98, "P11": 0.95,
        "omega_0": 1e-09, "alpha_0": 0.03, "beta_0": 0.95, "lambda_0": 0.0002,
        "omega_1": 1e-08, "alpha_1": 0.10, "beta_1": 0.85, "lambda_1": 0.002,
        "mu_J": 0.0, "sigma_J": 0.0005, "nu": 5.0,
    }

    def get_param_grid(self, frequency: str = "low", asset: Optional[str] = None) -> dict:
        is_high = frequency == "high"
        grid = {"p": [1], "q": [1]}
        if is_high:
            grid["lookback_days"] = [14, 21]
            grid["nu"] = [3.0, 4.0]
        else:
            grid["lookback_days"] = [45, 60]
            grid["nu"] = [4.0, 6.0]
        return grid

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
        
        params = self.default_params.copy()
        params.update(kwargs)
        from synth.miner.core.markov_garch_jump_simulator import simulate_markov_garch_jump
        return simulate_markov_garch_jump(
            prices_dict=prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **params
        )
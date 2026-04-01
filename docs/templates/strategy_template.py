from __future__ import annotations

from typing import Optional

import numpy as np

from synth.miner.strategies.base import BaseStrategy


class NewStrategyTemplate(BaseStrategy):
    name = "new_strategy_template"
    version = "1.0"
    description = "Template strategy"
    supported_assets = ["BTC", "ETH"]
    supported_frequencies = ["high", "low"]
    default_params = {
        "drift": 0.0,
        "vol": 0.01,
    }
    param_grid = {
        "drift": [-0.0002, 0.0, 0.0002],
        "vol": [0.005, 0.01, 0.02],
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
        if not prices_dict:
            raise ValueError("prices_dict is empty")
        drift = float(kwargs.get("drift", self.default_params["drift"]))
        vol = float(kwargs.get("vol", self.default_params["vol"]))
        steps = max(1, time_length // time_increment)
        ts = sorted(prices_dict.keys(), key=int)
        s0 = float(prices_dict[ts[-1]])
        rng = np.random.default_rng(seed)
        z = rng.normal(0.0, 1.0, size=(n_sims, steps))
        increments = drift + vol * z
        log_paths = np.cumsum(increments, axis=1)
        paths = np.empty((n_sims, steps + 1), dtype=float)
        paths[:, 0] = s0
        paths[:, 1:] = s0 * np.exp(log_paths)
        return np.maximum(paths, 1e-12)

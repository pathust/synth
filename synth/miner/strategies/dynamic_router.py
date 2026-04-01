from typing import Optional
import numpy as np

from synth.miner.strategies.base import BaseStrategy
from synth.miner.strategies.pattern_detector import detect_pattern
from synth.miner.entry import _get_simulate_fn

class DynamicRouterStrategy(BaseStrategy):
    name = "dynamic_router"
    description = "Routes dynamically based on Precog 1h pattern detection (bullish/bearish/neutral)"
    supported_asset_types = []
    supported_regimes = []
    
    # Default routing rules found from our pattern backtest
    default_routing = {
        "BTC": {
            "bullish": "weekly_garch_v4",
            "bearish": "garch_v2_1",
            "neutral": "garch_v2_1"
        },
        "ETH": {
            "bullish": "gjr_garch",
            "bearish": "garch_v2_2",
            "neutral": "ensemble_garch_v2_v4"
        },
        "SOL": {
            "bullish": "gjr_garch",
            "bearish": "weekly_garch_v4",
            "neutral": "garch_v4_2"
        },
        "DEFAULT": {
            "bullish": "garch_v4",
            "bearish": "garch_v4",
            "neutral": "garch_v2"
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
        
        # 1. Detect market condition using last 3 hours of granular data
        pattern_data = detect_pattern(prices_dict)
        bias = pattern_data["bias"]
        score = pattern_data["bias_score"]
        
        # 2. Select strategy mapping
        routing = self.default_routing.get(asset, self.default_routing["DEFAULT"])
        selected_strategy_name = routing.get(bias, routing.get("neutral"))
        
        print(f"[DynamicRouter] Asset: {asset} | Detected: {bias.upper()} (Score: {score:.3f}) -> Routing to: {selected_strategy_name}")
        
        # 3. Fetch underlying simulation function
        sim_fn = _get_simulate_fn(selected_strategy_name)
        if sim_fn is None:
            print(f"[DynamicRouter ERROR] Strategy {selected_strategy_name} not found in registry. Falling back to garch_v2.")
            sim_fn = _get_simulate_fn("garch_v2")
            
        # 4. Route Execution
        # We pass exactly the same arguments to the chosen function
        return sim_fn(
            prices_dict,
            asset=asset,
            time_increment=time_increment,
            time_length=time_length,
            n_sims=n_sims,
            seed=seed,
            **kwargs
        )

# Register automatically
strategy = DynamicRouterStrategy()

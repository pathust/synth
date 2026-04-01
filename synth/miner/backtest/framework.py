from typing import List, Callable, Optional
import numpy as np
from synth.miner.data.dataloader import UnifiedDataLoader
from synth.miner.my_simulation import simulate_crypto_price_paths
from synth.validator.prompt_config import get_prompt_labels_for_asset, HIGH_FREQUENCY, LOW_FREQUENCY

class BacktestEngine:
    """
    Standardized Engine for evaluating strategies against historical snapshots.
    Guarantees no data leakage between the model's history context and the evaluation outcome.
    """
    def __init__(self, data_loader: UnifiedDataLoader, evaluators: List):
        self.data_loader = data_loader
        self.evaluators = evaluators

    def _get_config(self, asset: str):
        labels = get_prompt_labels_for_asset(asset) or []
        if "high" in labels:
            return HIGH_FREQUENCY
        return LOW_FREQUENCY

    def run(self, strategy_name: str, simulate_fn: Callable, asset: str, eval_slots: List[str], num_simulations: int = 1000) -> List[dict]:
        """
        Executes an out-of-sample backtest over multiple target slots using the unified pipeline.
        
        Args:
            strategy_name: Name of the strategy string.
            simulate_fn: The legacy simulation function to test.
            asset: The target asset ticker.
            eval_slots: List of ISO datetime boundaries (where history ends and truth begins).
            num_simulations: Number of paths to generate.
            
        Returns:
            List[dict]: Metrics collected for each evaluated slot.
        """
        results = []
        cfg = self._get_config(asset)
        
        for slot in eval_slots:
            # 1. Simulate forward using the unified simulate_crypto_price_paths runner
            # This internally fetches strictly Out-Of-Sample data ending at `slot`.
            paths = simulate_crypto_price_paths(
                current_price=None,
                asset=asset,
                start_time=slot,
                time_increment=cfg.time_increment,
                time_length=cfg.time_length,
                num_simulations=num_simulations,
                simulate_fn=simulate_fn,
                seed=42
            )
            
            if paths is None or len(paths) == 0:
                print(f"[Engine] Skipping {slot}: Simulation failed or returned empty.")
                continue
            
            # 2. Fetch the real "future" truth data for calculating metrics
            truth = self.data_loader.get_future_data(asset, start_time=slot, time_length=cfg.time_length)
            
            if truth is None or len(truth) == 0:
                print(f"[Engine] Skipping {slot}: No truth data available for evaluation.")
                continue

            # 3. Evaluate predictions using plugged-in evaluators
            slot_metrics = {}
            for ev in self.evaluators:
                # E.g. CRPSEvaluator
                slot_metrics[ev.name] = ev.calculate(
                    predictions=paths, 
                    truth=truth, 
                    time_increment=cfg.time_increment, 
                    scoring_intervals=cfg.scoring_intervals
                )
            
            results.append({
                "slot": slot,
                "metrics": slot_metrics
            })
            
        return results

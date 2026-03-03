"""
backtest_framework.py

Flexible backtesting framework extending beyond CRPS to support custom fitness functions,
parameter tuning (GridSearch), and Ensemble simulations.

Usage:
    cd /Users/taiphan/Documents/synth
    PYTHONPATH=. python -m synth.miner.backtest_framework
"""

import json
import os
import sys
import time
import traceback
import itertools
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Callable, Optional, Any

# ── Auto-resolve `synth` module path ─────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from synth.miner.my_simulation import fetch_price_data, simulate_crypto_price_paths
from synth.miner.compute_score import cal_reward
from synth.miner.data_handler import DataHandler
from synth.db.models import ValidatorRequest
from synth.simulation_input import SimulationInput
from synth.utils.helpers import convert_prices_to_time_format
from synth.validator.response_validation_v2 import validate_responses

# Core simulators
from synth.miner.core.garch_simulator import simulate_single_price_path_with_garch as garch_v1
from synth.miner.core.grach_simulator_v2 import simulate_single_price_path_with_garch as garch_v2

# ── Metrics / Fitness Functions ──────────────────────────────────────

def compute_rmse(predictions: np.ndarray, real_prices: np.ndarray) -> float:
    """Calculate Root Mean Squared Error of the mean prediction path against real prices."""
    mean_pred = np.mean(predictions, axis=0) # Shape: (time_steps,)
    # Align lengths if mismatched
    min_len = min(len(mean_pred), len(real_prices))
    if min_len == 0: return float('inf')
    return float(np.sqrt(np.mean((mean_pred[:min_len] - real_prices[:min_len])**2)))

def compute_mae(predictions: np.ndarray, real_prices: np.ndarray) -> float:
    """Calculate Mean Absolute Error of the mean prediction path against real prices."""
    mean_pred = np.mean(predictions, axis=0)
    min_len = min(len(mean_pred), len(real_prices))
    if min_len == 0: return float('inf')
    return float(np.mean(np.abs(mean_pred[:min_len] - real_prices[:min_len])))

def compute_directional_accuracy(predictions: np.ndarray, real_prices: np.ndarray) -> float:
    """Calculate Directional Accuracy of the mean prediction path."""
    mean_pred = np.mean(predictions, axis=0)
    min_len = min(len(mean_pred), len(real_prices))
    if min_len <= 1: return 0.0
    
    pred_diff = np.diff(mean_pred[:min_len])
    real_diff = np.diff(real_prices[:min_len])
    
    correct_directions = np.sum(np.sign(pred_diff) == np.sign(real_diff))
    # Return as error metric (1 - accuracy) so smaller is better like the others
    acc = correct_directions / len(real_diff)
    return float(1.0 - acc)

# Set of available metrics
METRICS = {
    "CRPS": None, # Handled specially via cal_reward
    "RMSE": compute_rmse,
    "MAE": compute_mae,
    "DIR_ACC": compute_directional_accuracy
}

# ── Core Framework Components ───────────────────────────────────────

class Gym:
    """Evaluates a model using a specific fitness metric."""
    def __init__(self, data_handler: DataHandler, metric: str = "CRPS"):
        self.data_handler = data_handler
        self.metric = metric.upper()
        if self.metric not in METRICS:
            raise ValueError(f"Unknown metric {metric}. Available: {list(METRICS.keys())}")

    def evaluate(self, asset: str, start_time: datetime, time_increment: int, time_length: int, num_sims: int, seed: int, model_fn: Callable, **kwargs):
        t0 = time.time()
        try:
            simulation_input = SimulationInput(
                asset=asset, start_time=start_time.isoformat(),
                time_increment=time_increment, time_length=time_length, num_simulations=num_sims
            )
            
            # Generate paths
            paths = simulate_crypto_price_paths(
                current_price=None, asset=asset, start_time=start_time.isoformat(),
                time_increment=time_increment, time_length=time_length, num_simulations=num_sims,
                simulate_fn=model_fn, max_data_points=None, seed=seed, **kwargs
            )
            
            if paths is None or len(paths) == 0:
                return {"status": "FAIL", "error": "No paths generated", "elapsed": time.time() - t0, "score": float('inf')}
            
            predictions = convert_prices_to_time_format(paths.tolist(), start_time.isoformat(), time_increment)
            is_valid = validate_responses(predictions, simulation_input, "0") == "CORRECT"

            validator_request = ValidatorRequest(
                asset=asset, start_time=start_time, time_length=time_length, time_increment=time_increment
            )

            # Get real prices by relying on compute_score logic
            crps_score, _, _, real_prices = cal_reward(self.data_handler, validator_request, predictions)
            
            if self.metric == "CRPS":
                score = crps_score if crps_score != -1 else float('inf')
            else:
                score = METRICS[self.metric](paths, np.array(real_prices))
            
            return {
                "status": "SUCCESS", "score": score, "metric": self.metric,
                "format_valid": is_valid, "num_paths": paths.shape[0], "elapsed": round(time.time() - t0, 2),
                "kwargs_used": kwargs
            }
        except Exception as e:
            traceback.print_exc()
            return {"status": "ERROR", "error": str(e), "elapsed": round(time.time() - t0, 2), "score": float('inf')}

class GridSearch:
    """Performs parameter searching for a given model function."""
    def __init__(self, gym: Gym):
        self.gym = gym

    def run(self, asset: str, start_time: datetime, time_increment: int, time_length: int, num_sims: int, seed: int, model_fn: Callable, param_grid: Dict[str, List[Any]]):
        keys, values = zip(*param_grid.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        best_score = float('inf')
        best_params = None
        results = []

        print(f"[GridSearch] Optimizing {model_fn.__name__} over {len(permutations_dicts)} combinations...")
        for params in permutations_dicts:
            print(f"  Testing params: {params}")
            res = self.gym.evaluate(asset, start_time, time_increment, time_length, num_sims, seed, model_fn, **params)
            res["params"] = params
            results.append(res)
            
            if res["status"] == "SUCCESS" and res["score"] < best_score:
                best_score = res["score"]
                best_params = params
                
        print(f"[GridSearch] Best Params found: {best_params} with {self.gym.metric} score: {best_score}")
        return {"best_params": best_params, "best_score": best_score, "all_results": results}

class EnsembleSimulator:
    """Aggregates multiple simulated paths from different models into one."""
    def __init__(self, model_fns: List[Callable], weights: Optional[List[float]] = None):
        self.model_fns = model_fns
        self.weights = weights or [1.0 / len(model_fns)] * len(model_fns)

    def simulate(self, prices_dict, asset: str, time_increment: int, time_length: int, n_sims: int, seed: Optional[int] = 42, **kwargs):
        all_paths = []
        for i, model_fn in enumerate(self.model_fns):
            print(f"[Ensemble] Running model: {model_fn.__name__}")
            # Ensure each model gets a different seed if not explicitly passed across all
            paths = model_fn(prices_dict, asset, time_increment, time_length, n_sims, seed=seed+i if seed else None, **kwargs)
            all_paths.append(paths)
        
        # Weighted mean aggregation across different model predictions
        # Shape of each paths: (n_sims, time_steps)
        blended_paths = np.zeros_like(all_paths[0])
        for paths, w in zip(all_paths, self.weights):
            blended_paths += paths * w
            
        return blended_paths

def get_random_date(start_date: datetime, end_date: datetime, num_dates: int, seed: int = 42) -> List[datetime]:
    import random
    random.seed(seed)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days

    random_dates = []
    for _ in range(num_dates):
        random_number_of_days = random.randint(0, days_between_dates)
        # Randomize to minute precision blocks too to simulate real validator offsets
        random_start = start_date + timedelta(days=random_number_of_days, minutes=random.randint(0, 1440))
        # Ensure it rounds down to 5 minute bounds like actual validator
        random_start = random_start.replace(minute=random_start.minute - (random_start.minute % 5), second=0, microsecond=0)
        random_dates.append(random_start)
    return random_dates

def benchmark_test(
    gym: Gym, asset: str, time_increment: int, time_length: int, 
    num_sims: int, seed: int, model_fn: Callable, start_date: datetime, end_date: datetime, num_runs: int, **kwargs
):
    print(f"\n[Benchmark] Running {num_runs} tests for {asset} between {start_date.isoformat()} and {end_date.isoformat()}")
    dates = get_random_date(start_date, end_date, num_dates=num_runs, seed=seed)
    
    scores = []
    for idx, date_ in enumerate(dates):
        print(f"  [{idx+1}/{num_runs}] Testing {date_.isoformat()}...")
        res = gym.evaluate(asset, date_, time_increment, time_length, num_sims, seed, model_fn, **kwargs)
        if res["status"] == "SUCCESS":
            scores.append(res["score"])
        else:
            print(f"    Failed logic at {date_}: {res.get('error')}")
            
    if scores:
        print(f"\n[Benchmark] Completed! Average {gym.metric} score: {np.mean(scores):.4f}")
    else:
        print("\n[Benchmark] Completed with 0 successful scores.")
    return scores

# ── Main Example / Verification Script ────────────────────────────────
def main():
    RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "result")
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RESULT_DIR, f"backtest_framework_{timestamp}.log")
    
    sys.stdout = open(log_path, "w", encoding="utf-8") # Simple redirect for now
    
    print("=" * 80)
    print(f"BACKTEST FRAMEWORK RUN — {timestamp}")
    print("=" * 80)

    asset = "BTC"
    time_increment = 300
    time_length = 86400
    num_sims = 100
    seed = 42
    
    data_handler = DataHandler()
    gym = Gym(data_handler, metric="CRPS")
    
    # 1. Fetch data
    # We use only_load=True to avoid long historical fetches for the demonstration
    fetch_price_data(asset, time_increment, only_load=True)
    
    start_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(days=2)
    
    print("\n--- Testing Single Model Evaluation ---")
    res1 = gym.evaluate(asset, start_time, time_increment, time_length, num_sims, seed, garch_v1, p=1, q=1)
    print(f"Base GARCH(1,1) CRPS: {res1.get('score', 'N/A')}")
    
    print("\n--- Testing Grid Search Tuning ---")
    grid = GridSearch(gym)
    param_grid = {
        "p": [1, 2],
        "q": [1],
        "mean": ["Zero", "Constant"]
    }
    gs_res = grid.run(asset, start_time, time_increment, time_length, num_sims, seed, garch_v1, param_grid)

    print("\n--- Testing Ensemble Simulation ---")
    ensemble = EnsembleSimulator([garch_v1, garch_v2], weights=[0.5, 0.5])
    # Note: simulate_crypto_price_paths normally takes simulate_fn which is matched to ensemble.simulate
    res3 = gym.evaluate(asset, start_time, time_increment, time_length, num_sims, seed, ensemble.simulate)
    print(f"Ensemble (GARCHv1 + GARCHv2) CRPS: {res3.get('score', 'N/A')}")

    print("\n--- Testing Long-Range Benchmark Suite ---")
    # Using last 30 days window up until start_time, executing 10 random backtests to simulate sn50 benchmarking
    benchmark_end = start_time
    benchmark_start = benchmark_end - timedelta(days=30)
    grid_best_params = {"mean": "Zero", "dist": "StudentsT", "scale": 10000.0, "min_nu": 0.0, "vol_multiplier": 1.0, "lookback_days": 45}
    benchmark_test(gym, asset, time_increment, time_length, num_sims, seed, garch_v1, benchmark_start, benchmark_end, 10, **grid_best_params)

    print("\nFramework components verified successfully!")
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"Log written to {log_path}")

if __name__ == "__main__":
    main()

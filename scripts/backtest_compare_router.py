import sys
import os
import argparse
from datetime import datetime, timezone, timedelta

# Ensure the root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synth.miner.data.dataloader import UnifiedDataLoader
from synth.miner.backtest.framework import BacktestEngine
from synth.miner.backtest.metrics import CRPSEvaluator

from synth.miner.entry import (
    _build_simulator_functions, 
    _get_ensemble_builder, 
    _run_ensemble_via_builder
)
from synth.miner.config.asset_strategy_config import get_strategy_list

def run_comparison(asset: str, time_increment: int, time_length: int, days_back: int, num_slots: int, num_simulations: int = 50):
    loader = UnifiedDataLoader()
    evaluator = CRPSEvaluator()
    engine = BacktestEngine(data_loader=loader, evaluators=[evaluator])
    
    sim_fns = _build_simulator_functions()
    
    # ── Strategy 1: Dynamic Router ──
    dynamic_fn = sim_fns["dynamic_router"]
    
    # ── Strategy 2: Production Miner Ensemble ──
    # Create a wrapper function that behaves like a standard simulate_fn 
    # but internally calls the production generic ensemble builder
    def production_simulate_wrapper(
        prices_dict, asset, time_increment, time_length, n_sims, seed=42, **kwargs
    ):
        strategy_configs = get_strategy_list(asset, time_length)
        builder = _get_ensemble_builder()
        
        # start_time needs to be fetched from kwargs or we rely on the builder
        # Normally Engine relies on my_simulation which uses start_time. 
        # But wait! BacktestEngine already passes simulate_fn into my_simulation's simulate_crypto_price_paths.
        # my_simulation takes simulate_fn(filter_prices_dict, asset, time_increment, time_length, n_sims, seed, **kwargs).
        # We need start_time to pass to _run_ensemble_via_builder. In my_simulation, start_time translates to the end of prices_dict.
        # Actually, let's just cheat nicely by pulling the last timestamp from prices_dict as start_time representation.
        if prices_dict:
            last_ts = int(max(prices_dict.keys()))
            start_time = datetime.fromtimestamp(last_ts + 60, timezone.utc).isoformat().replace("+00:00", "Z")
        else:
            start_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            
        ensemble_paths = _run_ensemble_via_builder(
            builder, strategy_configs, sim_fns,
            asset, start_time, time_increment, time_length,
            n_sims, seed
        )
        return ensemble_paths

    # Generate evaluation slots (every 4 hours to get diverse situations)
    eval_slots = []
    base_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(days_back, days_back - num_slots, -1):
        for h in range(0, 24, 4):
            slot_dt = base_dt - timedelta(days=i) + timedelta(hours=h)
            eval_slots.append(slot_dt.isoformat().replace("+00:00", "Z"))
            
    print(f"=== Compare: Dynamic Router vs Production Ensemble for {asset} ===")
    print(f"Freq: {time_length}s, Slots: {len(eval_slots)}")
    print("--------------------------------------------------\n")
    
    results_summary = {"Dynamic Router": [], "Production Miner": []}
    
    # Evaluate Dynamic Router
    print("--> Evaluating strategy: Dynamic Router")
    router_results = engine.run(
        strategy_name="dynamic_router",
        simulate_fn=dynamic_fn,
        asset=asset,
        eval_slots=eval_slots,
        num_simulations=num_simulations
    )
    for r in router_results:
        crps = r["metrics"].get("CRPSEvaluator", -1.0)
        if crps > 0:
            results_summary["Dynamic Router"].append(crps)
            
    # Evaluate Production Miner
    print("\n--> Evaluating strategy: Production Miner (Ensemble)")
    prod_results = engine.run(
        strategy_name="production_miner",
        simulate_fn=production_simulate_wrapper,
        asset=asset,
        eval_slots=eval_slots,
        num_simulations=num_simulations
    )
    for r in prod_results:
        crps = r["metrics"].get("CRPSEvaluator", -1.0)
        if crps > 0:
            results_summary["Production Miner"].append(crps)
            
    # Final Output
    print("\n==================================================")
    print("COMPARISON RESULTS LEADERBOARD")
    print("==================================================")
    
    for name, scores in results_summary.items():
        if scores:
            mean = sum(scores) / len(scores)
            count = len(scores)
            print(f"{name:<20} : Mean CRPS = {mean:.4f} (across {count} valid slots)")
        else:
            print(f"{name:<20} : Failed or no valid scores.")
            
    print("\n[NOTE] Lower CRPS is better.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default="BTC")
    parser.add_argument("--days_back", type=int, default=3)
    parser.add_argument("--slots", type=int, default=1) 
    parser.add_argument("--sims", type=int, default=30)
    
    # High vs Low freq
    parser.add_argument("--high_freq", action="store_true", help="Run high frequency (1hr)")
    args = parser.parse_args()
    
    # High frequency = 3600 seconds, 60s increments
    # Low frequency = 86400 seconds, 300s increments
    inc = 60 if args.high_freq else 300
    leng = 3600 if args.high_freq else 86400
    
    run_comparison(args.asset, inc, leng, args.days_back, args.slots, args.sims)

import sys
import os
import argparse
from datetime import datetime, timezone, timedelta
import pandas as pd

# Ensure the root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synth.miner.data.dataloader import UnifiedDataLoader
from synth.miner.backtest.framework import BacktestEngine
from synth.miner.backtest.metrics import CRPSEvaluator

# Load the simulator functions directly from the entry
from synth.miner.entry import _build_simulator_functions

def run_sweep(asset: str, days_back: int, num_slots: int, num_simulations: int = 50):
    loader = UnifiedDataLoader()
    evaluator = CRPSEvaluator()
    engine = BacktestEngine(data_loader=loader, evaluators=[evaluator])
    
    sim_fns = _build_simulator_functions()
    
    # Filter strategies to test (focus on crypto/GARCH models)
    target_strategies = [
        name for name in sim_fns.keys() 
        if ("garch" in name.lower() or "har" in name.lower()) 
        and not name.endswith("_strat")
    ]
    
    print(f"=== Starting Backtest Sweep for {asset} ===")
    print(f"Strategies to evaluate: {target_strategies}")
    
    # Generate evaluation slots (e.g., last 5 days at 00:00 UTC)
    eval_slots = []
    base_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(days_back, days_back - num_slots, -1):
        slot_dt = base_dt - timedelta(days=i)
        eval_slots.append(slot_dt.isoformat().replace("+00:00", "Z"))
        
    print(f"Evaluation Slots: {eval_slots}")
    print("--------------------------------------------------\n")
    
    results_summary = []
    
    for strat_name in target_strategies:
        fn = sim_fns[strat_name]
        try:
            print(f"--> Evaluating strategy: {strat_name}")
            results = engine.run(
                strategy_name=strat_name,
                simulate_fn=fn,
                asset=asset,
                eval_slots=eval_slots,
                num_simulations=num_simulations
            )
            
            # Aggregate scores
            scores = []
            for r in results:
                crps = r["metrics"].get("CRPSEvaluator", -1.0)
                if crps > 0:
                    scores.append(crps)
                    
            if scores:
                mean_crps = sum(scores) / len(scores)
                print(f"    Mean CRPS: {mean_crps:.4f} (over {len(scores)} slots)\n")
                results_summary.append({
                    "strategy": strat_name,
                    "mean_crps": mean_crps,
                    "valid_slots": len(scores)
                })
            else:
                print("    Failed to generate valid CRPS scores.\n")
                
        except Exception as e:
            print(f"    Error evaluating {strat_name}: {e}\n")

    if not results_summary:
        print("No successful backtest results.")
        return

    # Sort results
    results_summary.sort(key=lambda x: x["mean_crps"])
    
    print("==================================================")
    print(f"BACKTEST RESULTS LEADERBOARD FOR {asset}")
    print("==================================================")
    for i, res in enumerate(results_summary):
        print(f"#{i+1} {res['strategy']:<15} : {res['mean_crps']:.4f} (over {res['valid_slots']} slots)")
    print("==================================================")
    print(f"Best Strategy for {asset} is: {results_summary[0]['strategy']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep strategies for an asset")
    parser.add_argument("--asset", type=str, default="BTC", help="Asset to backtest")
    parser.add_argument("--days_back", type=int, default=5, help="Start evaluating from N days ago")
    parser.add_argument("--slots", type=int, default=3, help="Number of daily slots to test")
    parser.add_argument("--sims", type=int, default=100, help="Number of simulations per eval")
    
    args = parser.parse_args()
    run_sweep(args.asset, args.days_back, args.slots, args.sims)

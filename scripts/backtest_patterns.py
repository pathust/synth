import sys
import os
import argparse
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# Ensure the root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synth.miner.data.dataloader import UnifiedDataLoader
from synth.miner.backtest.engine import BacktestEngine
from synth.miner.backtest.metrics import CRPSEvaluator
from synth.miner.entry import _build_simulator_functions
from synth.miner.strategies.pattern_detector import detect_pattern

def run_pattern_backtest(asset: str, days_back: int, num_slots: int, num_simulations: int = 50):
    loader = UnifiedDataLoader()
    evaluator = CRPSEvaluator()
    engine = BacktestEngine(data_loader=loader, evaluators=[evaluator])
    sim_fns = _build_simulator_functions()
    
    target_strategies = [
        name for name in sim_fns.keys() 
        if ("garch" in name.lower() or "har" in name.lower()) 
        and not name.endswith("_strat")
    ]
    
    print(f"=== Pattern-based Backtest Sweep for {asset} ===")
    
    # Generate evaluation slots
    eval_slots = []
    base_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(days_back, days_back - num_slots, -1):
        # We can evaluate every 4 hours to get more diverse slots
        for h in range(0, 24, 4):
            slot_dt = base_dt - timedelta(days=i) + timedelta(hours=h)
            eval_slots.append(slot_dt.isoformat().replace("+00:00", "Z"))
            
    print(f"Total Evaluation Slots: {len(eval_slots)}")
    
    # Store results as: results[pattern][strategy] = [crps_scores]
    grouped_results = defaultdict(lambda: defaultdict(list))
    
    for slot in eval_slots:
        print(f"\n--- Evaluating Slot: {slot} ---")
        # 1. Detect Pattern
        # Need 30 days of data for GARCH, but pattern detector only needs 3 hours of 1m data.
        # We fetch historical dict up to this slot
        hist_dict = loader.get_historical_dict(asset, end_time=slot, window_days=5)
        if not hist_dict:
            print("No historic data, skipping.")
            continue
            
        pattern_data = detect_pattern(hist_dict)
        bias = pattern_data["bias"]
        b_score = pattern_data["bias_score"]
        print(f"Detected Pattern: {bias.upper()} (Score: {b_score:.3f})")
        
        # 2. Evaluate all strategies on this slot
        for strat in target_strategies:
            fn = sim_fns[strat]
            # We run engine on just this slot
            try:
                res = engine.run_slots(
                    strategy_name=strat,
                    simulate_fn=fn,
                    asset=asset,
                    eval_slots=[slot],
                    num_simulations=num_simulations
                )
                if res:
                    crps = res[0]["metrics"].get("CRPSEvaluator", -1)
                    if crps > 0:
                        grouped_results[bias][strat].append(crps)
            except Exception as e:
                pass

    print("\n==================================================")
    print("PATTERN-BASED BACKTEST LEADERBOARDS")
    print("==================================================")
    
    for pattern, strategies in grouped_results.items():
        print(f"\n[ PATTERN: {pattern.upper()} ]")
        summary = []
        for strat, scores in strategies.items():
            if scores:
                mean_crps = sum(scores) / len(scores)
                summary.append((strat, mean_crps, len(scores)))
                
        summary.sort(key=lambda x: x[1])
        for i, (strat, mean_crps, count) in enumerate(summary):
            print(f"  #{i+1} {strat:<15} : {mean_crps:.4f} (over {count} slots)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default="BTC")
    parser.add_argument("--days_back", type=int, default=5)
    parser.add_argument("--slots", type=int, default=2) # days to span
    parser.add_argument("--sims", type=int, default=30)
    args = parser.parse_args()
    
    run_pattern_backtest(args.asset, args.days_back, args.slots, args.sims)

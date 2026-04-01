"""
run_duel.py — CLI to run head-to-head strategy comparison.

Usage:
    cd /Users/taiphan/Documents/synth
    PYTHONPATH=. python -m synth.miner.backtest.run_duel \
        --baseline garch_v2 \
        --suggested jump_diffusion \
        --assets BTC ETH \
        --frequency high \
        --num-runs 10 \
        --num-sims 100

    # List available strategies:
    PYTHONPATH=. python -m synth.miner.backtest.run_duel --list
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from synth.miner.strategies import StrategyRegistry
from synth.miner.backtest.duel import StrategyDuel


def main():
    parser = argparse.ArgumentParser(
        description="Head-to-head strategy comparison backtest"
    )
    parser.add_argument("--list", action="store_true", help="List available strategies")
    parser.add_argument("--baseline", type=str, default="production_baseline",
                        help="Baseline strategy name (default: production_baseline = current generate_simulations)")
    parser.add_argument("--suggested", type=str, help="Suggested strategy name to compare against baseline")
    parser.add_argument("--assets", nargs="+", default=["BTC"], help="Assets to test")
    parser.add_argument("--frequency", type=str, default="high", choices=["high", "low"])
    parser.add_argument("--num-runs", type=int, default=10, help="Number of random time points per asset")
    parser.add_argument("--num-sims", type=int, default=100, help="Number of simulation paths")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--window-days", type=int, default=30, help="Date range window in days")
    parser.add_argument("--result-dir", type=str, default="result", help="Directory for JSON output")
    parser.add_argument("--compare-mainnet", action="store_true", help="Fetch mainnet scores from DB to compare ranking")

    args = parser.parse_args()

    # Auto-discover strategies
    registry = StrategyRegistry()
    registry.auto_discover()

    if args.list:
        print("\nAvailable strategies:")
        print("-" * 50)
        for name in registry.list_all():
            strat = registry.get(name)
            print(
                f"  {name:<25} assets={strat.supported_asset_types or 'all'} "
                f"regimes={strat.supported_regimes or 'all'}"
            )
        print()
        return

    if not args.suggested:
        parser.error("--suggested is required (e.g. --suggested jump_diffusion)")

    baseline = registry.get(args.baseline)
    suggested = registry.get(args.suggested)

    print(f"\n{'='*70}")
    print(f"  STRATEGY DUEL")
    print(f"  Baseline:  {baseline.name} — {baseline.description[:60]}")
    print(f"  Suggested: {suggested.name} — {suggested.description[:60]}")
    print(f"  Assets: {args.assets} | Frequency: {args.frequency}")
    print(f"  Runs: {args.num_runs} | Sims: {args.num_sims} | Seed: {args.seed}")
    if args.compare_mainnet:
        print(f"  Mainnet Compare: ENABLED")
    print(f"{'='*70}")

    duel = StrategyDuel(
        baseline=baseline, 
        suggested=suggested, 
        compare_mainnet=args.compare_mainnet
    )
    result = duel.run(
        assets=args.assets,
        frequency=args.frequency,
        num_runs=args.num_runs,
        num_sims=args.num_sims,
        seed=args.seed,
        window_days=args.window_days,
    )

    StrategyDuel.print_report(result)
    StrategyDuel.export_json(result, result_dir=args.result_dir)


if __name__ == "__main__":
    main()

"""
run_strategy_scan.py — Auto-scan all strategies for all assets/frequencies.

Usage:
    cd /Users/taiphan/Documents/synth
    PYTHONPATH=. conda run -n synth python -m synth.miner.run_strategy_scan

    # Or with custom options:
    PYTHONPATH=. conda run -n synth python -m synth.miner.run_strategy_scan \\
        --assets BTC ETH XAU \\
        --frequencies low \\
        --num-runs 3 \\
        --num-sims 50 \\
        --tune-best
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from synth.miner.strategies import StrategyRegistry
from synth.miner.backtest.runner import BacktestRunner
from synth.miner.backtest.tuner import GridSearchTuner
from synth.miner.backtest.report import BacktestReport


# Default asset lists per frequency
ALL_ASSETS_LOW = [
    "BTC", "ETH", "XAU", "SOL", "SPYX",
    "NVDAX", "TSLAX", "AAPLX", "GOOGLX",
]
ALL_ASSETS_HIGH = ["BTC", "ETH", "XAU", "SOL"]


def main():
    parser = argparse.ArgumentParser(
        description="Scan all strategies × assets × frequencies"
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=None,
        help="Assets to scan (default: all from prompt_config)",
    )
    parser.add_argument(
        "--frequencies",
        nargs="+",
        default=["low"],
        choices=["high", "low"],
        help="Frequency labels to scan",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of random backtest dates per strategy",
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=100,
        help="Number of simulation paths per run",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Days window for random date selection",
    )
    parser.add_argument(
        "--tune-best",
        action="store_true",
        help="Run GridSearch on the best strategy per asset after scanning",
    )
    parser.add_argument(
        "--metric",
        default="CRPS",
        choices=["CRPS", "RMSE", "MAE", "DIR_ACC"],
        help="Scoring metric",
    )
    parser.add_argument(
        "--result-dir",
        default="result",
        help="Directory for output files",
    )
    parser.add_argument(
        "--tune-regimes",
        action="store_true",
        help="Tune strategies per market regime (bullish/bearish/neutral) instead of randomly"
    )
    args = parser.parse_args()

    # ── 1. Discover strategies ──
    registry = StrategyRegistry()
    registry.auto_discover()

    # ── 2. Determine assets ──
    if args.assets:
        assets = args.assets
    else:
        assets = set()
        if "low" in args.frequencies:
            assets.update(ALL_ASSETS_LOW)
        if "high" in args.frequencies:
            assets.update(ALL_ASSETS_HIGH)
        assets = sorted(assets)

    print(f"Assets: {assets}")
    print(f"Frequencies: {args.frequencies}")
    print(f"Strategies: {registry.list_all()}")
    print(f"Metric: {args.metric}")

    # ── 3. Run scan ──
    runner = BacktestRunner(metric=args.metric)
    scan_results = runner.scan_all(
        assets=assets,
        frequencies=args.frequencies,
        registry=registry,
        num_runs=args.num_runs,
        num_sims=args.num_sims,
        seed=args.seed,
        window_days=args.window_days,
    )

    # ── 4. Report ──
    report = BacktestReport(result_dir=args.result_dir)
    report.print_summary(scan_results)

    # ── 5. Tune best strategies ──
    tuning_results = []
    if args.tune_best:
        rankings = report.generate_rankings(scan_results)
        tuner = GridSearchTuner(runner)

        for key, info in rankings["per_asset_best"].items():
            strat_name = info["best_strategy"]
            asset = info["asset"]
            freq = info["frequency"]

            strategy = registry.get(strat_name)
            if not strategy.get_param_grid():
                print(
                    f"\n[Tune] Skipping {strat_name} — no param_grid defined"
                )
                continue

            if args.tune_regimes:
                print(f"\n[Tuner] Grid Search theo Regime cho {strat_name} tr\u00ean {asset}")
                tune_result = tuner.run(
                    strategy,
                    asset,
                    freq,
                    num_runs=max(args.num_runs, 3),
                    num_sims=args.num_sims,
                    seed=args.seed,
                    window_days=args.window_days,
                    use_regimes=True
                )
            else:
                print(f"\n[Tuner] Grid Search chung cho {strat_name} tr\u00ean {asset}")
                tune_result = tuner.run(
                    strategy,
                    asset,
                    freq,
                    num_runs=max(args.num_runs, 3),
                    num_sims=args.num_sims,
                    seed=args.seed,
                    window_days=args.window_days,
                )
            tuning_results.append(tune_result)

    # ── 6. Export ──
    report.export_json(scan_results, tuning_results)

    print("\n✅ Strategy scan complete!")


if __name__ == "__main__":
    main()

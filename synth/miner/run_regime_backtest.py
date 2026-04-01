"""
run_regime_backtest.py — taxonomy-driven tuning/evaluation for strategies.

Usage:
    PYTHONPATH=. conda run -n synth python -m synth.miner.run_regime_backtest \
      --input-csv data/history.csv \
      --output-dir result/regime_backtest
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

# Ensure project root is on path when script is executed directly.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from synth.miner.backtest import (
    PredictionBacktestEngine,
    RegimeEngineConfig,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run taxonomy-based prediction/backtest engine from a CSV file."
    )
    parser.add_argument("--input-csv", required=True, help="Path to historical CSV.")
    parser.add_argument("--output-dir", default="result/regime_backtest", help="Directory for outputs.")
    parser.add_argument("--assets", nargs="+", default=None, help="Optional asset filter.")
    parser.add_argument("--strategies", nargs="+", default=None, help="Optional strategy whitelist.")

    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--asset-col", default="asset")
    parser.add_argument("--price-col", default="close")
    parser.add_argument("--volume-col", default="volume")
    parser.add_argument("--frequency", default="low", choices=["high", "low"])
    parser.add_argument("--num-sims", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-case-points", type=int, default=120)
    parser.add_argument("--min-split-points", type=int, default=9)
    parser.add_argument("--max-tune-dates", type=int, default=24)
    parser.add_argument("--max-eval-dates", type=int, default=24)
    parser.add_argument("--max-param-combinations", type=int, default=0)
    parser.add_argument("--split-mode", default="single", choices=["single", "walk_forward"])
    parser.add_argument("--walk-forward-folds", type=int, default=3)
    parser.add_argument("--walk-forward-min-train-ratio", type=float, default=0.5)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.input_csv)
    if args.assets:
        assets = {a.upper() for a in args.assets}
        df = df[df[args.asset_col].astype(str).str.upper().isin(assets)].copy()

    config = RegimeEngineConfig(
        timestamp_col=args.timestamp_col,
        asset_col=args.asset_col,
        price_col=args.price_col,
        volume_col=args.volume_col,
        frequency=args.frequency,
        num_sims=args.num_sims,
        seed=args.seed,
        min_case_points=args.min_case_points,
        min_split_points=args.min_split_points,
        max_tune_dates=args.max_tune_dates,
        max_eval_dates=args.max_eval_dates,
        max_param_combinations=(
            args.max_param_combinations if args.max_param_combinations > 0 else None
        ),
        split_mode=args.split_mode,
        walk_forward_folds=args.walk_forward_folds,
        walk_forward_min_train_ratio=args.walk_forward_min_train_ratio,
    )

    engine = PredictionBacktestEngine()
    report = engine.run(df, config=config, strategy_names=args.strategies)

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"regime_report_{stamp}.json")
    taxonomy_yaml = os.path.join(args.output_dir, f"regime_taxonomy_{stamp}.yaml")
    runtime_yaml = os.path.join(args.output_dir, f"strategies_runtime_{stamp}.yaml")

    engine.export_report_json(report, json_path)
    engine.export_taxonomy_yaml(report, taxonomy_yaml)
    engine.export_runtime_yaml(
        report,
        runtime_yaml,
        default_frequency=args.frequency,
        max_models_per_asset=3,
    )

    print(f"[RegimeBacktest] Cases: {report.get('num_cases', 0)}")
    print(f"[RegimeBacktest] Report: {json_path}")
    print(f"[RegimeBacktest] Taxonomy YAML: {taxonomy_yaml}")
    print(f"[RegimeBacktest] Runtime YAML: {runtime_yaml}")


if __name__ == "__main__":
    main()

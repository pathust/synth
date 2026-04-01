"""
run_strategy_scan.py — Auto-scan all strategies for all assets/frequencies.

One process always runs: scan_all → print_summary → optional export.
Adding --tune-best runs grid search *after* the same scan (see --tune-top-k). Add
--tune-regimes to grid-search per production regime (``detect_regime``: crypto high vs low,
gold BBW, equity clock). Use --tune-regimes-pattern for legacy bullish/bearish/neutral on 1m.
Skip --tune-best if you only want rankings without tuning (faster).

Why scan before tune: tuning every strategy on a full grid is usually too expensive; the
scan picks a shortlist. For a fairer final pick, use --tune-top-k 3 to tune the top 3
scan leaders per asset×frequency, then compare tuned scores in tuning_results.

Examples:
    # Full default grid: all LOW pairs first, then all HIGH (no --assets / --frequencies)
    PYTHONPATH=. uv run python -m synth.miner.run_strategy_scan --scan-all-default

    # Scan only (leaderboard, no tuning)
    PYTHONPATH=. uv run python -m synth.miner.run_strategy_scan --assets BTC --frequencies high

    # Scan + tune + optional draft YAML (copy of strategies.yaml, only tuned regimes patched)
    PYTHONPATH=. uv run python -m synth.miner.run_strategy_scan \\
        --assets BTC --frequencies high --tune-best --tune-regimes \\
        --write-strategies-draft --result-dir result/btc_high
"""

from __future__ import annotations

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
from synth.miner.backtest.strategies_draft import write_strategies_draft_yaml


# Default asset lists per frequency
ALL_ASSETS_LOW = [
    "BTC", "ETH", "XAU", "SOL", "SPYX",
    "NVDAX", "TSLAX", "AAPLX", "GOOGLX",
]
ALL_ASSETS_HIGH = ["BTC", "ETH", "XAU", "SOL"]


def _full_default_scan_pairs() -> tuple[list[str], list[tuple[str, str]]]:
    """
    Sequential scan grid from synth.validator.prompt_config:

    - **low**: ``LOW_FREQUENCY.asset_list`` (nine tickers including equities)
    - **high**: ``HIGH_FREQUENCY.asset_list`` (BTC, ETH, XAU, SOL)

    Order: **low block first** (5m / LOW prompt config), **then high block** (1m / HIGH).
    Data per pair: ``run_single`` → ``fetch_price_data(asset, time_increment)`` with
    ``time_increment`` 300 (low) or 60 (high) — aligned with validator prompts.
    """
    from synth.validator.prompt_config import HIGH_FREQUENCY, LOW_FREQUENCY

    pairs: list[tuple[str, str]] = []
    for a in LOW_FREQUENCY.asset_list:
        pairs.append((a, "low"))
    for a in HIGH_FREQUENCY.asset_list:
        pairs.append((a, "high"))
    assets = sorted({a for a, _ in pairs})
    return assets, pairs


def main():
    parser = argparse.ArgumentParser(
        description="Scan all strategies × assets × frequencies"
    )
    parser.add_argument(
        "--scan-all-default",
        action="store_true",
        help=(
            "Run the full default grid from prompt_config without --assets/--frequencies: "
            "for each asset in LOW_FREQUENCY.asset_list (low / 5m), then for each asset in "
            "HIGH_FREQUENCY.asset_list (high / 1m). Order: low block → high block."
        ),
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        default=None,
        help="Assets to scan (ignored when --scan-all-default is set)",
    )
    parser.add_argument(
        "--frequencies",
        nargs="+",
        default=["low"],
        choices=["high", "low"],
        help="Frequency labels to scan (ignored when --scan-all-default is set)",
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
        help="Run GridSearch after scanning on the top --tune-top-k strategies per asset×frequency",
    )
    parser.add_argument(
        "--tune-top-k",
        type=int,
        default=3t,
        metavar="K",
        help="With --tune-best, grid-search the K best strategies per asset×frequency (default: 1)",
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
        help=(
            "Tune per regime using date buckets (default: production detect_regime — "
            "crypto bull/high_vol/ranging, gold trending/mean_reverting, equity sessions)."
        ),
    )
    parser.add_argument(
        "--tune-regimes-pattern",
        action="store_true",
        help=(
            "With --tune-regimes, use legacy 1m pattern buckets (bullish/bearish/neutral) "
            "instead of synth.miner.regimes.detector.detect_regime."
        ),
    )
    parser.add_argument(
        "--production-regime-pool-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "With --tune-regimes (production mode), sample N random candidate dates per "
            "asset×frequency to fill regime buckets (default: max(2500, num_runs*150); "
            "raise for rare buckets e.g. equity earnings)."
        ),
    )
    parser.add_argument(
        "--write-strategies-draft",
        action="store_true",
        help=(
            "After tuning, write result_dir/strategies_draft_<timestamp>.yaml: copy of "
            "strategies.yaml with only tune-regimes cells patched (rank_slot=1). "
            "Requires --tune-best and --tune-regimes."
        ),
    )
    parser.add_argument(
        "--strategies-yaml-base",
        default=None,
        help="Path to base strategies.yaml (default: synth/miner/config/strategies.yaml)",
    )
    args = parser.parse_args()

    if args.scan_all_default:
        if args.assets is not None:
            print(
                "[scan-all-default] Ignoring --assets; using HIGH/LOW lists from "
                "synth.validator.prompt_config"
            )
        if "--frequencies" in sys.argv:
            print(
                "[scan-all-default] Ignoring --frequencies; using high (HIGH list) "
                "then low (LOW list)"
            )

    # ── 1. Discover strategies ──
    registry = StrategyRegistry()
    registry.auto_discover()

    # ── 2. Determine assets and optional explicit (asset, freq) pairs ──
    scan_pairs: list[tuple[str, str]] | None = None
    if args.scan_all_default:
        assets, scan_pairs = _full_default_scan_pairs()
        tune_pairs = scan_pairs
        print(
            "Mode: --scan-all-default (high=HIGH_FREQUENCY.asset_list, "
            "low=LOW_FREQUENCY.asset_list)"
        )
    else:
        if args.assets:
            assets = args.assets
        else:
            assets_set: set[str] = set()
            if "low" in args.frequencies:
                assets_set.update(ALL_ASSETS_LOW)
            if "high" in args.frequencies:
                assets_set.update(ALL_ASSETS_HIGH)
            assets = sorted(assets_set)
        tune_pairs = [(a, f) for a in assets for f in args.frequencies]

    print(f"Assets: {assets}")
    print(f"Frequencies: {args.frequencies}" + (" (fixed by --scan-all-default)" if args.scan_all_default else ""))
    if scan_pairs is not None:
        print(f"Scan order: {len(scan_pairs)} pairs (low block → high block)")

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
        pairs=scan_pairs,
    )

    # ── 4. Report ──
    report = BacktestReport(result_dir=args.result_dir)
    report.print_summary(scan_results)

    # ── 5. Tune top-K strategies per asset×frequency (after scan shortlist) ──
    tuning_results = []
    if args.tune_best:
        if args.tune_top_k < 1:
            raise SystemExit("--tune-top-k must be >= 1")
        tuner = GridSearchTuner(runner)

        for asset, freq in tune_pairs:
            ranked = report.ranked_strategies_for_asset_freq(
                scan_results, asset, freq, top_k=args.tune_top_k
            )
            if not ranked:
                continue
            for slot, r in enumerate(ranked, start=1):
                strat_name = r["strategy"]
                strategy = registry.get(strat_name)
                if not strategy.get_param_grid():
                    print(
                        f"\n[Tune] Skipping {strat_name} ({asset}/{freq}) "
                        f"[{slot}/{len(ranked)}] — no param_grid defined"
                    )
                    continue

                if args.tune_regimes:
                    print(
                        f"\n[Tuner] Grid Search theo Regime: {strat_name} "
                        f"({asset}/{freq}) [{slot}/{len(ranked)}]"
                    )
                    tune_result = tuner.run(
                        strategy,
                        asset,
                        freq,
                        num_runs=max(args.num_runs, 3),
                        num_sims=args.num_sims,
                        seed=args.seed,
                        window_days=args.window_days,
                        use_regimes=True,
                        regime_date_mode=(
                            "pattern" if args.tune_regimes_pattern else "production"
                        ),
                        production_regime_pool_size=args.production_regime_pool_size,
                    )
                else:
                    print(
                        f"\n[Tuner] Grid Search chung: {strat_name} "
                        f"({asset}/{freq}) [{slot}/{len(ranked)}]"
                    )
                    tune_result = tuner.run(
                        strategy,
                        asset,
                        freq,
                        num_runs=max(args.num_runs, 3),
                        num_sims=args.num_sims,
                        seed=args.seed,
                        window_days=args.window_days,
                    )
                tuning_results.append(
                    {
                        "asset": asset,
                        "frequency": freq,
                        "rank_slot": slot,
                        "scan_avg_score": r["avg_score"],
                        **tune_result,
                    }
                )

    # ── 6. Export ──
    report.export_json(scan_results, tuning_results)

    if args.write_strategies_draft:
        write_strategies_draft_yaml(
            tuning_results,
            base_path=args.strategies_yaml_base,
            result_dir=args.result_dir,
            rank_slot=1,
        )

    print("\n✅ Strategy scan complete!")


if __name__ == "__main__":
    main()

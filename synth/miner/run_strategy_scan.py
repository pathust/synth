"""
run_strategy_scan.py — Scan all strategies, then optional per-regime tuning.

Flow:

1. **scan_all** — every compatible strategy × asset × frequency; random dates in ``window_days``.
2. If **--tune-best** (requires **--tune-regimes**): for each (asset, frequency), tune **every**
   eligible strategy with production regime buckets (or **--tune-regimes-pattern**), then
   **per_regime_winners** picks the best strategy per regime (lowest tuned score).

Export JSON includes ``per_regime_winners`` when tuning ran. Use **--write-strategies-draft**
for ``strategies_draft_regime_winners_*.yaml``.

Examples:
    PYTHONPATH=. uv run python -m synth.miner.run_strategy_scan --scan-all-default

    PYTHONPATH=. uv run python -m synth.miner.run_strategy_scan \\
        --scan-all-default --tune-best --tune-regimes \\
        --write-strategies-draft --result-dir result/full_tune
"""

from __future__ import annotations

import sys
import os
import warnings

# Suppress arch model DataScaleWarning and other warnings flooding PM2 logs
warnings.filterwarnings("ignore", category=Warning, module="arch")
warnings.filterwarnings("ignore", message=".*poorly scaled.*")

# Suppress statsmodels ValueWarning
warnings.filterwarnings("ignore", category=Warning, module="statsmodels")
warnings.filterwarnings("ignore", message=".*A date index has been provided.*")

# Ensure project root is on path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

# Importing backtest.runner pulls bittensor, which handles global --help before our argparse.
_SHOW_HELP = False
if __name__ == "__main__" and (
    "--help" in sys.argv or "-h" in sys.argv
):
    _SHOW_HELP = True
    sys.argv = [a for a in sys.argv if a not in ("--help", "-h")]

import argparse

from synth.miner.strategies import StrategyRegistry
from synth.miner.backtest.runner import BacktestRunner
from synth.miner.backtest.tuner import GridSearchTuner
from synth.miner.backtest.report import BacktestReport
from synth.miner.backtest.strategies_draft import (
    write_strategies_draft_from_regime_winners,
    write_strategies_draft_yaml,
)


# Default asset lists per frequency
ALL_ASSETS_LOW = [
    "BTC", "ETH", "XAU", "SOL", "SPYX",
    "NVDAX", "TSLAX", "AAPLX", "GOOGLX",
]
ALL_ASSETS_HIGH = ["BTC", "ETH", "XAU", "SOL"]


def _full_default_scan_pairs() -> tuple[list[str], list[tuple[str, str]]]:
    """
    Sequential scan grid from synth.validator.prompt_config:

    - **low**: ``LOW_FREQUENCY.asset_list``
    - **high**: ``HIGH_FREQUENCY.asset_list``

    Order: low block → high block.
    """
    from synth.validator.prompt_config import HIGH_FREQUENCY, LOW_FREQUENCY

    pairs: list[tuple[str, str]] = []
    for a in LOW_FREQUENCY.asset_list:
        pairs.append((a, "low"))
    for a in HIGH_FREQUENCY.asset_list:
        pairs.append((a, "high"))
    assets = sorted({a for a, _ in pairs})
    return assets, pairs


def _strategies_for_asset_freq(
    registry: StrategyRegistry,
    asset: str,
    frequency: str,
    *,
    skip_ensemble: bool = True,
):
    """All registered strategies for ``asset`` that support ``frequency``."""
    strategies = registry.get_for_asset(asset)
    if skip_ensemble:
        strategies = [s for s in strategies if s.name != "ensemble_weighted"]
    return [s for s in strategies if s.supports_frequency(frequency)]


def main():
    parser = argparse.ArgumentParser(
        description="Scan all strategies × assets × frequencies; optional per-regime tuning"
    )
    parser.add_argument(
        "--scan-all-default",
        action="store_true",
        help=(
            "Full grid from prompt_config: each LOW_FREQUENCY asset (low), then each "
            "HIGH_FREQUENCY asset (high). Ignores --assets / --frequencies."
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
        help="Frequency labels (ignored when --scan-all-default is set)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Random backtest dates per run / regime bucket size target",
    )
    parser.add_argument(
        "--num-sims",
        type=int,
        default=100,
        help="Simulation paths per run",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help=(
            "Days before 'now' used to sample random backtest dates only. "
            "Per-run history length is synth.miner.constants.HISTORY_WINDOW_DAYS."
        ),
    )
    parser.add_argument(
        "--tune-best",
        action="store_true",
        help=(
            "After scan: tune every eligible strategy per (asset, frequency) with regime "
            "buckets; then select best strategy per regime. Requires --tune-regimes."
        ),
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
            "Required with --tune-best: production detect_regime buckets (crypto / gold / equity)."
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
            "Sample N random candidate dates per asset×frequency for regime buckets "
            "(default: max(2500, num_runs*150))."
        ),
    )
    parser.add_argument(
        "--write-strategies-draft",
        action="store_true",
        help=(
            "Write strategies_draft_regime_winners_*.yaml from per_regime_winners when tuning; "
            "otherwise legacy patch from tuning_results if any."
        ),
    )
    parser.add_argument(
        "--strategies-yaml-base",
        default=None,
        help="Path to base strategies.yaml (default: synth/miner/config/strategies.yaml)",
    )
    if _SHOW_HELP:
        parser.print_help()
        raise SystemExit(0)
    args = parser.parse_args()

    if args.tune_best and not args.tune_regimes:
        raise SystemExit("--tune-best requires --tune-regimes (per-regime tuning only)")

    if args.scan_all_default:
        if args.assets is not None:
            print(
                "[scan-all-default] Ignoring --assets; using LOW then HIGH lists from "
                "synth.validator.prompt_config"
            )
        if "--frequencies" in sys.argv:
            print(
                "[scan-all-default] Ignoring --frequencies; using low block then high block"
            )

    # ── 1. Discover strategies ──
    registry = StrategyRegistry()
    registry.auto_discover()

    # ── 2. Assets and (asset, freq) pairs ──
    scan_pairs: list[tuple[str, str]] | None = None
    if args.scan_all_default:
        assets, scan_pairs = _full_default_scan_pairs()
        tune_pairs = scan_pairs
        print("Mode: --scan-all-default (low block → high block)")
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
        print(f"Scan order: {len(scan_pairs)} pairs (low → high)")

    print(f"Strategies: {registry.list_all()}")
    print(f"Metric: {args.metric}")

    # ── 3. Scan ──
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

    # ── 5. Tune: all strategies × (asset, freq), regime grid, then aggregate winners ──
    tuning_results = []
    if args.tune_best:
        tuner = GridSearchTuner(runner)
        print(
            "\n[Tuner] Per-regime sweep: every eligible strategy per asset×frequency "
            "(then per_regime_winners)"
        )
        for asset, freq in tune_pairs:
            strats = _strategies_for_asset_freq(registry, asset, freq)
            if not strats:
                continue
            print(f"\n[Tuner] {asset} × {freq}: {len(strats)} strategies")
            for strategy in strats:
                print(
                    f"\n[Tuner] Regime grid: {strategy.name} ({asset}/{freq})"
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
                tuning_results.append(
                    {
                        "asset": asset,
                        "frequency": freq,
                        "tune_mode": "per_regime_all_strategies",
                        "strategy": strategy.name,
                        **tune_result,
                    }
                )

    per_regime_winners: dict | None = None
    if tuning_results:
        per_regime_winners = BacktestReport.build_per_regime_winners(tuning_results)
        if per_regime_winners:
            report.print_per_regime_winners(per_regime_winners)

    # ── 6. Export ──
    report.export_json(
        scan_results, tuning_results, per_regime_winners=per_regime_winners
    )

    if args.write_strategies_draft:
        if per_regime_winners:
            write_strategies_draft_from_regime_winners(
                per_regime_winners,
                base_path=args.strategies_yaml_base,
                result_dir=args.result_dir,
            )
        else:
            write_strategies_draft_yaml(
                tuning_results,
                base_path=args.strategies_yaml_base,
                result_dir=args.result_dir,
                rank_slot=1,
            )

    print("\n✅ Strategy scan complete!")


if __name__ == "__main__":
    main()

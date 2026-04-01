"""
engine.py — High-level BacktestEngine that orchestrates experiments.

Provides a single interface for running multi-strategy × multi-asset
experiments, with integrated visualization and export.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from synth.miner.strategies.registry import StrategyRegistry
from synth.miner.backtest.runner import BacktestRunner
from synth.miner.backtest.tuner import GridSearchTuner


@dataclass
class ExperimentConfig:
    """Configuration for a backtest experiment."""
    name: str = "experiment"
    assets: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL", "XAU"])
    frequencies: list[str] = field(default_factory=lambda: ["high", "low"])
    num_runs: int = 5
    num_sims: int = 100
    seed: int = 42
    window_days: int = 30
    metric: str = "CRPS"
    skip_ensemble: bool = True
    strategies: list[str] | None = None  # None = all discovered


class BacktestEngine:
    """
    High-level orchestrator for backtest experiments.

    Wraps BacktestRunner, GridSearchTuner, and visualization to provide
    a single experiment-oriented interface.

    Usage:
        engine = BacktestEngine()
        results = engine.run(ExperimentConfig(assets=["BTC"], num_runs=10))
        engine.export(results, "result/my_experiment")
        engine.visualize(results)
    """

    def __init__(self, metric: str = "CRPS"):
        self.registry = StrategyRegistry()
        self.registry.auto_discover()
        self.runner = BacktestRunner(metric=metric)
        self.tuner = GridSearchTuner(self.runner)

    def run(self, config: ExperimentConfig) -> list[dict]:
        """
        Run a full experiment: scan all compatible strategies × assets.

        Args:
            config: ExperimentConfig with experiment parameters.

        Returns:
            List of benchmark result dicts.
        """
        # If specific strategies requested, filter registry
        if config.strategies:
            filtered_registry = StrategyRegistry()
            for name in config.strategies:
                try:
                    strat = self.registry.get(name)
                    filtered_registry.register(strat)
                except KeyError:
                    print(f"[Engine] Strategy '{name}' not found, skipping")
            registry = filtered_registry
        else:
            registry = self.registry

        results = self.runner.scan_all(
            assets=config.assets,
            frequencies=config.frequencies,
            registry=registry,
            num_runs=config.num_runs,
            num_sims=config.num_sims,
            seed=config.seed,
            window_days=config.window_days,
            skip_ensemble=config.skip_ensemble,
        )

        return results

    def tune(
        self,
        strategy_name: str,
        asset: str,
        frequency: str = "low",
        param_grid: dict | None = None,
        num_runs: int = 3,
        num_sims: int = 100,
    ) -> dict:
        """
        Run hyperparameter tuning for a single strategy.

        Returns:
            Dict with best_params, best_score, all_results.
        """
        strategy = self.registry.get(strategy_name)
        return self.tuner.run(
            strategy=strategy,
            asset=asset,
            frequency=frequency,
            num_runs=num_runs,
            num_sims=num_sims,
            param_grid=param_grid,
        )

    def compare(self, results: list[dict]) -> dict:
        """
        Analyze and rank results, returning a summary.

        Returns:
            Dict with per-asset rankings and overall summary.
        """
        by_asset: dict[str, list[dict]] = {}
        for r in results:
            asset = r.get("asset", "?")
            if asset not in by_asset:
                by_asset[asset] = []
            by_asset[asset].append(r)

        rankings = {}
        for asset, asset_results in by_asset.items():
            sorted_r = sorted(
                asset_results,
                key=lambda x: x.get("avg_score", float("inf"))
            )
            rankings[asset] = [
                {
                    "rank": i + 1,
                    "strategy": r["strategy"],
                    "frequency": r.get("frequency", "?"),
                    "avg_score": r.get("avg_score", float("inf")),
                    "median_score": r.get("median_score", float("inf")),
                    "success_rate": f"{r.get('successful_runs', 0)}/{r.get('num_runs', 0)}",
                }
                for i, r in enumerate(sorted_r)
            ]

        return {"rankings": rankings, "total_combinations": len(results)}

    def export(
        self,
        results: list[dict],
        output_dir: str = "result/experiments",
    ) -> str:
        """Save experiment results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"experiment_{ts}.json")

        export_data = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "comparison": self.compare(results),
        }

        with open(path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"[Engine] Results exported to {path}")
        return path

    def export_to_production(
        self,
        results: list[dict],
        top_n: int = 3,
    ) -> None:
        """
        Export best results to production config.

        This is the key workflow: backtest → rank → deploy.
        """
        from synth.miner.deploy.exporter import export_best_config
        from synth.miner.deploy.applier import apply_to_miner

        config = export_best_config(results, top_n=top_n)
        apply_to_miner()

    def visualize(
        self,
        results: list[dict],
        output_dir: str = "result/charts",
    ) -> None:
        """Generate visualization charts from results."""
        os.makedirs(output_dir, exist_ok=True)

        try:
            from synth.miner.viz.strategy_compare import (
                plot_strategy_comparison,
                plot_score_distribution,
            )
            from synth.miner.viz.backtest_report import generate_html_report

            plot_strategy_comparison(
                results,
                output_path=os.path.join(output_dir, "strategy_comparison.png"),
            )
            plot_score_distribution(
                results,
                output_path=os.path.join(output_dir, "score_distribution.png"),
            )
            generate_html_report(results, output_dir=output_dir)

        except ImportError as e:
            print(f"[Engine] Visualization requires matplotlib: {e}")

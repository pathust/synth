"""
tuner.py — Hyperparameter tuning via Grid Search.

Extracted and enhanced from backtest_framework.py's GridSearch class.
"""

import itertools
import time
from datetime import datetime
from typing import Any

from synth.miner.strategies.base import BaseStrategy
from synth.miner.backtest.runner import BacktestRunner


class GridSearchTuner:
    """Performs grid search over a strategy's parameter space."""

    def __init__(self, runner: BacktestRunner):
        self.runner = runner

    def run(
        self,
        strategy: BaseStrategy,
        asset: str,
        frequency: str = "low",
        num_runs: int = 3,
        num_sims: int = 100,
        seed: int = 42,
        window_days: int = 30,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> dict:
        """
        Grid search over parameter combinations.
        
        Args:
            strategy: Strategy to tune
            asset: Target asset
            frequency: "high" or "low"
            num_runs: Benchmark runs per parameter combo
            num_sims: Simulations per run
            seed: Random seed
            window_days: Historical window for random dates
            param_grid: Override param grid; defaults to strategy's own grid
            
        Returns:
            dict with best_params, best_score, all_results
        """
        grid = param_grid or strategy.get_param_grid()
        if not grid:
            print(
                f"[Tuner] No param_grid for {strategy.name}, "
                f"running with defaults"
            )
            result = self.runner.run_benchmark(
                strategy, asset, frequency, num_runs, num_sims, seed,
                window_days,
            )
            return {
                "best_params": strategy.get_default_params(),
                "best_score": result["avg_score"],
                "all_results": [result],
            }

        keys, values = zip(*grid.items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(
            f"\n[Tuner] Grid search for {strategy.name} × {asset} × "
            f"{frequency}: {len(combos)} combinations"
        )

        best_score = float("inf")
        best_params = None
        all_results = []

        for idx, params in enumerate(combos):
            print(
                f"  [{idx + 1}/{len(combos)}] Params: {params}"
            )
            t0 = time.time()
            result = self.runner.run_benchmark(
                strategy, asset, frequency, num_runs, num_sims, seed,
                window_days, **params,
            )
            elapsed = time.time() - t0

            entry = {
                "params": params,
                "avg_score": result["avg_score"],
                "median_score": result["median_score"],
                "successful_runs": result["successful_runs"],
                "elapsed": round(elapsed, 2),
            }
            all_results.append(entry)

            if result["avg_score"] < best_score:
                best_score = result["avg_score"]
                best_params = params

            print(
                f"    → avg {self.runner.metric}: "
                f"{result['avg_score']:.4f} ({elapsed:.1f}s)"
            )

        print(
            f"\n[Tuner] Best params for {strategy.name} × {asset}: "
            f"{best_params} (score: {best_score:.4f})"
        )

        return {
            "strategy": strategy.name,
            "asset": asset,
            "frequency": frequency,
            "best_params": best_params,
            "best_score": best_score,
            "all_results": all_results,
        }

"""
tuner.py — Hyperparameter tuning via Grid Search.

Extracted and enhanced from backtest_framework.py's GridSearch class.
"""

import itertools
import time
from datetime import datetime
from typing import Any, Optional

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
        use_regimes: bool = False,
        dates: Optional[list[datetime]] = None,
        max_combinations: Optional[int] = None,
    ) -> dict:
        """
        Grid search over parameter combinations.
        If use_regimes=True, it finds best params per regime.
        
        Args:
            strategy: Strategy to tune
            asset: Target asset
            frequency: "high" or "low"
            num_runs: Benchmark runs per parameter combo
            num_sims: Simulations per run
            seed: Random seed
            window_days: Historical window for random dates
            param_grid: Override param grid; defaults to strategy's own grid
            use_regimes: If True, tune parameters per regime
            
        Returns:
            dict with best_params, best_score, all_results
        """
        if not use_regimes:
            effective_runs = len(dates) if dates else num_runs
            return self._run_grid_search(
                strategy,
                asset,
                frequency,
                dates,
                effective_runs,
                num_sims,
                seed,
                window_days,
                param_grid,
                max_combinations=max_combinations,
            )

        from datetime import timedelta, timezone
        end_date = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        ) - timedelta(days=2)
        start_date = end_date - timedelta(days=window_days)

        regime_dates = self.runner.get_regime_dates(
            asset, start_date, end_date, num_per_regime=num_runs, seed=seed
        )

        regime_results = {}
        for rtype, dates in regime_dates.items():
            if not dates:
                continue
            print(f"\n[Tuner] === Tuning cho Regime: {rtype.upper()} ({len(dates)} ngày) ===")
            res = self._run_grid_search(
                strategy,
                asset,
                frequency,
                dates,
                len(dates),
                num_sims,
                seed,
                window_days,
                param_grid,
                max_combinations=max_combinations,
            )
            regime_results[rtype] = res

        return {
            "strategy": strategy.name,
            "asset": asset,
            "use_regimes": True,
            "regime_results": regime_results
        }

    def _run_grid_search(
        self,
        strategy,
        asset,
        frequency,
        dates,
        num_runs,
        num_sims,
        seed,
        window_days,
        param_grid,
        max_combinations: Optional[int] = None,
    ):
        grid = param_grid or strategy.get_param_grid()
        if not grid:
            print(
                f"[Tuner] No param_grid for {strategy.name}, "
                f"running with defaults"
            )
            result = self.runner.run_benchmark(
                strategy, asset, frequency, num_runs, num_sims, seed,
                window_days, dates=dates,
            )
            return {
                "best_params": strategy.get_default_params(),
                "best_score": result["avg_score"],
                "all_results": [result],
            }

        keys, values = zip(*grid.items())
        combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
        if max_combinations and max_combinations > 0:
            combos = combos[:max_combinations]

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
                window_days, dates=dates, **params,
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

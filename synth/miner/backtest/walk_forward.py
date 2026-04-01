"""
walk_forward.py — Walk-forward validation to prevent overfitting.

Splits historical data into sequential train/test windows and
evaluates strategy performance on out-of-sample data.
"""

from __future__ import annotations

import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional

from synth.miner.strategies.base import BaseStrategy
from synth.miner.backtest.runner import BacktestRunner, _get_prompt_config


class WalkForwardValidator:
    """
    Walk-forward validation: train on window, validate on next period.

    This prevents overfitting by ensuring that parameter optimization
    is always evaluated on out-of-sample data.

    Usage:
        wf = WalkForwardValidator(runner)
        result = wf.run(strategy, "BTC", "low", train_days=30, test_days=7, n_folds=4)
    """

    def __init__(self, runner: BacktestRunner):
        self.runner = runner

    def run(
        self,
        strategy: BaseStrategy,
        asset: str,
        frequency: str = "low",
        train_days: int = 30,
        test_days: int = 7,
        n_folds: int = 4,
        num_sims: int = 100,
        seed: int = 42,
        num_runs_per_fold: int = 3,
    ) -> dict:
        """
        Run walk-forward validation.

        Creates n_folds sequential windows:
            Fold 1: train=[t0, t0+train_days], test=[t0+train_days, t0+train_days+test_days]
            Fold 2: shifted by test_days
            ...

        Args:
            strategy: Strategy to evaluate.
            asset: Asset symbol.
            frequency: "high" or "low".
            train_days: Training window size in days.
            test_days: Test window size in days.
            n_folds: Number of walk-forward folds.
            num_sims: Number of simulation paths.
            seed: Random seed.
            num_runs_per_fold: Number of random dates within each test window.

        Returns:
            Dict with fold results, aggregate metrics, and stability analysis.
        """
        # Calculate fold boundaries working backwards from "2 days ago"
        end_date = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        ) - timedelta(days=2)

        total_days_needed = train_days + test_days * n_folds
        earliest_start = end_date - timedelta(days=total_days_needed)

        fold_results = []
        train_scores = []
        test_scores = []

        for fold_idx in range(n_folds):
            fold_offset = fold_idx * test_days

            train_start = earliest_start + timedelta(days=fold_offset)
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)

            print(
                f"\n[WalkForward] Fold {fold_idx + 1}/{n_folds}: "
                f"Train [{train_start.strftime('%m/%d')} → {train_end.strftime('%m/%d')}] "
                f"Test [{test_start.strftime('%m/%d')} → {test_end.strftime('%m/%d')}]"
            )

            # Evaluate on train window
            train_result = self.runner.run_benchmark(
                strategy, asset, frequency,
                num_runs=num_runs_per_fold,
                num_sims=num_sims,
                seed=seed + fold_idx,
                window_days=train_days,
            )

            # Evaluate on test window  
            test_result = self.runner.run_benchmark(
                strategy, asset, frequency,
                num_runs=num_runs_per_fold,
                num_sims=num_sims,
                seed=seed + fold_idx + 1000,
                window_days=test_days,
            )

            fold_entry = {
                "fold": fold_idx + 1,
                "train_period": f"{train_start.isoformat()} → {train_end.isoformat()}",
                "test_period": f"{test_start.isoformat()} → {test_end.isoformat()}",
                "train_avg_score": train_result["avg_score"],
                "test_avg_score": test_result["avg_score"],
                "train_successful": train_result["successful_runs"],
                "test_successful": test_result["successful_runs"],
            }
            fold_results.append(fold_entry)

            if train_result["avg_score"] != float("inf"):
                train_scores.append(train_result["avg_score"])
            if test_result["avg_score"] != float("inf"):
                test_scores.append(test_result["avg_score"])

            print(
                f"  → Train: {train_result['avg_score']:.4f} | "
                f"Test: {test_result['avg_score']:.4f}"
            )

        # Stability analysis
        stability = self._analyze_stability(train_scores, test_scores)

        return {
            "strategy": strategy.name,
            "asset": asset,
            "frequency": frequency,
            "n_folds": n_folds,
            "train_days": train_days,
            "test_days": test_days,
            "folds": fold_results,
            "aggregate": {
                "mean_train_score": float(np.mean(train_scores)) if train_scores else float("inf"),
                "mean_test_score": float(np.mean(test_scores)) if test_scores else float("inf"),
                "std_test_score": float(np.std(test_scores)) if test_scores else float("inf"),
            },
            "stability": stability,
        }

    @staticmethod
    def _analyze_stability(
        train_scores: list[float],
        test_scores: list[float],
    ) -> dict:
        """Analyze train/test score stability for overfitting detection."""
        if not train_scores or not test_scores:
            return {"status": "INSUFFICIENT_DATA"}

        mean_train = np.mean(train_scores)
        mean_test = np.mean(test_scores)
        
        # Overfitting indicator: test significantly worse than train
        if mean_train > 0:
            degradation = (mean_test - mean_train) / mean_train * 100
        else:
            degradation = 0

        # Score variance across folds
        cv_test = np.std(test_scores) / np.mean(test_scores) * 100 if np.mean(test_scores) > 0 else 0

        if degradation > 50:
            status = "OVERFITTING_LIKELY"
        elif degradation > 20:
            status = "MODERATE_DEGRADATION"
        elif cv_test > 50:
            status = "HIGH_VARIANCE"
        else:
            status = "STABLE"

        return {
            "status": status,
            "train_test_degradation_pct": round(degradation, 2),
            "test_cv_pct": round(cv_test, 2),
        }

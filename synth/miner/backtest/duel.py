"""
duel.py — Head-to-head strategy comparison backtest.

Compares two strategies (baseline vs suggested) using the validator's
exact CRPS scoring pipeline on the same historical data points.

Usage:
    from synth.miner.backtest.duel import StrategyDuel
    from synth.miner.strategies import StrategyRegistry

    registry = StrategyRegistry()
    registry.auto_discover()

    duel = StrategyDuel(
        baseline=registry.get("garch_v2"),
        suggested=registry.get("jump_diffusion"),
    )
    result = duel.run(assets=["BTC"], frequency="high", num_runs=10)
    duel.print_report(result)
"""

import json
import os
import time
import traceback
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from synth.miner.strategies.base import BaseStrategy
from synth.miner.backtest.runner import get_random_dates, _get_prompt_config
from synth.miner.data_handler import DataHandler
from synth.miner.my_simulation import fetch_price_data
from synth.utils.helpers import convert_prices_to_time_format
from synth.simulation_input import SimulationInput
from synth.validator.crps_calculation import calculate_crps_for_miner
from synth.validator.response_validation_v2 import validate_responses
from synth.validator.price_data_provider import PriceDataProvider
from synth.db.models import ValidatorRequest


# ── Result containers ────────────────────────────────────────────────

@dataclass
class TurnResult:
    """Result for one turn (one time point)."""
    turn_idx: int
    asset: str
    start_time: str
    baseline_crps: float
    suggested_crps: float
    baseline_rmse: float
    suggested_rmse: float
    baseline_mae: float
    suggested_mae: float
    baseline_dir_acc: float  # 0-1, higher is better
    suggested_dir_acc: float
    baseline_var: float
    suggested_var: float
    baseline_es: float
    suggested_es: float
    # Per-interval CRPS breakdown
    baseline_interval_crps: dict = field(default_factory=dict)
    suggested_interval_crps: dict = field(default_factory=dict)
    # Mainnet comparison
    baseline_mainnet_rank: int = -1
    suggested_mainnet_rank: int = -1
    baseline_top1_gap: float = 0.0
    suggested_top1_gap: float = 0.0
    status: str = "SUCCESS"
    error: str = ""
    elapsed: float = 0.0


@dataclass
class DuelResult:
    """Aggregated duel results."""
    baseline_name: str
    suggested_name: str
    frequency: str
    assets: list
    num_runs: int
    num_sims: int
    turns: list  # list of TurnResult dicts
    # Aggregate metrics
    crps_win_rate: float = 0.0
    mean_crps_baseline: float = 0.0
    mean_crps_suggested: float = 0.0
    median_crps_baseline: float = 0.0
    median_crps_suggested: float = 0.0
    crps_improvement_pct: float = 0.0
    mean_rmse_baseline: float = 0.0
    mean_rmse_suggested: float = 0.0
    mean_mae_baseline: float = 0.0
    mean_mae_suggested: float = 0.0
    mean_dir_acc_baseline: float = 0.0
    mean_dir_acc_suggested: float = 0.0
    mean_var_baseline: float = 0.0
    mean_var_suggested: float = 0.0
    mean_var_suggested: float = 0.0
    mean_es_baseline: float = 0.0
    mean_es_suggested: float = 0.0
    # Mainnet comparison meta
    compare_mainnet: bool = False
    mean_rank_baseline: float = -1.0
    mean_rank_suggested: float = -1.0
    mean_top1_gap_baseline: float = 0.0
    mean_top1_gap_suggested: float = 0.0
    wilcoxon_p_value: float = 1.0
    interval_crps_comparison: dict = field(default_factory=dict)
    successful_turns: int = 0
    total_elapsed: float = 0.0


# ── Helper functions ─────────────────────────────────────────────────

def _compute_rmse(paths: np.ndarray, real_prices: np.ndarray) -> float:
    """RMSE of mean prediction path vs real prices."""
    mean_pred = np.mean(paths, axis=0)
    min_len = min(len(mean_pred), len(real_prices))
    if min_len == 0:
        return float("inf")
    return float(np.sqrt(np.mean((mean_pred[:min_len] - real_prices[:min_len]) ** 2)))


def _compute_mae(paths: np.ndarray, real_prices: np.ndarray) -> float:
    """MAE of mean prediction path vs real prices."""
    mean_pred = np.mean(paths, axis=0)
    min_len = min(len(mean_pred), len(real_prices))
    if min_len == 0:
        return float("inf")
    return float(np.mean(np.abs(mean_pred[:min_len] - real_prices[:min_len])))


def _compute_directional_accuracy(paths: np.ndarray, real_prices: np.ndarray) -> float:
    """Directional accuracy (0-1, higher = better)."""
    mean_pred = np.mean(paths, axis=0)
    min_len = min(len(mean_pred), len(real_prices))
    if min_len <= 1:
        return 0.0
    pred_diff = np.diff(mean_pred[:min_len])
    real_diff = np.diff(real_prices[:min_len])
    correct = np.sum(np.sign(pred_diff) == np.sign(real_diff))
    return float(correct / len(real_diff))

from synth.miner.backtest.metrics import compute_var, compute_es


def _extract_interval_crps(detailed_crps_data: list) -> dict:
    """Extract per-interval total CRPS from detailed data."""
    result = {}
    for entry in detailed_crps_data:
        if entry.get("Increment") == "Total" and entry.get("Interval") != "Overall":
            interval_name = entry["Interval"]
            crps_val = float(entry["CRPS"]) if not np.isnan(entry["CRPS"]) else float("inf")
            result[interval_name] = crps_val
    return result


def _get_real_prices(
    data_handler: DataHandler,
    asset: str,
    start_time: datetime,
    time_length: int,
    time_increment: int,
) -> list:
    """Get real prices from the price data provider (like the validator does)."""
    validator_request = ValidatorRequest(
        asset=asset,
        start_time=start_time,
        time_length=time_length,
        time_increment=time_increment,
    )
    return data_handler.get_real_prices(validator_request=validator_request)


# ── Main Duel Class ──────────────────────────────────────────────────

class StrategyDuel:
    """Head-to-head comparison of two strategies."""

    def __init__(
        self,
        baseline: BaseStrategy,
        suggested: BaseStrategy,
        compare_mainnet: bool = False,
    ):
        self.baseline = baseline
        self.suggested = suggested
        self.compare_mainnet = compare_mainnet
        self.data_handler = DataHandler()
        
        if self.compare_mainnet:
            from synth.miner.mysql_handler import MySQLHandler
            self.mysql = MySQLHandler()
        else:
            self.mysql = None

    def _run_strategy(
        self,
        strategy: BaseStrategy,
        prices_dict: dict,
        asset: str,
        time_increment: int,
        time_length: int,
        num_sims: int,
        seed: int,
    ) -> Optional[np.ndarray]:
        """Run a strategy and return paths array, or None on failure."""
        try:
            paths = strategy.simulate(
                prices_dict,
                asset=asset,
                time_increment=time_increment,
                time_length=time_length,
                n_sims=num_sims,
                seed=seed,
            )
            if paths is None or len(paths) == 0:
                return None
            return paths
        except Exception as e:
            print(f"    ⚠ {strategy.name} failed: {e}")
            traceback.print_exc()
            return None

    def _run_single_turn(
        self,
        turn_idx: int,
        asset: str,
        start_time: datetime,
        frequency: str,
        num_sims: int,
        seed: int,
    ) -> TurnResult:
        """Run one turn: both strategies on the same data point."""
        cfg = _get_prompt_config(frequency)
        time_increment = cfg.time_increment
        time_length = cfg.time_length
        scoring_intervals = cfg.scoring_intervals

        t0 = time.time()

        # 1. Load historical prices
        time_frame = "1m" if time_increment == 60 else "5m"
        hist_data = self.data_handler.load_price_data(asset, time_frame)

        if not hist_data or time_frame not in hist_data:
            return TurnResult(
                turn_idx=turn_idx, asset=asset,
                start_time=start_time.isoformat(),
                baseline_crps=float("inf"), suggested_crps=float("inf"),
                baseline_rmse=float("inf"), suggested_rmse=float("inf"),
                baseline_mae=float("inf"), suggested_mae=float("inf"),
                baseline_dir_acc=0.0, suggested_dir_acc=0.0,
                status="FAIL", error=f"No data for {asset}/{time_frame}",
                elapsed=round(time.time() - t0, 2),
            )

        prices_dict = hist_data[time_frame]

        # Only use prices BEFORE start_time
        start_ts = int(start_time.timestamp())
        filtered = {k: v for k, v in prices_dict.items() if int(k) < start_ts}
        if not filtered:
            return TurnResult(
                turn_idx=turn_idx, asset=asset,
                start_time=start_time.isoformat(),
                baseline_crps=float("inf"), suggested_crps=float("inf"),
                baseline_rmse=float("inf"), suggested_rmse=float("inf"),
                baseline_mae=float("inf"), suggested_mae=float("inf"),
                baseline_dir_acc=0.0, suggested_dir_acc=0.0,
                status="FAIL", error=f"No prices before {start_time}",
                elapsed=round(time.time() - t0, 2),
            )

        # 2. Get real prices (what actually happened)
        try:
            real_prices = _get_real_prices(
                self.data_handler, asset, start_time, time_length, time_increment
            )
        except Exception as e:
            return TurnResult(
                turn_idx=turn_idx, asset=asset,
                start_time=start_time.isoformat(),
                baseline_crps=float("inf"), suggested_crps=float("inf"),
                baseline_rmse=float("inf"), suggested_rmse=float("inf"),
                baseline_mae=float("inf"), suggested_mae=float("inf"),
                baseline_dir_acc=0.0, suggested_dir_acc=0.0,
                status="FAIL", error=f"Failed to get real prices: {e}",
                elapsed=round(time.time() - t0, 2),
            )

        if len(real_prices) == 0:
            return TurnResult(
                turn_idx=turn_idx, asset=asset,
                start_time=start_time.isoformat(),
                baseline_crps=float("inf"), suggested_crps=float("inf"),
                baseline_rmse=float("inf"), suggested_rmse=float("inf"),
                baseline_mae=float("inf"), suggested_mae=float("inf"),
                baseline_dir_acc=0.0, suggested_dir_acc=0.0,
                status="FAIL", error="Empty real prices",
                elapsed=round(time.time() - t0, 2),
            )

        real_prices_array = np.array(real_prices, dtype=np.float64)

        # 3. Run both strategies with SAME data
        baseline_paths = self._run_strategy(
            self.baseline, filtered, asset, time_increment, time_length, num_sims, seed
        )
        suggested_paths = self._run_strategy(
            self.suggested, filtered, asset, time_increment, time_length, num_sims, seed
        )

        # 4. Compute CRPS for both (using exact validator logic)
        def compute_crps(paths):
            if paths is None:
                return float("inf"), [], {}
            try:
                score, detailed = calculate_crps_for_miner(
                    paths, real_prices_array, time_increment, scoring_intervals
                )
                if np.isnan(score) or score == -1:
                    return float("inf"), detailed, {}
                interval_crps = _extract_interval_crps(detailed)
                return float(score), detailed, interval_crps
            except Exception as e:
                print(f"    ⚠ CRPS calculation failed: {e}")
                return float("inf"), [], {}

        b_crps, _, b_interval = compute_crps(baseline_paths)
        s_crps, _, s_interval = compute_crps(suggested_paths)

        # 5. Compute additional metrics
        def safe_metric(fn, paths):
            if paths is None:
                return float("inf") if fn != _compute_directional_accuracy else 0.0
            try:
                return fn(paths, real_prices_array)
            except Exception:
                return float("inf") if fn != _compute_directional_accuracy else 0.0

        # 6. Mainnet Rank Evaluation (if enabled)
        b_rank, s_rank = -1, -1
        b_gap, s_gap = 0.0, 0.0
        if self.compare_mainnet and self.mysql:
            st = start_time.strftime("%Y-%m-%d %H:%M:%S")
            scores = self.mysql.get_validation_scores(asset, st, st, time_length)
            if scores:
                mainnet_crps = [s["crps"] for s in scores if s["crps"] is not None]
                if mainnet_crps:
                    top1_crps = min(mainnet_crps)
                    
                    if b_crps != float("inf"):
                        # Rank is how many are better/equal + 1
                        b_rank = sum(1 for c in mainnet_crps if c <= b_crps) + 1
                        if top1_crps > 0:
                            b_gap = (b_crps - top1_crps) / top1_crps
                    
                    if s_crps != float("inf"):
                        s_rank = sum(1 for c in mainnet_crps if c <= s_crps) + 1
                        if top1_crps > 0:
                            s_gap = (s_crps - top1_crps) / top1_crps

        elapsed = round(time.time() - t0, 2)

        return TurnResult(
            turn_idx=turn_idx,
            asset=asset,
            start_time=start_time.isoformat(),
            baseline_crps=b_crps,
            suggested_crps=s_crps,
            baseline_rmse=safe_metric(_compute_rmse, baseline_paths),
            suggested_rmse=safe_metric(_compute_rmse, suggested_paths),
            baseline_mae=safe_metric(_compute_mae, baseline_paths),
            suggested_mae=safe_metric(_compute_mae, suggested_paths),
            baseline_dir_acc=safe_metric(_compute_directional_accuracy, baseline_paths),
            suggested_dir_acc=safe_metric(_compute_directional_accuracy, suggested_paths),
            baseline_var=safe_metric(compute_var, baseline_paths),
            suggested_var=safe_metric(compute_var, suggested_paths),
            baseline_es=safe_metric(compute_es, baseline_paths),
            suggested_es=safe_metric(compute_es, suggested_paths),
            baseline_interval_crps=b_interval,
            suggested_interval_crps=s_interval,
            baseline_mainnet_rank=b_rank,
            suggested_mainnet_rank=s_rank,
            baseline_top1_gap=b_gap,
            suggested_top1_gap=s_gap,
            status="SUCCESS",
            elapsed=elapsed,
        )

    def run(
        self,
        assets: list[str],
        frequency: str = "high",
        num_runs: int = 10,
        num_sims: int = 100,
        seed: int = 42,
        window_days: int = 30,
    ) -> DuelResult:
        """
        Run the full duel: N turns across given assets.

        Args:
            assets: List of assets to test
            frequency: "high" or "low"
            num_runs: Number of random time points per asset
            num_sims: Number of simulation paths
            seed: Random seed
            window_days: How far back to sample dates

        Returns:
            DuelResult with aggregated comparison metrics.
        """
        cfg = _get_prompt_config(frequency)
        total_t0 = time.time()

        # Generate random dates
        end_date = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        ) - timedelta(days=2)
        start_date = end_date - timedelta(days=window_days)

        all_turns = []
        turn_idx = 0

        for asset in assets:
            print(f"\n{'='*60}")
            print(f"  ASSET: {asset} | {frequency.upper()} frequency")
            print(f"{'='*60}")

            # Preload data
            fetch_price_data(asset, cfg.time_increment, only_load=True)

            dates = get_random_dates(start_date, end_date, num_runs, seed)

            for i, date_ in enumerate(dates):
                turn_idx += 1
                print(
                    f"  [{i+1}/{num_runs}] {date_.isoformat()} ... ",
                    end="", flush=True,
                )

                result = self._run_single_turn(
                    turn_idx, asset, date_, frequency, num_sims, seed + i
                )
                all_turns.append(result)

                if result.status == "SUCCESS":
                    winner = "BASELINE" if result.baseline_crps <= result.suggested_crps else "SUGGEST"
                    print(
                        f"B={result.baseline_crps:.2f} vs S={result.suggested_crps:.2f} "
                        f"→ {winner} ({result.elapsed}s)"
                    )
                else:
                    print(f"⚠ {result.error}")

        # ── Aggregate results ──
        successful = [t for t in all_turns if t.status == "SUCCESS"
                      and t.baseline_crps < float("inf")
                      and t.suggested_crps < float("inf")]

        duel_result = DuelResult(
            baseline_name=self.baseline.name,
            suggested_name=self.suggested.name,
            compare_mainnet=self.compare_mainnet,
            frequency=frequency,
            assets=assets,
            num_runs=num_runs,
            num_sims=num_sims,
            turns=[self._turn_to_dict(t) for t in all_turns],
            successful_turns=len(successful),
            total_elapsed=round(time.time() - total_t0, 2),
        )

        if not successful:
            print("\n⚠ No successful turns to aggregate!")
            return duel_result

        b_crps = np.array([t.baseline_crps for t in successful])
        s_crps = np.array([t.suggested_crps for t in successful])

        duel_result.crps_win_rate = float(np.mean(s_crps < b_crps))
        duel_result.mean_crps_baseline = float(np.mean(b_crps))
        duel_result.mean_crps_suggested = float(np.mean(s_crps))
        duel_result.median_crps_baseline = float(np.median(b_crps))
        duel_result.median_crps_suggested = float(np.median(s_crps))

        if duel_result.mean_crps_baseline > 0:
            duel_result.crps_improvement_pct = float(
                (duel_result.mean_crps_baseline - duel_result.mean_crps_suggested)
                / duel_result.mean_crps_baseline * 100
            )

        duel_result.mean_rmse_baseline = float(np.mean([t.baseline_rmse for t in successful]))
        duel_result.mean_rmse_suggested = float(np.mean([t.suggested_rmse for t in successful]))
        duel_result.mean_mae_baseline = float(np.mean([t.baseline_mae for t in successful]))
        duel_result.mean_mae_suggested = float(np.mean([t.suggested_mae for t in successful]))
        duel_result.mean_dir_acc_baseline = float(np.mean([t.baseline_dir_acc for t in successful]))
        duel_result.mean_dir_acc_suggested = float(np.mean([t.suggested_dir_acc for t in successful]))
        duel_result.mean_var_baseline = float(np.mean([t.baseline_var for t in successful]))
        duel_result.mean_var_suggested = float(np.mean([t.suggested_var for t in successful]))
        duel_result.mean_es_baseline = float(np.mean([t.baseline_es for t in successful]))
        duel_result.mean_es_suggested = float(np.mean([t.suggested_es for t in successful]))

        if self.compare_mainnet:
            ranked_turns = [t for t in successful if t.baseline_mainnet_rank != -1]
            if ranked_turns:
                duel_result.mean_rank_baseline = float(np.mean([t.baseline_mainnet_rank for t in ranked_turns]))
                duel_result.mean_rank_suggested = float(np.mean([t.suggested_mainnet_rank for t in ranked_turns]))
                duel_result.mean_top1_gap_baseline = float(np.mean([t.baseline_top1_gap for t in ranked_turns]))
                duel_result.mean_top1_gap_suggested = float(np.mean([t.suggested_top1_gap for t in ranked_turns]))

        # Wilcoxon signed-rank test
        try:
            from scipy.stats import wilcoxon
            diff = b_crps - s_crps
            if np.any(diff != 0) and len(diff) >= 6:
                _, p_val = wilcoxon(diff)
                duel_result.wilcoxon_p_value = float(p_val)
        except ImportError:
            pass  # scipy not available
        except Exception:
            pass

        # Per-interval comparison
        all_intervals = set()
        for t in successful:
            all_intervals.update(t.baseline_interval_crps.keys())
            all_intervals.update(t.suggested_interval_crps.keys())

        interval_comparison = {}
        for interval in sorted(all_intervals):
            b_vals = [t.baseline_interval_crps.get(interval, float("inf")) for t in successful
                      if interval in t.baseline_interval_crps]
            s_vals = [t.suggested_interval_crps.get(interval, float("inf")) for t in successful
                      if interval in t.suggested_interval_crps]
            if b_vals and s_vals:
                interval_comparison[interval] = {
                    "baseline_mean": float(np.mean(b_vals)),
                    "suggested_mean": float(np.mean(s_vals)),
                    "suggested_wins": int(sum(s < b for s, b in zip(s_vals, b_vals))),
                    "total": min(len(b_vals), len(s_vals)),
                }
        duel_result.interval_crps_comparison = interval_comparison

        return duel_result

    @staticmethod
    def _turn_to_dict(turn: TurnResult) -> dict:
        return {
            "turn_idx": turn.turn_idx,
            "asset": turn.asset,
            "start_time": turn.start_time,
            "baseline_crps": turn.baseline_crps,
            "suggested_crps": turn.suggested_crps,
            "baseline_rmse": turn.baseline_rmse,
            "suggested_rmse": turn.suggested_rmse,
            "baseline_mae": turn.baseline_mae,
            "suggested_mae": turn.suggested_mae,
            "baseline_dir_acc": turn.baseline_dir_acc,
            "suggested_dir_acc": turn.suggested_dir_acc,
            "baseline_var": turn.baseline_var,
            "suggested_var": turn.suggested_var,
            "baseline_es": turn.baseline_es,
            "suggested_es": turn.suggested_es,
            "status": turn.status,
            "error": turn.error,
            "elapsed": turn.elapsed,
        }

    @staticmethod
    def print_report(result: DuelResult) -> None:
        """Print a formatted comparison report."""
        print(f"\n{'='*70}")
        print(f"  DUEL REPORT: {result.baseline_name} vs {result.suggested_name}")
        print(f"{'='*70}")
        print(f"  Frequency: {result.frequency} | Assets: {result.assets}")
        print(f"  Turns: {result.successful_turns}/{len(result.turns)} successful")
        print(f"  Sims per turn: {result.num_sims} | Time: {result.total_elapsed}s")

        if result.successful_turns == 0:
            print("\n  ⚠ No successful turns — cannot generate report.\n")
            return

        # ── Headline: Who wins? ──
        print(f"\n{'─'*70}")
        if result.crps_improvement_pct > 0:
            print(f"  🏆 {result.suggested_name} beats {result.baseline_name} "
                  f"by {result.crps_improvement_pct:+.2f}% CRPS")
        elif result.crps_improvement_pct < 0:
            print(f"  ❌ {result.suggested_name} is worse than {result.baseline_name} "
                  f"by {result.crps_improvement_pct:.2f}% CRPS")
        else:
            print(f"  🤝 Dead heat — both strategies have equal CRPS")
        print(f"{'─'*70}")

        # ── CRPS comparison table ──
        print(f"\n  {'Metric':<30} {'Baseline':>12} {'Suggested':>12} {'Winner':>10}")
        print(f"  {'─'*64}")

        rows = [
            ("CRPS Win Rate", f"—",
             f"{result.crps_win_rate:.1%}",
             "SUGGEST" if result.crps_win_rate > 0.5 else "BASELINE"),
            ("Mean CRPS ↓", f"{result.mean_crps_baseline:.2f}",
             f"{result.mean_crps_suggested:.2f}",
             "SUGGEST" if result.mean_crps_suggested < result.mean_crps_baseline else "BASELINE"),
            ("Median CRPS ↓", f"{result.median_crps_baseline:.2f}",
             f"{result.median_crps_suggested:.2f}",
             "SUGGEST" if result.median_crps_suggested < result.median_crps_baseline else "BASELINE"),
            ("CRPS Improvement", "—",
             f"{result.crps_improvement_pct:+.2f}%",
             "SUGGEST" if result.crps_improvement_pct > 0 else "BASELINE"),
            ("Mean RMSE ↓", f"{result.mean_rmse_baseline:.2f}",
             f"{result.mean_rmse_suggested:.2f}",
             "SUGGEST" if result.mean_rmse_suggested < result.mean_rmse_baseline else "BASELINE"),
            ("Mean MAE ↓", f"{result.mean_mae_baseline:.2f}",
             f"{result.mean_mae_suggested:.2f}",
             "SUGGEST" if result.mean_mae_suggested < result.mean_mae_baseline else "BASELINE"),
            ("Mean VaR (95%) ↑", f"{result.mean_var_baseline:.2%}",
             f"{result.mean_var_suggested:.2%}",
             "SUGGEST" if result.mean_var_suggested > result.mean_var_baseline else "BASELINE"),
            ("Mean ES (95%) ↑", f"{result.mean_es_baseline:.2%}",
             f"{result.mean_es_suggested:.2%}",
             "SUGGEST" if result.mean_es_suggested > result.mean_es_baseline else "BASELINE"),
            ("Dir. Accuracy ↑", f"{result.mean_dir_acc_baseline:.1%}",
             f"{result.mean_dir_acc_suggested:.1%}",
             "SUGGEST" if result.mean_dir_acc_suggested > result.mean_dir_acc_baseline else "BASELINE"),
            ("Wilcoxon p-value", "—",
             f"{result.wilcoxon_p_value:.4f}",
             "SIG ✓" if result.wilcoxon_p_value < 0.05 else "NOT SIG"),
        ]

        for label, b_val, s_val, winner in rows:
            print(f"  {label:<30} {b_val:>12} {s_val:>12} {winner:>10}")

        if getattr(result, 'compare_mainnet', False):
            print(f"\n  {'─'*64}")
            print(f"  Mainnet Integration Results (Rank against historical network scores):")
            print(f"  {'Metric':<30} {'Baseline':>12} {'Suggested':>12} {'Winner':>10}")
            print(f"  {'─'*64}")
            mr_rows = [
                ("Mean Mainnet Rank ↓", f"#{result.mean_rank_baseline:.1f}", f"#{result.mean_rank_suggested:.1f}", 
                "SUGGEST" if result.mean_rank_suggested < result.mean_rank_baseline else "BASELINE"),
                ("Average CRPS Gap to Top #1 ↓", f"{result.mean_top1_gap_baseline:+.2%}", f"{result.mean_top1_gap_suggested:+.2%}",
                 "SUGGEST" if result.mean_top1_gap_suggested < result.mean_top1_gap_baseline else "BASELINE"),
            ]
            for label, b_val, s_val, winner in mr_rows:
                print(f"  {label:<30} {b_val:>12} {s_val:>12} {winner:>10}")

        # ── Per-interval breakdown ──
        if result.interval_crps_comparison:
            print(f"\n  {'─'*64}")
            print(f"  Per-Interval CRPS Breakdown:")
            print(f"  {'Interval':<20} {'Baseline':>12} {'Suggested':>12} {'Wins':>10}")
            print(f"  {'─'*54}")
            for interval, data in result.interval_crps_comparison.items():
                wins = f"{data['suggested_wins']}/{data['total']}"
                winner_mark = "✓" if data['suggested_wins'] > data['total'] / 2 else ""
                print(
                    f"  {interval:<20} {data['baseline_mean']:>12.2f} "
                    f"{data['suggested_mean']:>12.2f} {wins:>8} {winner_mark}"
                )

        # ── Per-turn detail ──
        print(f"\n  {'─'*64}")
        print(f"  Per-Turn Results:")
        print(f"  {'#':<4} {'Asset':<8} {'Time':<22} {'B_CRPS':>10} {'S_CRPS':>10} {'Δ':>10}")
        print(f"  {'─'*64}")
        for t in result.turns:
            if t["status"] != "SUCCESS":
                print(f"  {t['turn_idx']:<4} {t['asset']:<8} {t['start_time'][:19]:<22} {'FAIL':>30}")
                continue
            b = t["baseline_crps"]
            s = t["suggested_crps"]
            if b == float("inf") or s == float("inf"):
                continue
            delta = b - s  # positive = suggested is better
            marker = "✓" if delta > 0 else ""
            print(
                f"  {t['turn_idx']:<4} {t['asset']:<8} {t['start_time'][:19]:<22} "
                f"{b:>10.2f} {s:>10.2f} {delta:>+10.2f} {marker}"
            )

        print(f"\n{'='*70}\n")

    @staticmethod
    def export_json(result: DuelResult, result_dir: str = "result") -> str:
        """Export duel result to JSON file."""
        os.makedirs(result_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"duel_{result.baseline_name}_vs_{result.suggested_name}_{ts}.json"
        filepath = os.path.join(result_dir, filename)

        export = {
            "timestamp": datetime.now().isoformat(),
            "baseline": result.baseline_name,
            "suggested": result.suggested_name,
            "frequency": result.frequency,
            "assets": result.assets,
            "num_runs": result.num_runs,
            "num_sims": result.num_sims,
            "successful_turns": result.successful_turns,
            "total_elapsed": result.total_elapsed,
            "summary": {
                "crps_win_rate": result.crps_win_rate,
                "mean_crps_baseline": result.mean_crps_baseline,
                "mean_crps_suggested": result.mean_crps_suggested,
                "median_crps_baseline": result.median_crps_baseline,
                "median_crps_suggested": result.median_crps_suggested,
                "crps_improvement_pct": result.crps_improvement_pct,
                "mean_rmse_baseline": result.mean_rmse_baseline,
                "mean_rmse_suggested": result.mean_rmse_suggested,
                "mean_mae_baseline": result.mean_mae_baseline,
                "mean_mae_suggested": result.mean_mae_suggested,
                "mean_dir_acc_baseline": result.mean_dir_acc_baseline,
                "mean_dir_acc_suggested": result.mean_dir_acc_suggested,
                "mean_var_baseline": result.mean_var_baseline,
                "mean_var_suggested": result.mean_var_suggested,
                "mean_es_baseline": result.mean_es_baseline,
                "mean_es_suggested": result.mean_es_suggested,
                "wilcoxon_p_value": result.wilcoxon_p_value,
            },
            "interval_crps_comparison": result.interval_crps_comparison,
            "turns": result.turns,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, default=str)

        print(f"[Duel] Results exported to: {filepath}")
        return filepath

"""
runner.py — Backtest runner that evaluates strategies against historical data.

Uses the strategy registry to auto-discover strategies and runs them
through the validator's CRPS scoring pipeline.
"""

import time
import traceback
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional

from synth.miner.strategies import StrategyRegistry, BaseStrategy
from synth.miner.backtest.metrics import METRICS
from synth.miner.data_handler import DataHandler
from synth.miner.my_simulation import fetch_price_data
from synth.miner.compute_score import cal_reward
from synth.db.models import ValidatorRequest
from synth.miner.regime import get_random_dates, scan_regime_dates
from synth.simulation_input import SimulationInput
from synth.utils.helpers import convert_prices_to_time_format
from synth.validator.response_validation_v2 import validate_responses
from synth.validator import prompt_config


def _get_prompt_config(frequency: str):
    """Return the appropriate PromptConfig for the frequency label."""
    if frequency == "high":
        return prompt_config.HIGH_FREQUENCY
    return prompt_config.LOW_FREQUENCY


class BacktestRunner:
    """Evaluates strategies using CRPS or other metrics."""

    def __init__(self, metric: str = "CRPS"):
        self.data_handler = DataHandler()
        self.metric = metric.upper()
        if self.metric not in METRICS:
            raise ValueError(
                f"Unknown metric '{metric}'. Available: {list(METRICS.keys())}"
            )

    def run_single(
        self,
        strategy: BaseStrategy,
        asset: str,
        start_time: datetime,
        frequency: str = "low",
        num_sims: int = 100,
        seed: int = 42,
        **strategy_kwargs,
    ) -> dict:
        """
        Evaluate a single strategy on one date.
        
        Returns dict with keys: status, score, metric, elapsed, etc.
        """
        cfg = _get_prompt_config(frequency)
        time_increment = cfg.time_increment
        time_length = cfg.time_length

        t0 = time.time()
        try:
            sim_input = SimulationInput(
                asset=asset,
                start_time=start_time.isoformat(),
                time_increment=time_increment,
                time_length=time_length,
                num_simulations=num_sims,
            )

            # Fetch data for this asset (only load from DB)
            hist_data = fetch_price_data(asset, time_increment, only_load=True)

            # Get historical prices dict
            time_frame = "1m" if time_increment == 60 else "5m"
            if not hist_data or time_frame not in hist_data:
                # Try loading directly if fetch_price_data returned nothing
                hist_data = self.data_handler.load_price_data(asset, time_frame)

            if not hist_data or time_frame not in hist_data:
                return {
                    "status": "FAIL",
                    "error": f"No data for {asset}/{time_frame}",
                    "elapsed": round(time.time() - t0, 2),
                    "score": float("inf"),
                }

            prices_dict = hist_data[time_frame]

            # Filter: only use prices BEFORE start_time
            start_ts = int(start_time.timestamp())
            filtered = {
                k: v for k, v in prices_dict.items() if int(k) < start_ts
            }
            if not filtered:
                return {
                    "status": "FAIL",
                    "error": f"No prices before {start_time}",
                    "elapsed": round(time.time() - t0, 2),
                    "score": float("inf"),
                }

            # Run strategy
            paths = strategy.simulate(
                filtered,
                asset=asset,
                time_increment=time_increment,
                time_length=time_length,
                n_sims=num_sims,
                seed=seed,
                **strategy_kwargs,
            )

            if paths is None or len(paths) == 0:
                return {
                    "status": "FAIL",
                    "error": "No paths generated",
                    "elapsed": round(time.time() - t0, 2),
                    "score": float("inf"),
                }

            # Convert to prediction format
            predictions = convert_prices_to_time_format(
                paths.tolist(), start_time.isoformat(), time_increment
            )
            is_valid = (
                validate_responses(predictions, sim_input, "0") == "CORRECT"
            )

            # Compute CRPS via cal_reward
            validator_request = ValidatorRequest(
                asset=asset,
                start_time=start_time,
                time_length=time_length,
                time_increment=time_increment,
            )
            crps_score, _, _, real_prices = cal_reward(
                self.data_handler, validator_request, predictions
            )

            if self.metric == "CRPS":
                score = crps_score if crps_score != -1 else float("inf")
            else:
                score = METRICS[self.metric](paths, np.array(real_prices))

            return {
                "status": "SUCCESS",
                "score": float(score),
                "metric": self.metric,
                "strategy": strategy.name,
                "asset": asset,
                "frequency": frequency,
                "format_valid": is_valid,
                "num_paths": int(paths.shape[0]),
                "start_time": start_time.isoformat(),
                "elapsed": round(time.time() - t0, 2),
                "kwargs": strategy_kwargs,
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "status": "ERROR",
                "error": str(e),
                "strategy": strategy.name,
                "asset": asset,
                "elapsed": round(time.time() - t0, 2),
                "score": float("inf"),
            }

    def get_regime_dates(
        self,
        asset: str,
        start_date: datetime,
        end_date: datetime,
        num_per_regime: int = 5,
        pool_size: int = 150,
        seed: int = 42,
    ) -> dict[str, list[datetime]]:
        """
        Tìm và phân loại các ngày backtest vào 3 tập regime: bullish, bearish, neutral.
        Sử dụng pattern_detector_v2 để có độ chính xác cao hơn regime_detection_er.
        """
        # 1. Load dữ liệu 1m để thực hiện detect pattern
        hist_data = self.data_handler.load_price_data(asset, "1m")
        if not hist_data or "1m" not in hist_data:
            print(f"  ⚠ Không có dữ liệu 1m cho {asset}, không thể xác định regime.")
            return {"bullish": [], "bearish": [], "neutral": []}
        return scan_regime_dates(
            prices_dict=hist_data["1m"],
            start_date=start_date,
            end_date=end_date,
            num_per_regime=num_per_regime,
            pool_size=pool_size,
            seed=seed,
        )

    def run_benchmark(
        self,
        strategy: BaseStrategy,
        asset: str,
        frequency: str = "low",
        num_runs: int = 5,
        num_sims: int = 100,
        seed: int = 42,
        window_days: int = 30,
        dates: Optional[list[datetime]] = None,
        **strategy_kwargs,
    ) -> dict:
        """
        Run multiple backtests over random dates or specific dates and return aggregate results.
        """
        if dates is None:
            end_date = datetime.now(timezone.utc).replace(
                minute=0, second=0, microsecond=0
            ) - timedelta(days=2)
            start_date = end_date - timedelta(days=window_days)
            dates = get_random_dates(start_date, end_date, num_runs, seed)
        else:
            num_runs = len(dates)

        scores = []
        results = []
        for idx, date_ in enumerate(dates):
            print(
                f"  [{idx + 1}/{num_runs}] {strategy.name} / {asset} / "
                f"{date_.isoformat()}"
            )
            res = self.run_single(
                strategy, asset, date_, frequency, num_sims, seed + idx,
                **strategy_kwargs,
            )
            results.append(res)
            if res["status"] == "SUCCESS":
                scores.append(res["score"])
            else:
                print(f"    ⚠ {res.get('error', 'unknown error')}")

        avg_score = float(np.mean(scores)) if scores else float("inf")
        median_score = float(np.median(scores)) if scores else float("inf")

        return {
            "strategy": strategy.name,
            "asset": asset,
            "frequency": frequency,
            "metric": self.metric,
            "num_runs": num_runs,
            "successful_runs": len(scores),
            "avg_score": avg_score,
            "median_score": median_score,
            "min_score": float(min(scores)) if scores else float("inf"),
            "max_score": float(max(scores)) if scores else float("inf"),
            "all_scores": scores,
            "details": results,
        }

    def scan_all(
        self,
        assets: list[str],
        frequencies: list[str],
        registry: StrategyRegistry,
        num_runs: int = 5,
        num_sims: int = 100,
        seed: int = 42,
        window_days: int = 30,
        skip_ensemble: bool = True,
    ) -> list[dict]:
        """
        Scan all compatible strategy × asset × frequency combinations.

        Returns list of benchmark result dicts.
        """
        all_results = []
        total_combos = 0

        for asset in assets:
            for freq in frequencies:
                strategies = registry.get_for_asset(asset)
                if skip_ensemble:
                    strategies = [
                        s for s in strategies
                        if s.name != "ensemble_weighted"
                    ]
                for strategy in strategies:
                    if strategy.supports_frequency(freq):
                        total_combos += 1

        print(
            f"\n{'='*60}\n"
            f"STRATEGY SCAN: {len(assets)} assets × {len(frequencies)} freqs "
            f"= {total_combos} combinations\n"
            f"{'='*60}"
        )

        combo_idx = 0
        for asset in assets:
            for freq in frequencies:
                strategies = registry.get_for_asset(asset)
                if skip_ensemble:
                    strategies = [
                        s for s in strategies
                        if s.name != "ensemble_weighted"
                    ]

                cfg = _get_prompt_config(freq)
                fetch_price_data(asset, cfg.time_increment, only_load=True)

                for strategy in strategies:
                    if not strategy.supports_frequency(freq):
                        continue
                    combo_idx += 1
                    print(
                        f"\n[{combo_idx}/{total_combos}] "
                        f"{strategy.name} × {asset} × {freq}"
                    )
                    result = self.run_benchmark(
                        strategy,
                        asset,
                        freq,
                        num_runs=num_runs,
                        num_sims=num_sims,
                        seed=seed,
                        window_days=window_days,
                    )
                    all_results.append(result)
                    print(
                        f"  → avg {self.metric}: {result['avg_score']:.4f} "
                        f"({result['successful_runs']}/{num_runs} OK)"
                    )

        return all_results

"""
report.py — Results aggregation and reporting.

Generates summary tables and JSON exports from backtest results.
"""

import json
import os
from datetime import datetime
from typing import Optional


class BacktestReport:
    """Aggregates backtest results and exports reports."""

    def __init__(self, result_dir: str = "result"):
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)

    def generate_rankings(self, scan_results: list[dict]) -> dict:
        """
        Generate per-asset and per-frequency strategy rankings.
        
        Args:
            scan_results: List of benchmark result dicts from runner.scan_all()
            
        Returns:
            dict with rankings
        """
        # ── Per-asset best strategy ──
        per_asset = {}
        for r in scan_results:
            asset = r["asset"]
            freq = r["frequency"]
            key = f"{asset}_{freq}"

            if key not in per_asset or r["avg_score"] < per_asset[key]["avg_score"]:
                per_asset[key] = {
                    "asset": asset,
                    "frequency": freq,
                    "best_strategy": r["strategy"],
                    "avg_score": r["avg_score"],
                    "median_score": r["median_score"],
                    "successful_runs": r["successful_runs"],
                    "num_runs": r["num_runs"],
                }

        # ── Overall best strategy per frequency ──
        per_freq = {}
        for r in scan_results:
            freq = r["frequency"]
            if freq not in per_freq:
                per_freq[freq] = {}
            strat = r["strategy"]
            if strat not in per_freq[freq]:
                per_freq[freq][strat] = []
            if r["avg_score"] < float("inf"):
                per_freq[freq][strat].append(r["avg_score"])

        # Average across assets for each strategy
        freq_rankings = {}
        for freq, strategies in per_freq.items():
            ranked = []
            for strat, scores in strategies.items():
                if scores:
                    avg = sum(scores) / len(scores)
                    ranked.append({"strategy": strat, "avg_score_across_assets": avg, "num_assets": len(scores)})
            ranked.sort(key=lambda x: x["avg_score_across_assets"])
            freq_rankings[freq] = ranked

        return {
            "per_asset_best": per_asset,
            "per_frequency_rankings": freq_rankings,
        }

    def print_summary(self, scan_results: list[dict]) -> None:
        """Print a summary table to console."""
        rankings = self.generate_rankings(scan_results)

        print("\n" + "=" * 70)
        print("BEST STRATEGY PER ASSET × FREQUENCY")
        print("=" * 70)
        print(f"{'Asset':<10} {'Freq':<6} {'Strategy':<25} {'Avg CRPS':<12} {'Runs'}")
        print("-" * 70)
        for key in sorted(rankings["per_asset_best"].keys()):
            r = rankings["per_asset_best"][key]
            print(
                f"{r['asset']:<10} {r['frequency']:<6} "
                f"{r['best_strategy']:<25} "
                f"{r['avg_score']:<12.4f} "
                f"{r['successful_runs']}/{r['num_runs']}"
            )

        for freq, ranked in rankings["per_frequency_rankings"].items():
            print(f"\n{'='*50}")
            print(f"OVERALL STRATEGY RANKING — {freq.upper()} frequency")
            print(f"{'='*50}")
            print(f"{'Rank':<6} {'Strategy':<25} {'Avg CRPS':<12} {'Assets'}")
            print("-" * 50)
            for i, r in enumerate(ranked):
                print(
                    f"{i+1:<6} {r['strategy']:<25} "
                    f"{r['avg_score_across_assets']:<12.4f} "
                    f"{r['num_assets']}"
                )

    def export_json(
        self,
        scan_results: list[dict],
        tuning_results: Optional[list[dict]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Export full results to JSON file.
        
        Returns the path to the exported file.
        """
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_scan_{ts}.json"

        filepath = os.path.join(self.result_dir, filename)

        rankings = self.generate_rankings(scan_results)

        # Clean scan results (remove heavy 'details' for the summary export)
        clean_results = []
        for r in scan_results:
            clean = {k: v for k, v in r.items() if k != "details"}
            clean_results.append(clean)

        export = {
            "timestamp": datetime.now().isoformat(),
            "rankings": rankings,
            "scan_results": clean_results,
        }
        if tuning_results:
            export["tuning_results"] = tuning_results

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, default=str)

        print(f"\n[Report] Results exported to: {filepath}")
        return filepath

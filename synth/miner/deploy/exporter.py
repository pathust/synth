"""
exporter.py — Export best backtest results to production config.

Reads backtest scan results (from BacktestRunner.scan_all() or
GridSearchTuner), ranks strategies per (asset, frequency),
and generates strategies.yaml for runtime hot-reload.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import yaml

from synth.miner.strategies.base import StrategyConfig


_DEFAULT_OUTPUT = "synth/miner/config/strategies.yaml"
_DEFAULT_FALLBACK = {
    "L1_timeout_ms": 600,
    "L2_model": "garch_v2",
    "L3_model": "garch_v4",
}


def export_best_config(
    scan_results: list[dict],
    top_n: int = 3,
    output_path: str = _DEFAULT_OUTPUT,
    backup: bool = True,
    min_successful_runs: int = 2,
) -> dict[tuple[str, str], list[StrategyConfig]]:
    """
    From backtest scan results, pick top N strategies per (asset, frequency),
    compute weights proportional to inverse score, and write the config file.

    Args:
        scan_results: List of benchmark result dicts from BacktestRunner.scan_all()
            Each dict must have: strategy, asset, frequency, avg_score, successful_runs
        top_n: Number of top strategies per (asset, freq) to include.
        output_path: Where to write the generated config module.
        backup: If True, backup existing config before overwriting.
        min_successful_runs: Minimum successful runs to consider a result valid.

    Returns:
        The generated config dict.
    """
    # Group results by (asset, frequency)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for result in scan_results:
        if result.get("successful_runs", 0) < min_successful_runs:
            continue
        if result.get("avg_score", float("inf")) == float("inf"):
            continue

        key = (result["asset"], result["frequency"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    # Rank and select top N per group
    config: dict[tuple[str, str], list[StrategyConfig]] = {}

    for key, results in grouped.items():
        # Sort by avg_score (lower is better)
        sorted_results = sorted(results, key=lambda r: r["avg_score"])
        top_results = sorted_results[:top_n]

        if not top_results:
            continue

        # Compute weights: inverse of score, normalized
        scores = [r["avg_score"] for r in top_results]
        inverse_scores = [1.0 / max(s, 1e-6) for s in scores]
        total_inv = sum(inverse_scores)

        strategy_configs = []
        for result, inv_score in zip(top_results, inverse_scores):
            weight = round(inv_score / total_inv, 2)
            sc = StrategyConfig(
                strategy_name=result["strategy"],
                weight=weight,
                params=result.get("kwargs", {}),
            )
            strategy_configs.append(sc)

        config[key] = strategy_configs

    if output_path:
        _write_config_yaml(config, output_path, backup)

    _save_audit_json(config, scan_results, output_path)

    return config


def _write_config_yaml(
    config: dict[tuple[str, str], list[StrategyConfig]],
    output_path: str,
    backup: bool,
) -> None:
    payload = {
        "version": "2.0",
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "fallback_chain": dict(_DEFAULT_FALLBACK),
        "routing": {},
    }
    for (asset, frequency), strategies in sorted(config.items()):
        if asset not in payload["routing"]:
            payload["routing"][asset] = {}
        payload["routing"][asset][frequency] = {
            "ensemble_method": "weighted_average",
            "models": [
                {
                    "name": sc.strategy_name,
                    "weight": float(sc.weight),
                    "params": sc.params or {},
                }
                for sc in strategies
            ],
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if backup and os.path.exists(output_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{output_path}.bak.{ts}"
        os.rename(output_path, backup_path)
        print(f"[Export] Backed up existing config to {backup_path}")
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    with open(tmp_path, "r", encoding="utf-8") as f:
        yaml.safe_load(f)
    os.replace(tmp_path, output_path)
    print(f"[Export] Wrote YAML config to {output_path} ({len(config)} entries)")


def _save_audit_json(
    config: dict[tuple[str, str], list[StrategyConfig]],
    scan_results: list[dict],
    output_path: str,
) -> None:
    """Save raw JSON audit trail alongside the config."""
    audit = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            f"{k[0]}_{k[1]}": [
                {"name": sc.strategy_name, "weight": sc.weight, "params": sc.params}
                for sc in v
            ]
            for k, v in config.items()
        },
        "top_results_summary": [
            {
                "strategy": r.get("strategy"),
                "asset": r.get("asset"),
                "frequency": r.get("frequency"),
                "avg_score": r.get("avg_score"),
                "median_score": r.get("median_score"),
                "successful_runs": r.get("successful_runs"),
            }
            for r in scan_results
            if r.get("avg_score", float("inf")) != float("inf")
        ],
    }

    base, _ = os.path.splitext(output_path)
    audit_path = f"{base}_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"[Export] Audit trail saved to {audit_path}")

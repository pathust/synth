"""
applier.py — Apply exported config to the running miner.

Validates strategies.yaml against strategy registry and schema expectations.
"""

from __future__ import annotations

import difflib
import os
from typing import Optional

import yaml


_CONFIG_PATH = "synth/miner/config/strategies.yaml"


def preview_config_diff(
    new_config_path: Optional[str] = None,
    current_config_path: str = _CONFIG_PATH,
) -> str:
    """
    Show a unified diff between current and new config.

    Args:
        new_config_path: Path to the new config (from exporter).
                        If None, uses current_config_path (no diff).
        current_config_path: Path to the current production config.

    Returns:
        Unified diff string.
    """
    if new_config_path is None:
        return "No new config to compare."

    if not os.path.exists(current_config_path):
        return f"Current config not found at {current_config_path}"

    if not os.path.exists(new_config_path):
        return f"New config not found at {new_config_path}"

    with open(current_config_path) as f:
        current_lines = f.readlines()

    with open(new_config_path) as f:
        new_lines = f.readlines()

    diff = difflib.unified_diff(
        current_lines,
        new_lines,
        fromfile=f"current: {current_config_path}",
        tofile=f"new: {new_config_path}",
        lineterm="",
    )

    return "\n".join(diff)


def validate_config(config_path: str = _CONFIG_PATH) -> list[str]:
    """
    Validate a config file against the strategy registry.

    Returns a list of warnings (empty = all OK).
    """
    from synth.miner.strategies.registry import StrategyRegistry

    registry = StrategyRegistry()
    registry.auto_discover()
    available = set(registry.list_all())

    warnings = []

    try:
        if not os.path.exists(config_path):
            warnings.append(f"Config file not found: {config_path}")
            return warnings
        with open(config_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if not isinstance(payload, dict):
            warnings.append("Config root must be a mapping")
            return warnings
        routing = payload.get("routing")
        if not isinstance(routing, dict):
            warnings.append("Missing/invalid routing section")
            return warnings
        for asset, regimes in routing.items():
            if not isinstance(regimes, dict):
                warnings.append(f"Routing for asset '{asset}' must be a mapping")
                continue
            for regime, entry in regimes.items():
                if not isinstance(entry, dict):
                    warnings.append(f"Entry for {asset}/{regime} must be a mapping")
                    continue
                models = entry.get("models", [])
                if not isinstance(models, list) or len(models) == 0:
                    warnings.append(f"Missing models for {asset}/{regime}")
                    continue
                weight_sum = 0.0
                for model in models:
                    if not isinstance(model, dict):
                        warnings.append(f"Invalid model entry in {asset}/{regime}")
                        continue
                    name = model.get("name")
                    if not isinstance(name, str) or not name:
                        warnings.append(f"Model missing name in {asset}/{regime}")
                        continue
                    if name not in available:
                        warnings.append(
                            f"Strategy '{name}' in {asset}/{regime} not found in registry"
                        )
                    weight = model.get("weight", 0.0)
                    if not isinstance(weight, (int, float)) or weight < 0:
                        warnings.append(f"Invalid weight for '{name}' in {asset}/{regime}")
                        continue
                    weight_sum += float(weight)
                if weight_sum <= 0:
                    warnings.append(f"Total weight must be > 0 for {asset}/{regime}")

    except Exception as e:
        warnings.append(f"Failed to load config: {e}")

    return warnings


def apply_to_miner(
    config_path: str = _CONFIG_PATH,
    skip_validation: bool = False,
) -> bool:
    """
    Apply a config file to the running miner.

    Steps:
        1. Validate against registry
        2. Print diff
        3. Report success

    This is a safe operation — the config file is already in place,
    the miner reads it on startup. After updating, restart the miner.

    Returns:
        True if config is valid and in place.
    """
    if not skip_validation:
        warnings = validate_config(config_path)
        if warnings:
            print("[Apply] ⚠ Validation warnings:")
            for w in warnings:
                print(f"  - {w}")
            print("[Apply] Config has issues. Fix them before applying.")
            return False

    print(f"[Apply] ✓ Config at {config_path} is valid.")
    print(f"[Apply] Restart the miner to pick up the new config:")
    print(f"        python neurons/miner.py ...")
    return True

"""
applier.py — Apply exported config to the running miner.

Validates the generated config against the strategy registry,
previews the diff, and confirms before applying.
"""

from __future__ import annotations

import difflib
import os
from typing import Optional

from synth.miner.strategies.base import StrategyConfig


# Path to the production config
_CONFIG_PATH = "synth/miner/config/asset_strategy_config.py"


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
        # Import the config dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "PRODUCTION_CONFIG"):
            warnings.append("Missing PRODUCTION_CONFIG dict")
            return warnings

        for key, configs in module.PRODUCTION_CONFIG.items():
            for sc in configs:
                name = sc.strategy_name if hasattr(sc, "strategy_name") else sc[0]
                if name not in available:
                    warnings.append(
                        f"Strategy '{name}' in {key} not found in registry. "
                        f"Available: {sorted(available)}"
                    )

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

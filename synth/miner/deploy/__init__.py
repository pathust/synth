"""
deploy/ — Bridge between backtest results and production configuration.

Usage:
    from synth.miner.deploy import export_best_config, apply_to_miner

    # After running a backtest scan:
    export_best_config(scan_results, output_path="config/asset_strategy_config.py")
    apply_to_miner()
"""

from synth.miner.deploy.exporter import export_best_config
from synth.miner.deploy.applier import apply_to_miner, preview_config_diff

__all__ = ["export_best_config", "apply_to_miner", "preview_config_diff"]

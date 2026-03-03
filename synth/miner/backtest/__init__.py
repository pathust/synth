"""
backtest/ — Modular backtesting framework.

Usage:
    from synth.miner.backtest import BacktestRunner, GridSearchTuner

    runner = BacktestRunner()
    result = runner.run_single("garch_v2", "BTC", start_time, config)
"""

from synth.miner.backtest.metrics import METRICS, compute_crps_score
from synth.miner.backtest.runner import BacktestRunner
from synth.miner.backtest.tuner import GridSearchTuner
from synth.miner.backtest.report import BacktestReport

__all__ = [
    "METRICS",
    "compute_crps_score",
    "BacktestRunner",
    "GridSearchTuner",
    "BacktestReport",
]

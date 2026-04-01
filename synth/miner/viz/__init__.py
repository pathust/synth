"""
viz/ — Visualization module for strategy comparison and backtest results.

Usage:
    from synth.miner.viz import plot_strategy_comparison, plot_fan_chart

    fig = plot_strategy_comparison(scan_results, metric="CRPS")
    fig = plot_fan_chart(paths, real_prices=real_prices)
"""

from synth.miner.viz.strategy_compare import (
    plot_strategy_comparison,
    plot_score_distribution,
)
from synth.miner.viz.path_plotter import plot_fan_chart
from synth.miner.viz.backtest_report import generate_html_report

__all__ = [
    "plot_strategy_comparison",
    "plot_score_distribution",
    "plot_fan_chart",
    "generate_html_report",
]

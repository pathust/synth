"""
backtest/ — Modular backtesting framework.

Usage:
    from synth.miner.backtest import BacktestEngine, ExperimentConfig

    engine = BacktestEngine()
    results = engine.run(ExperimentConfig(assets=["BTC"], num_runs=10))
    engine.export_to_production(results)
"""

"""
backtest/ — Modular backtesting framework.
"""

# IMPORTANT:
# Do NOT import heavy deps at package import time.
# Some environments run lightweight scripts without numpy installed.
# Tools should import from submodules directly when needed.

__all__ = ["BacktestEngine", "ExperimentConfig"]


def __getattr__(name: str):
    if name in {"BacktestEngine", "ExperimentConfig"}:
        from synth.miner.backtest.engine import BacktestEngine, ExperimentConfig

        mapping = {
            "BacktestEngine": BacktestEngine,
            "ExperimentConfig": ExperimentConfig,
        }
        return mapping[name]
    raise AttributeError(name)

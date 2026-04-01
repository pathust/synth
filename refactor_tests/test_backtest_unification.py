from __future__ import annotations

import numpy as np

from synth.miner.backtest.engine import BacktestEngine
from synth.miner.backtest.framework import BacktestEngine as LegacyBacktestEngine


class _Cfg:
    time_increment = 300
    time_length = 3600
    scoring_intervals = [1, 6, 12]


class _Loader:
    def get_future_data(self, asset: str, start_time: str, time_length: int):
        return np.ones(13)


class _Eval:
    name = "DummyEval"

    def calculate(self, predictions, truth, time_increment, scoring_intervals):
        return 0.1


def test_engine_run_slots(monkeypatch):
    monkeypatch.setattr(
        "synth.miner.backtest.engine.simulate_crypto_price_paths",
        lambda **kwargs: np.ones((50, 13)),
    )
    monkeypatch.setattr(
        "synth.miner.backtest.engine.BacktestEngine._get_prompt_config",
        lambda self, asset: _Cfg(),
    )
    engine = BacktestEngine(data_loader=_Loader(), evaluators=[_Eval()])
    out = engine.run_slots(
        strategy_name="x",
        simulate_fn=lambda *args, **kwargs: np.ones((50, 13)),
        asset="BTC",
        eval_slots=["2026-03-31T00:00:00Z"],
        num_simulations=50,
    )
    assert len(out) == 1
    assert out[0]["metrics"]["DummyEval"] == 0.1


def test_legacy_framework_delegates(monkeypatch):
    monkeypatch.setattr(
        "synth.miner.backtest.engine.simulate_crypto_price_paths",
        lambda **kwargs: np.ones((50, 13)),
    )
    monkeypatch.setattr(
        "synth.miner.backtest.engine.BacktestEngine._get_prompt_config",
        lambda self, asset: _Cfg(),
    )
    legacy = LegacyBacktestEngine(data_loader=_Loader(), evaluators=[_Eval()])
    out = legacy.run(
        strategy_name="x",
        simulate_fn=lambda *args, **kwargs: np.ones((50, 13)),
        asset="BTC",
        eval_slots=["2026-03-31T00:00:00Z"],
        num_simulations=50,
    )
    assert len(out) == 1

from synth.miner.strategies.base import BaseStrategy
from synth.miner.strategies.registry import StrategyRegistry


class _S1(BaseStrategy):
    name = "test_strategy_1"

    def simulate(self, prices_dict, asset, time_increment, time_length, n_sims, seed=42, **kwargs):
        return None


class _S2(BaseStrategy):
    name = "test_strategy_2"

    def simulate(self, prices_dict, asset, time_increment, time_length, n_sims, seed=42, **kwargs):
        return None


def test_registry_instance_isolation():
    a = StrategyRegistry()
    b = StrategyRegistry()

    a.register(_S1())
    b.register(_S2())

    assert "test_strategy_1" in a.list_all()
    assert "test_strategy_2" not in a.list_all()
    assert "test_strategy_2" in b.list_all()
    assert "test_strategy_1" not in b.list_all()

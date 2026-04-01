import warnings

from synth.miner.backtest.engine import BacktestEngine as UnifiedBacktestEngine
from synth.miner.data.dataloader import UnifiedDataLoader


class BacktestEngine:
    def __init__(self, data_loader: UnifiedDataLoader, evaluators: list):
        warnings.warn(
            "synth.miner.backtest.framework.BacktestEngine is deprecated; use synth.miner.backtest.engine.BacktestEngine",
            DeprecationWarning,
            stacklevel=2,
        )
        self._engine = UnifiedBacktestEngine(
            data_loader=data_loader,
            evaluators=evaluators,
        )

    def run(
        self,
        strategy_name: str,
        simulate_fn,
        asset: str,
        eval_slots: list[str],
        num_simulations: int = 1000,
    ) -> list[dict]:
        return self._engine.run_slots(
            strategy_name=strategy_name,
            simulate_fn=simulate_fn,
            asset=asset,
            eval_slots=eval_slots,
            num_simulations=num_simulations,
        )

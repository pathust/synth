from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from synth.miner.backtest.regime_engine import (
    CaseKey,
    MarketTaxonomyRouter,
    PredictionBacktestEngine,
    RegimeEngineConfig,
)
from synth.miner.strategies.base import BaseStrategy
from synth.miner.strategies.registry import StrategyRegistry


class _StrategyA(BaseStrategy):
    name = "strategy_a"
    supported_regimes = ["bearish", "unknown", "mean_reverting", "market_open", "overnight"]
    param_grid = {"alpha": [1, 2, 3]}

    def simulate(
        self,
        prices_dict: dict,
        asset: str,
        time_increment: int,
        time_length: int,
        n_sims: int,
        seed: int = 42,
        **kwargs,
    ) -> np.ndarray:
        return np.ones((n_sims, 3))


class _StrategyB(BaseStrategy):
    name = "strategy_b"
    supported_regimes = ["bearish", "unknown", "mean_reverting", "market_open", "overnight"]
    param_grid = {"alpha": [1, 2, 3]}

    def simulate(
        self,
        prices_dict: dict,
        asset: str,
        time_increment: int,
        time_length: int,
        n_sims: int,
        seed: int = 42,
        **kwargs,
    ) -> np.ndarray:
        return np.ones((n_sims, 3))


class _FakeRunner:
    metric = "CRPS"

    def run_benchmark(
        self,
        strategy: BaseStrategy,
        asset: str,
        frequency: str = "low",
        num_runs: int = 3,
        num_sims: int = 100,
        seed: int = 42,
        window_days: int = 30,
        dates: list[datetime] | None = None,
        **strategy_kwargs,
    ) -> dict:
        alpha = float(strategy_kwargs.get("alpha", 2))
        base = 1.00 if strategy.name == "strategy_a" else 0.55
        score = base + abs(alpha - 2.0) * 0.1
        runs = len(dates) if dates is not None else num_runs
        return {
            "strategy": strategy.name,
            "asset": asset,
            "frequency": frequency,
            "metric": self.metric,
            "num_runs": runs,
            "successful_runs": runs,
            "avg_score": score,
            "median_score": score,
            "min_score": score,
            "max_score": score,
            "all_scores": [score for _ in range(max(runs, 1))],
            "details": [],
        }


def _build_history_df() -> pd.DataFrame:
    rows = []
    base = datetime(2026, 3, 3, 0, 0, tzinfo=timezone.utc)

    # BTC: first half low-vol/low-volume -> LOW/UNKNOWN,
    # then high-vol uptrend + downtrend -> HIGH/(BULLISH|BEARISH)
    for i in range(180):
        ts = base + timedelta(minutes=i)
        if i < 90:
            price = 100.0 + (i * 0.0001)
            volume = 1.0
        elif i < 135:
            price = 100.0 + (i - 90) * 0.30
            volume = 500.0
        else:
            price = 113.5 - (i - 135) * 0.35
            volume = 500.0
        rows.append(
            {
                "timestamp": ts,
                "asset": "BTC",
                "close": float(price),
                "volume": float(volume),
            }
        )

    # XAU
    for i in range(160):
        ts = base + timedelta(minutes=i)
        price = 1900.0 + np.sin(i / 6.0) * 2.0
        rows.append(
            {
                "timestamp": ts,
                "asset": "XAU",
                "close": float(price),
                "volume": 20.0,
            }
        )

    # AAPLX: market_open + overnight buckets
    open_base = datetime(2026, 3, 3, 14, 30, tzinfo=timezone.utc)  # 09:30 ET
    overnight_base = datetime(2026, 3, 3, 1, 0, tzinfo=timezone.utc)
    for i in range(80):
        rows.append(
            {
                "timestamp": open_base + timedelta(minutes=i),
                "asset": "AAPLX",
                "close": 200.0 + i * 0.02,
                "volume": 80.0,
            }
        )
        rows.append(
            {
                "timestamp": overnight_base + timedelta(minutes=i),
                "asset": "AAPLX",
                "close": 199.0 + i * 0.01,
                "volume": 15.0,
            }
        )
    return pd.DataFrame(rows)


def test_market_taxonomy_router_slices_cases():
    df = _build_history_df()
    router = MarketTaxonomyRouter(
        min_case_points=20,
        crypto_vol_quantile=0.5,
        crypto_volume_quantile=0.5,
    )
    cases = router.slice_dataframe(df)

    assert cases
    assert any(k.asset == "BTC" and k.market_type == "high" for k in cases)
    assert any(k.asset == "BTC" and k.market_type == "low" and k.regime == "unknown" for k in cases)
    assert any(k.asset == "XAU" and k.asset_class == "xau" for k in cases)
    assert any(k.asset == "AAPLX" and k.regime == "market_open" for k in cases)
    assert any(k.asset == "AAPLX" and k.regime == "overnight" for k in cases)


def test_prediction_backtest_engine_selects_best_strategy():
    registry = StrategyRegistry()
    registry.register(_StrategyA())
    registry.register(_StrategyB())

    engine = PredictionBacktestEngine(
        runner=_FakeRunner(),
        registry=registry,
        router=MarketTaxonomyRouter(min_case_points=1),
    )

    case_key = CaseKey(
        asset="BTC",
        asset_class="crypto",
        market_type="high",
        regime="bearish",
    )
    start = datetime(2026, 3, 1, tzinfo=timezone.utc)
    case_df = pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(60)],
            "asset": ["BTC"] * 60,
            "close": np.linspace(100.0, 105.0, 60),
            "volume": np.linspace(50.0, 80.0, 60),
        }
    )
    cfg = RegimeEngineConfig(
        timestamp_col="timestamp",
        asset_col="asset",
        price_col="close",
        volume_col="volume",
        min_split_points=9,
        max_tune_dates=12,
        max_eval_dates=8,
        max_param_combinations=10,
        split_mode="walk_forward",
        walk_forward_folds=3,
    )

    outcome = engine.evaluate_case(case_key, case_df, cfg)

    assert outcome.status == "SUCCESS"
    assert outcome.selected_strategy == "strategy_b"
    assert outcome.selected_params == {"alpha": 2}
    assert outcome.validation_score < 1.0
    assert outcome.test_score < 1.0
    assert outcome.case.path == "btc/high/bearish"
    assert outcome.folds_used >= 1


def test_regime_engine_runtime_export_builders(tmp_path):
    engine = PredictionBacktestEngine(
        runner=_FakeRunner(),
        registry=StrategyRegistry(),
        router=MarketTaxonomyRouter(min_case_points=1),
    )

    report = {
        "num_cases": 3,
        "results": [
            {
                "case": {
                    "asset": "BTC",
                    "asset_class": "crypto",
                    "market_type": "high",
                    "regime": "bearish",
                },
                "selected_strategy": "strategy_b",
                "selected_params": {"alpha": 2},
                "tuning_score": 0.5,
                "validation_score": 0.55,
                "test_score": 0.50,
                "train_points": 12,
                "validation_points": 6,
                "test_points": 6,
                "folds_used": 2,
                "status": "SUCCESS",
            },
            {
                "case": {
                    "asset": "BTC",
                    "asset_class": "crypto",
                    "market_type": "low",
                    "regime": "unknown",
                },
                "selected_strategy": "strategy_a",
                "selected_params": {"alpha": 2},
                "tuning_score": 0.9,
                "validation_score": 1.0,
                "test_score": 0.95,
                "train_points": 10,
                "validation_points": 5,
                "test_points": 5,
                "folds_used": 2,
                "status": "SUCCESS",
            },
            {
                "case": {
                    "asset": "XAU",
                    "asset_class": "xau",
                    "market_type": "spot",
                    "regime": "mean_reverting",
                },
                "selected_strategy": "strategy_b",
                "selected_params": {"alpha": 2},
                "tuning_score": 0.7,
                "validation_score": 0.8,
                "test_score": 0.75,
                "train_points": 10,
                "validation_points": 5,
                "test_points": 5,
                "folds_used": 1,
                "status": "SUCCESS",
            },
        ],
    }

    taxonomy = engine.build_taxonomy_routing(report)
    assert taxonomy["BTC"]["high"]["bearish"]["strategy"] == "strategy_b"
    assert taxonomy["BTC"]["low"]["unknown"]["strategy"] == "strategy_a"

    runtime = engine.build_runtime_routing(report, default_frequency="low")
    assert "BTC" in runtime and "XAU" in runtime
    assert "high" in runtime["BTC"] and "low" in runtime["BTC"]
    assert runtime["BTC"]["high"]["models"][0]["name"] == "strategy_b"
    assert runtime["XAU"]["low"]["models"][0]["name"] == "strategy_b"

    json_path = tmp_path / "report.json"
    taxonomy_path = tmp_path / "taxonomy.yaml"
    runtime_path = tmp_path / "runtime.yaml"
    engine.export_report_json(report, str(json_path))
    engine.export_taxonomy_yaml(report, str(taxonomy_path))
    engine.export_runtime_yaml(report, str(runtime_path), default_frequency="low")

    assert json_path.exists()
    assert taxonomy_path.exists()
    assert runtime_path.exists()

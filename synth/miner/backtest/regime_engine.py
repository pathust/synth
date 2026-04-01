"""
regime_engine.py — Prediction & Backtest engine driven by routing taxonomy.

Taxonomy path:
    [asset]/[type]/[regime]/[strategy]

Examples:
    btc/high/bearish/garch_v4
    xau/spot/mean_reverting/garch_v4_1
    aaplx/session/market_open/arima_equity
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synth.miner.backtest.runner import BacktestRunner
from synth.miner.backtest.tuner import GridSearchTuner
from synth.miner.strategies import StrategyRegistry
from synth.miner.strategies.base import BaseStrategy


_CRYPTO_ASSETS = {"BTC", "ETH", "SOL"}
_EQUITY_ASSETS = {"SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"}
_XAU_ASSETS = {"XAU"}

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass(frozen=True)
class CaseKey:
    asset: str
    asset_class: str
    market_type: str
    regime: str

    @property
    def path(self) -> str:
        return (
            f"{self.asset.lower()}/"
            f"{self.market_type.lower()}/"
            f"{self.regime.lower()}"
        )


@dataclass
class TimeSeriesSplit:
    train: list[datetime]
    validation: list[datetime]
    test: list[datetime]


@dataclass
class RegimeEngineConfig:
    timestamp_col: str = "timestamp"
    asset_col: str = "asset"
    price_col: str = "close"
    volume_col: str = "volume"
    frequency: str = "low"
    num_sims: int = 100
    seed: int = 42
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    min_case_points: int = 120
    min_split_points: int = 9
    max_tune_dates: int = 24
    max_eval_dates: int = 24
    max_param_combinations: Optional[int] = None
    split_mode: str = "single"  # "single" | "walk_forward"
    walk_forward_folds: int = 3
    walk_forward_min_train_ratio: float = 0.5


@dataclass
class CaseEvaluation:
    case: CaseKey
    selected_strategy: Optional[str]
    selected_params: dict
    tuning_score: float
    validation_score: float
    test_score: float
    candidate_count: int
    train_points: int
    validation_points: int
    test_points: int
    folds_used: int = 1
    status: str = "SUCCESS"
    reason: str = ""

    def to_dict(self) -> dict:
        out = asdict(self)
        out["case_path"] = self.case.path
        return out


class MarketTaxonomyRouter:
    """
    Slice a historical dataframe into routing cases using if-else taxonomy.
    """

    def __init__(
        self,
        crypto_vol_window: int = 48,
        crypto_volume_window: int = 48,
        crypto_regime_window: int = 24,
        crypto_vol_quantile: float = 0.6,
        crypto_volume_quantile: float = 0.6,
        crypto_momentum_floor: float = 0.0005,
        xau_er_window: int = 20,
        xau_er_threshold: float = 0.30,
        earnings_calendar: Optional[dict[str, set[str]]] = None,
        min_case_points: int = 120,
    ):
        self.crypto_vol_window = crypto_vol_window
        self.crypto_volume_window = crypto_volume_window
        self.crypto_regime_window = crypto_regime_window
        self.crypto_vol_quantile = crypto_vol_quantile
        self.crypto_volume_quantile = crypto_volume_quantile
        self.crypto_momentum_floor = crypto_momentum_floor
        self.xau_er_window = xau_er_window
        self.xau_er_threshold = xau_er_threshold
        self.earnings_calendar = earnings_calendar or {}
        self.min_case_points = min_case_points

    def slice_dataframe(
        self,
        df: pd.DataFrame,
        *,
        timestamp_col: str = "timestamp",
        asset_col: str = "asset",
        price_col: str = "close",
        volume_col: str = "volume",
    ) -> dict[CaseKey, pd.DataFrame]:
        if df is None or df.empty:
            return {}
        required = {timestamp_col, asset_col, price_col}
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        working = df.copy()
        working[timestamp_col] = pd.to_datetime(
            working[timestamp_col], utc=True, errors="coerce"
        )
        working = working.dropna(subset=[timestamp_col, asset_col, price_col])
        if working.empty:
            return {}
        working = working.sort_values([asset_col, timestamp_col])

        routed_groups: list[pd.DataFrame] = []
        for asset, group in working.groupby(asset_col, sort=False):
            routed_groups.append(
                self._route_asset_group(
                    asset=str(asset).upper(),
                    group=group,
                    timestamp_col=timestamp_col,
                    price_col=price_col,
                    volume_col=volume_col,
                )
            )
        if not routed_groups:
            return {}
        routed_df = pd.concat(routed_groups, ignore_index=True)

        cases: dict[CaseKey, pd.DataFrame] = {}
        internal_cols = ["_asset", "_asset_class", "_market_type", "_regime"]
        for values, chunk in routed_df.groupby(internal_cols, sort=False):
            asset, asset_class, market_type, regime = values
            if len(chunk) < self.min_case_points:
                continue
            key = CaseKey(
                asset=str(asset),
                asset_class=str(asset_class),
                market_type=str(market_type),
                regime=str(regime),
            )
            cases[key] = chunk.drop(columns=internal_cols).reset_index(drop=True)
        return cases

    def _route_asset_group(
        self,
        *,
        asset: str,
        group: pd.DataFrame,
        timestamp_col: str,
        price_col: str,
        volume_col: str,
    ) -> pd.DataFrame:
        frame = group.copy()
        asset_class = self._get_asset_class(asset)
        frame["_asset"] = asset
        frame["_asset_class"] = asset_class

        if asset_class == "crypto":
            market_type, regime = self._route_crypto(frame, price_col, volume_col)
        elif asset_class == "xau":
            market_type = pd.Series("spot", index=frame.index)
            regime = self._route_xau(frame, price_col)
        else:
            market_type = pd.Series("session", index=frame.index)
            regime = self._route_equity(asset, frame, timestamp_col)

        frame["_market_type"] = market_type.values
        frame["_regime"] = regime.values
        return frame

    def _get_asset_class(self, asset: str) -> str:
        if asset in _XAU_ASSETS:
            return "xau"
        if asset in _EQUITY_ASSETS:
            return "equity"
        if asset in _CRYPTO_ASSETS:
            return "crypto"
        return "crypto"

    def _route_crypto(
        self,
        frame: pd.DataFrame,
        price_col: str,
        volume_col: str,
    ) -> tuple[pd.Series, pd.Series]:
        prices = frame[price_col].astype(float)
        returns = np.log(prices.replace(0.0, np.nan)).diff().fillna(0.0)
        rolling_vol = returns.rolling(self.crypto_vol_window).std().fillna(0.0)
        vol_cut = float(rolling_vol.quantile(self.crypto_vol_quantile))
        if np.isnan(vol_cut):
            vol_cut = float(rolling_vol.median())
        high_mask = rolling_vol >= vol_cut

        if volume_col in frame.columns:
            rolling_volume = (
                frame[volume_col]
                .astype(float)
                .fillna(0.0)
                .rolling(self.crypto_volume_window)
                .mean()
                .fillna(0.0)
            )
            volume_cut = float(rolling_volume.quantile(self.crypto_volume_quantile))
            if np.isnan(volume_cut):
                volume_cut = float(rolling_volume.median())
            high_mask = high_mask | (rolling_volume >= volume_cut)

        market_type = pd.Series(
            np.where(high_mask, "high", "low"),
            index=frame.index,
        )

        momentum = returns.rolling(self.crypto_regime_window).mean().fillna(0.0)
        threshold = max(
            float(momentum.std()) * 0.5,
            self.crypto_momentum_floor,
        )
        regime = np.where(
            market_type == "low",
            "unknown",
            np.where(
                momentum > threshold,
                "bullish",
                np.where(momentum < -threshold, "bearish", "neutral"),
            ),
        )
        return market_type, pd.Series(regime, index=frame.index)

    def _route_xau(self, frame: pd.DataFrame, price_col: str) -> pd.Series:
        prices = frame[price_col].astype(float)
        change = prices.diff(self.xau_er_window).abs()
        volatility = prices.diff().abs().rolling(self.xau_er_window).sum()
        er = (change / volatility.replace(0.0, np.nan)).fillna(0.0)
        regime = np.where(
            er > self.xau_er_threshold,
            "trending",
            "mean_reverting",
        )
        return pd.Series(regime, index=frame.index)

    def _route_equity(
        self,
        asset: str,
        frame: pd.DataFrame,
        timestamp_col: str,
    ) -> pd.Series:
        ts_et = frame[timestamp_col].dt.tz_convert("America/New_York")
        weekday_mask = ts_et.dt.weekday < 5
        minutes = ts_et.dt.hour * 60 + ts_et.dt.minute
        market_open_mask = weekday_mask & (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)

        regime = np.where(market_open_mask, "market_open", "overnight")
        if asset in self.earnings_calendar and self.earnings_calendar[asset]:
            dates = set(ts_et.dt.strftime("%Y-%m-%d").tolist())
            earnings_dates = self.earnings_calendar[asset]
            overlap = dates & earnings_dates
            if overlap:
                day_str = ts_et.dt.strftime("%Y-%m-%d")
                earnings_mask = day_str.isin(earnings_dates)
                regime = np.where(earnings_mask, "earnings", regime)
        return pd.Series(regime, index=frame.index)


class PredictionBacktestEngine:
    """
    Run tuning + validation/test strategy selection per routing case.
    """

    def __init__(
        self,
        runner: Optional[BacktestRunner] = None,
        registry: Optional[StrategyRegistry] = None,
        router: Optional[MarketTaxonomyRouter] = None,
    ):
        self.runner = runner or BacktestRunner(metric="CRPS")
        if registry is None:
            registry = StrategyRegistry()
            registry.auto_discover()
        self.registry = registry
        self.tuner = GridSearchTuner(self.runner)
        self.router = router or MarketTaxonomyRouter()

    def slice_cases(
        self,
        df: pd.DataFrame,
        config: Optional[RegimeEngineConfig] = None,
    ) -> dict[CaseKey, pd.DataFrame]:
        cfg = config or RegimeEngineConfig()
        self.router.min_case_points = cfg.min_case_points
        return self.router.slice_dataframe(
            df,
            timestamp_col=cfg.timestamp_col,
            asset_col=cfg.asset_col,
            price_col=cfg.price_col,
            volume_col=cfg.volume_col,
        )

    def run(
        self,
        df: pd.DataFrame,
        config: Optional[RegimeEngineConfig] = None,
        strategy_names: Optional[list[str]] = None,
    ) -> dict:
        cfg = config or RegimeEngineConfig()
        case_map = self.slice_cases(df, cfg)

        results: list[dict] = []
        for case_key, case_df in case_map.items():
            outcome = self.evaluate_case(
                case_key,
                case_df,
                cfg,
                strategy_names=strategy_names,
            )
            results.append(outcome.to_dict())

        return {
            "num_cases": len(case_map),
            "results": results,
            "config": asdict(cfg),
        }

    def evaluate_case(
        self,
        case_key: CaseKey,
        case_df: pd.DataFrame,
        config: RegimeEngineConfig,
        strategy_names: Optional[list[str]] = None,
    ) -> CaseEvaluation:
        splits = self._build_splits(case_df, config)
        if not splits:
            return CaseEvaluation(
                case=case_key,
                selected_strategy=None,
                selected_params={},
                tuning_score=float("inf"),
                validation_score=float("inf"),
                test_score=float("inf"),
                candidate_count=0,
                train_points=0,
                validation_points=0,
                test_points=0,
                folds_used=0,
                status="SKIP",
                reason="insufficient_split_points",
            )

        fold_dates: list[tuple[list[datetime], list[datetime], list[datetime]]] = []
        train_count = 0
        val_count = 0
        test_count = 0
        for split in splits:
            train_dates = self._downsample(split.train, config.max_tune_dates)
            val_dates = self._downsample(split.validation, config.max_eval_dates)
            test_dates = self._downsample(split.test, config.max_eval_dates)
            if not train_dates or not val_dates or not test_dates:
                continue
            fold_dates.append((train_dates, val_dates, test_dates))
            train_count += len(train_dates)
            val_count += len(val_dates)
            test_count += len(test_dates)

        if not fold_dates:
            return CaseEvaluation(
                case=case_key,
                selected_strategy=None,
                selected_params={},
                tuning_score=float("inf"),
                validation_score=float("inf"),
                test_score=float("inf"),
                candidate_count=0,
                train_points=0,
                validation_points=0,
                test_points=0,
                folds_used=0,
                status="SKIP",
                reason="insufficient_split_points",
            )

        candidates = self._get_candidate_strategies(case_key, strategy_names)
        if not candidates:
            return CaseEvaluation(
                case=case_key,
                selected_strategy=None,
                selected_params={},
                tuning_score=float("inf"),
                validation_score=float("inf"),
                test_score=float("inf"),
                candidate_count=0,
                train_points=train_count,
                validation_points=val_count,
                test_points=test_count,
                folds_used=len(fold_dates),
                status="SKIP",
                reason="no_candidate_strategy",
            )

        best_strategy: Optional[BaseStrategy] = None
        best_params: dict = {}
        best_tune = float("inf")
        best_val = float("inf")

        # Use case-specific frequency if available, otherwise fallback to config
        effective_freq = case_key.market_type if case_key.market_type in ["high", "low"] else config.frequency

        for strategy in candidates:
            fold_tune_scores: list[float] = []
            fold_val_scores: list[float] = []
            fold_params: list[tuple[dict, float]] = []
            try:
                for fold_idx, (train_dates, val_dates, _) in enumerate(fold_dates):
                    tune_result = self.tuner.run(
                        strategy=strategy,
                        asset=case_key.asset,
                        frequency=effective_freq,
                        num_runs=len(train_dates),
                        num_sims=config.num_sims,
                        seed=config.seed + fold_idx,
                        dates=train_dates,
                        max_combinations=config.max_param_combinations,
                    )
                    params = tune_result.get("best_params") or {}
                    tune_score = float(
                        tune_result.get("best_score", float("inf"))
                    )
                    val_result = self.runner.run_benchmark(
                        strategy,
                        case_key.asset,
                        effective_freq,
                        num_runs=len(val_dates),
                        num_sims=config.num_sims,
                        seed=config.seed + 100 + fold_idx,
                        dates=val_dates,
                        **params,
                    )
                    val_score = float(val_result.get("avg_score", float("inf")))
                    if np.isfinite(tune_score):
                        fold_tune_scores.append(tune_score)
                    if np.isfinite(val_score):
                        fold_val_scores.append(val_score)
                        fold_params.append((params, val_score))
            except Exception as exc:
                print(f"[RegimeEngine] skip {strategy.name} in {case_key.path}: {exc}")
                continue

            if not fold_val_scores:
                continue

            val_score = float(np.mean(fold_val_scores))
            tune_score = (
                float(np.mean(fold_tune_scores))
                if fold_tune_scores
                else float("inf")
            )
            params = min(fold_params, key=lambda item: item[1])[0]

            if val_score < best_val:
                best_strategy = strategy
                best_params = params
                best_tune = tune_score
                best_val = val_score

        if best_strategy is None:
            return CaseEvaluation(
                case=case_key,
                selected_strategy=None,
                selected_params={},
                tuning_score=float("inf"),
                validation_score=float("inf"),
                test_score=float("inf"),
                candidate_count=len(candidates),
                train_points=train_count,
                validation_points=val_count,
                test_points=test_count,
                folds_used=len(fold_dates),
                status="SKIP",
                reason="all_candidates_failed",
            )

        fold_test_scores: list[float] = []
        for fold_idx, (_, _, test_dates) in enumerate(fold_dates):
            test_result = self.runner.run_benchmark(
                best_strategy,
                case_key.asset,
                effective_freq,
                num_runs=len(test_dates),
                num_sims=config.num_sims,
                seed=config.seed + 200 + fold_idx,
                dates=test_dates,
                **best_params,
            )
            score = float(test_result.get("avg_score", float("inf")))
            if np.isfinite(score):
                fold_test_scores.append(score)

        test_score = (
            float(np.mean(fold_test_scores))
            if fold_test_scores
            else float("inf")
        )

        return CaseEvaluation(
            case=case_key,
            selected_strategy=best_strategy.name,
            selected_params=best_params,
            tuning_score=best_tune,
            validation_score=best_val,
            test_score=test_score,
            candidate_count=len(candidates),
            train_points=train_count,
            validation_points=val_count,
            test_points=test_count,
            folds_used=len(fold_dates),
            status="SUCCESS",
        )

    def _get_candidate_strategies(
        self,
        case_key: CaseKey,
        strategy_names: Optional[list[str]],
    ) -> list[BaseStrategy]:
        strategies: list[BaseStrategy] = []
        if strategy_names:
            for name in strategy_names:
                try:
                    strategies.append(self.registry.get(name))
                except KeyError:
                    print(f"[RegimeEngine] strategy '{name}' not found, skipping")
        else:
            strategies = self.registry.get_for_asset(case_key.asset)

        filtered = [
            s for s in strategies if s.supports_regime(case_key.regime)
        ]
        filtered.sort(key=lambda s: s.name)
        return filtered

    def _build_splits(
        self,
        case_df: pd.DataFrame,
        config: RegimeEngineConfig,
    ) -> list[TimeSeriesSplit]:
        primary = self._time_series_split(case_df, config)
        if (
            config.split_mode != "walk_forward"
            or config.walk_forward_folds <= 1
        ):
            if not primary.train or not primary.validation or not primary.test:
                return []
            return [primary]

        ts = pd.to_datetime(
            case_df[config.timestamp_col],
            utc=True,
            errors="coerce",
        )
        ordered = [pd.Timestamp(t).to_pydatetime() for t in sorted(ts.dropna().unique())]
        n = len(ordered)
        if n < config.min_split_points:
            return []

        val_size = max(1, int(n * config.validation_ratio))
        test_size = max(1, int(n * (1.0 - config.train_ratio - config.validation_ratio)))
        min_train = max(
            int(n * config.walk_forward_min_train_ratio),
            max(1, int(n * config.train_ratio)),
        )
        max_train = n - (val_size + test_size)
        if max_train <= min_train:
            if not primary.train or not primary.validation or not primary.test:
                return []
            return [primary]

        folds = max(1, int(config.walk_forward_folds))
        if folds == 1:
            train_ends = [min_train]
        else:
            step = max(1, (max_train - min_train) // (folds - 1))
            train_ends = [min_train + (step * i) for i in range(folds)]

        splits: list[TimeSeriesSplit] = []
        seen_ranges: set[tuple[int, int, int]] = set()
        for train_end in train_ends:
            if train_end > max_train:
                train_end = max_train
            val_end = train_end + val_size
            test_end = val_end + test_size
            if test_end > n:
                continue
            signature = (train_end, val_end, test_end)
            if signature in seen_ranges:
                continue
            seen_ranges.add(signature)
            splits.append(
                TimeSeriesSplit(
                    train=ordered[:train_end],
                    validation=ordered[train_end:val_end],
                    test=ordered[val_end:test_end],
                )
            )

        if not splits and primary.train and primary.validation and primary.test:
            return [primary]
        return splits

    def _time_series_split(
        self,
        case_df: pd.DataFrame,
        config: RegimeEngineConfig,
    ) -> TimeSeriesSplit:
        ts = pd.to_datetime(case_df[config.timestamp_col], utc=True, errors="coerce")
        uniq_ts = sorted(ts.dropna().unique())
        ordered = [pd.Timestamp(t).to_pydatetime() for t in uniq_ts]
        n = len(ordered)
        if n < config.min_split_points:
            return TimeSeriesSplit(train=[], validation=[], test=[])

        train_end = int(n * config.train_ratio)
        val_end = int(n * (config.train_ratio + config.validation_ratio))
        train_end = max(1, min(train_end, n - 2))
        val_end = max(train_end + 1, min(val_end, n - 1))

        return TimeSeriesSplit(
            train=ordered[:train_end],
            validation=ordered[train_end:val_end],
            test=ordered[val_end:],
        )

    @staticmethod
    def _downsample(dates: list[datetime], max_points: int) -> list[datetime]:
        if max_points <= 0 or len(dates) <= max_points:
            return list(dates)
        idx = np.linspace(0, len(dates) - 1, max_points, dtype=int)
        return [dates[i] for i in idx]

    def build_taxonomy_routing(self, report: dict) -> dict:
        """
        Build nested taxonomy mapping:
            asset -> market_type -> regime -> selected strategy
        """
        routing: dict = {}
        for row in self._iter_success_rows(report):
            case = row.get("case", {})
            asset = str(case.get("asset", "")).upper()
            market_type = str(case.get("market_type", "")).lower()
            regime = str(case.get("regime", "")).lower()
            if not asset or not market_type or not regime:
                continue

            detail = {
                "strategy": row.get("selected_strategy"),
                "params": row.get("selected_params") or {},
                "scores": {
                    "tuning": row.get("tuning_score"),
                    "validation": row.get("validation_score"),
                    "test": row.get("test_score"),
                },
                "support": {
                    "train_points": row.get("train_points", 0),
                    "validation_points": row.get("validation_points", 0),
                    "test_points": row.get("test_points", 0),
                    "folds_used": row.get("folds_used", 1),
                },
            }
            routing.setdefault(asset, {}).setdefault(market_type, {})[regime] = detail
        return routing

    def build_runtime_routing(
        self,
        report: dict,
        *,
        default_frequency: str = "low",
        max_models_per_asset: int = 3,
    ) -> dict:
        """
        Build runtime routing in current strategies.yaml schema.
        """
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in self._iter_success_rows(report):
            case = row.get("case", {})
            asset = str(case.get("asset", "")).upper()
            asset_class = str(case.get("asset_class", "")).lower()
            market_type = str(case.get("market_type", "")).lower()
            if not asset:
                continue
            if asset_class == "crypto" and market_type in {"high", "low"}:
                frequency = market_type
            else:
                frequency = default_frequency

            grouped.setdefault((asset, frequency), []).append(row)

        runtime_routing: dict = {}
        for (asset, frequency), rows in grouped.items():
            per_strategy: dict[str, dict] = {}
            for row in rows:
                strategy_name = row.get("selected_strategy")
                if not strategy_name:
                    continue
                score = self._effective_score(row)
                support = float(
                    row.get("validation_points", 0) + row.get("test_points", 0)
                )
                if support <= 0:
                    support = 1.0
                contribution = support / max(score, 1e-9)

                item = per_strategy.setdefault(
                    strategy_name,
                    {
                        "name": strategy_name,
                        "raw_weight": 0.0,
                        "best_score": float("inf"),
                        "params": {},
                    },
                )
                item["raw_weight"] += contribution
                if score < item["best_score"]:
                    item["best_score"] = score
                    item["params"] = row.get("selected_params") or {}

            models = sorted(
                per_strategy.values(),
                key=lambda x: x["raw_weight"],
                reverse=True,
            )[: max(1, int(max_models_per_asset))]
            if not models:
                continue

            total_weight = sum(m["raw_weight"] for m in models)
            if total_weight <= 0:
                total_weight = float(len(models))
            normalized = []
            for item in models:
                normalized.append(
                    {
                        "name": item["name"],
                        "weight": round(float(item["raw_weight"] / total_weight), 4),
                        "params": item["params"],
                    }
                )

            runtime_routing.setdefault(asset, {})[frequency] = {
                "ensemble_method": "weighted_average",
                "models": normalized,
            }
        return runtime_routing

    def export_report_json(self, report: dict, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        return output_path

    def export_taxonomy_yaml(self, report: dict, output_path: str) -> str:
        payload = {
            "version": "1.0",
            "updated_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "routing": self.build_taxonomy_routing(report),
        }
        return self._write_yaml_or_json(payload, output_path)

    def export_runtime_yaml(
        self,
        report: dict,
        output_path: str,
        *,
        default_frequency: str = "low",
        max_models_per_asset: int = 3,
        version: str = "2.1",
        fallback_chain: Optional[dict] = None,
    ) -> str:
        if fallback_chain is None:
            fallback_chain = {
                "L1_timeout_ms": 600,
                "L2_model": "garch_v2",
                "L3_model": "garch_v4",
            }
        payload = {
            "version": version,
            "updated_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "fallback_chain": fallback_chain,
            "routing": self.build_runtime_routing(
                report,
                default_frequency=default_frequency,
                max_models_per_asset=max_models_per_asset,
            ),
        }
        return self._write_yaml_or_json(payload, output_path)

    def _write_yaml_or_json(self, payload: dict, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if yaml is not None and output_path.endswith((".yaml", ".yml")):
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    payload,
                    f,
                    sort_keys=False,
                    allow_unicode=False,
                )
            return output_path

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return output_path

    def _iter_success_rows(self, report: dict) -> list[dict]:
        rows = report.get("results", [])
        if not isinstance(rows, list):
            return []
        out: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("status") != "SUCCESS":
                continue
            if not row.get("selected_strategy"):
                continue
            out.append(row)
        return out

    @staticmethod
    def _effective_score(row: dict) -> float:
        test_score = float(row.get("test_score", float("inf")))
        val_score = float(row.get("validation_score", float("inf")))
        if np.isfinite(test_score):
            return test_score
        if np.isfinite(val_score):
            return val_score
        return float("inf")

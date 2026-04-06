"""
registry.py — Strategy registry with Asset × Regime routing.

Usage:
    registry = StrategyRegistry()
    registry.auto_discover()

    all_names = registry.list_all()
    strat = registry.get("garch_v2")
    compatible = registry.get_for_asset("BTC")

Mapping per ARCHITECTURE Section 3:
    Crypto (BTC, ETH, SOL): bull / high_vol / ranging
    Gold (XAU): mean_reverting / trending
    Equity (SPYX, NVDAX, TSLAX, AAPLX, GOOGLX): market_open / overnight / earnings
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Optional

from synth.miner.strategies.base import BaseStrategy, get_asset_type

class StrategyRegistry:
    _type_regime_map: dict[tuple[str, str], list[type]] = {}

    def __init__(self):
        self._strategies: dict[str, BaseStrategy] = {}

    @classmethod
    def register_asset_regime(
        cls,
        asset_type: str,
        regime: str,
        strategy_cls: type,
    ) -> None:
        key = (asset_type, regime)
        cls._type_regime_map.setdefault(key, []).append(strategy_cls)

    def register(
        self,
        strategy: BaseStrategy,
        asset_type: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> None:
        if not strategy.name:
            raise ValueError(
                f"Strategy {strategy.__class__.__name__} must have a non-empty 'name'."
            )
        if strategy.name in self._strategies:
            print(f"[Registry] WARNING: overwriting strategy '{strategy.name}'")

        self._strategies[strategy.name] = strategy

        if asset_type and regime:
            key = (asset_type, regime)
            self._type_regime_map.setdefault(key, []).append(strategy.__class__)

    def get(self, name: str) -> BaseStrategy:
        if name not in self._strategies:
            available = list(self._strategies.keys())
            raise KeyError(
                f"Strategy '{name}' not found. Available: {available}"
            )
        return self._strategies[name]

    def list_all(self) -> list[str]:
        return sorted(self._strategies.keys())

    def get_all(self) -> dict[str, BaseStrategy]:
        return dict(self._strategies)

    def get_for_asset(self, asset: str) -> list[BaseStrategy]:
        asset_type = get_asset_type(asset)
        result = []
        for strat in self._strategies.values():
            if strat.supports_asset(asset):
                result.append(strat)
        return result

    def get_for_asset_and_regime(
        self, asset: str, regime: str
    ) -> list[BaseStrategy]:
        result = []
        for strat in self._strategies.values():
            if strat.supports_asset(asset):
                strat_regimes = strat.supported_regimes
                if not strat_regimes or regime in strat_regimes:
                    result.append(strat)
        return result

    def get_by_type_and_regime(
        self, asset_type: str, regime: str
    ) -> list[BaseStrategy]:
        result = []
        for strat in self._strategies.values():
            if not strat.supported_asset_types or asset_type in strat.supported_asset_types:
                if not strat.supported_regimes or regime in strat.supported_regimes:
                    result.append(strat)
        return result

    def auto_discover(self) -> None:
        package_path = Path(__file__).parent
        package_name = "synth.miner.strategies"

        for _, module_name, is_pkg in pkgutil.iter_modules([str(package_path)]):
            if is_pkg or module_name.startswith("_"):
                continue
            if module_name in ("base", "registry"):
                continue

            try:
                module = importlib.import_module(f"{package_name}.{module_name}")
            except Exception as e:
                print(f"[Registry] Failed to import {module_name}: {e}")
                continue

            if hasattr(module, "strategy"):
                obj = getattr(module, "strategy")
                if isinstance(obj, BaseStrategy):
                    self.register(obj)
                    continue

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseStrategy)
                    and attr is not BaseStrategy
                ):
                    try:
                        instance = attr()
                        self.register(instance)
                    except Exception as e:
                        print(
                            f"[Registry] Failed to instantiate {attr_name}: {e}"
                        )

        print(
            f"[Registry] Auto-discovered {len(self._strategies)} strategies: "
            f"{self.list_all()}"
        )
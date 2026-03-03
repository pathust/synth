"""
registry.py — Strategy registry with auto-discovery.

Usage:
    registry = StrategyRegistry()
    registry.auto_discover()        # scans strategies/ for BaseStrategy subclasses
    
    all_names = registry.list_all()
    strat = registry.get("garch_v2")
    compatible = registry.get_for_asset("BTC", "low")
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Optional

from synth.miner.strategies.base import BaseStrategy


class StrategyRegistry:
    """Central registry for all simulation strategies."""

    def __init__(self):
        self._strategies: dict[str, BaseStrategy] = {}

    def register(self, strategy: BaseStrategy) -> None:
        """Register a strategy instance."""
        if not strategy.name:
            raise ValueError(
                f"Strategy {strategy.__class__.__name__} must have a non-empty 'name'."
            )
        if strategy.name in self._strategies:
            print(f"[Registry] WARNING: overwriting strategy '{strategy.name}'")
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> BaseStrategy:
        """Get a strategy by name. Raises KeyError if not found."""
        if name not in self._strategies:
            available = list(self._strategies.keys())
            raise KeyError(
                f"Strategy '{name}' not found. Available: {available}"
            )
        return self._strategies[name]

    def list_all(self) -> list[str]:
        """Return sorted list of all registered strategy names."""
        return sorted(self._strategies.keys())

    def get_all(self) -> dict[str, BaseStrategy]:
        """Return dict of all registered strategies."""
        return dict(self._strategies)

    def get_for_asset(
        self, asset: str, frequency: Optional[str] = None
    ) -> list[BaseStrategy]:
        """
        Return strategies compatible with the given asset and frequency.
        
        Args:
            asset: Asset name (e.g. "BTC")
            frequency: "high" or "low" (None = no filter)
        """
        result = []
        for strategy in self._strategies.values():
            if not strategy.supports_asset(asset):
                continue
            if frequency and not strategy.supports_frequency(frequency):
                continue
            result.append(strategy)
        return result

    def auto_discover(self) -> None:
        """
        Auto-discover and register all BaseStrategy subclasses in the
        synth.miner.strategies package.
        """
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

            # Look for a module-level `strategy` instance or STRATEGY variable
            if hasattr(module, "strategy"):
                obj = getattr(module, "strategy")
                if isinstance(obj, BaseStrategy):
                    self.register(obj)
                    continue

            # Fallback: find BaseStrategy subclasses and instantiate them
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

from typing import Type, Dict, List
from .base_strategy import BaseStrategy

class StrategyRegistry:
    """
    Registry for managing available strategies.
    Allows dynamic registration and instantiation without hardcoded imports.
    """
    _strategies: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy class."""
        def wrapper(strategy_cls: Type[BaseStrategy]):
            cls._strategies[name] = strategy_cls
            return strategy_cls
        return wrapper

    @classmethod
    def get(cls, name: str) -> Type[BaseStrategy]:
        """Retrieve a registered strategy class by name."""
        if name not in cls._strategies:
            raise ValueError(f"Strategy '{name}' not found in registry. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name]

    @classmethod
    def list_all(cls) -> List[str]:
        """List all available strategy names."""
        return list(cls._strategies.keys())

    @classmethod
    def create(cls, name: str, params: dict = None) -> BaseStrategy:
        """Factory method to instantiate a strategy."""
        strategy_cls = cls.get(name)
        return strategy_cls(name=name, params=params)

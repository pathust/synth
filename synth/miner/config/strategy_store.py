from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Union

from synth.miner.config.asset_strategy_config import (
    DEFAULT_FALLBACK_CHAIN as PY_DEFAULT_FALLBACK_CHAIN,
)
from synth.miner.config.asset_strategy_config import (
    get_strategy_list as get_strategy_list_from_python,
)
from synth.miner.strategies.base import StrategyConfig

try:
    import yaml
except Exception:
    yaml = None


def _prompt_type(time_length: int) -> str:
    return "high" if time_length == 3600 else "low"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


_RoutingKey = Union[tuple[str, str], tuple[str, str, str]]


def _parse_models(entry: Any) -> list[StrategyConfig]:
    if not isinstance(entry, dict):
        return []
    models = entry.get("models", [])
    if not isinstance(models, list):
        return []
    configs: list[StrategyConfig] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        weight = item.get("weight", 1.0)
        if not isinstance(weight, (int, float)):
            continue
        params = item.get("params", {})
        if not isinstance(params, dict):
            params = {}
        configs.append(
            StrategyConfig(
                strategy_name=name,
                weight=float(weight),
                params=params,
            )
        )
    return configs


def _normalize_routing(raw: Any) -> dict[_RoutingKey, list[StrategyConfig]]:
    out: dict[_RoutingKey, list[StrategyConfig]] = {}
    if not isinstance(raw, dict):
        return {}
    for asset, freq_map in raw.items():
        if not isinstance(asset, str) or not isinstance(freq_map, dict):
            continue
        for freq, freq_entry in freq_map.items():
            if not isinstance(freq, str) or not isinstance(freq_entry, dict):
                continue
            legacy_configs = _parse_models(freq_entry)
            if legacy_configs:
                out[(asset, freq)] = legacy_configs
                continue
            for market_regime, market_entry in freq_entry.items():
                if not isinstance(market_regime, str) or not isinstance(market_entry, dict):
                    continue
                configs = _parse_models(market_entry)
                if configs:
                    out[(asset, freq, market_regime)] = configs
    return out


def _normalize_fallback_chain(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return list(PY_DEFAULT_FALLBACK_CHAIN)
    keys = ["L2_model", "L3_model"]
    chain: list[str] = []
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value and value not in chain:
            chain.append(value)
    for name in PY_DEFAULT_FALLBACK_CHAIN:
        if name not in chain:
            chain.append(name)
    return chain


@dataclass
class StrategyStoreSnapshot:
    version: str
    updated_at: str
    routing: dict[_RoutingKey, list[StrategyConfig]]
    fallback_chain: list[str]


class StrategyStore:
    def __init__(self, path: Optional[str] = None):
        if path is None:
            root = os.path.dirname(__file__)
            path = os.path.join(root, "strategies.yaml")
        self._path = path
        self._last_mtime: Optional[float] = None
        self._snapshot: Optional[StrategyStoreSnapshot] = None

    @property
    def path(self) -> str:
        return self._path

    def reload_if_changed(self) -> StrategyStoreSnapshot:
        if not os.path.exists(self._path):
            return self._build_python_fallback_snapshot()
        mtime = os.path.getmtime(self._path)
        if self._snapshot is not None and self._last_mtime == mtime:
            return self._snapshot
        self._snapshot = self._load_from_yaml()
        self._last_mtime = mtime
        return self._snapshot

    def get_strategy_list(
        self,
        asset: str,
        time_length: int,
        market_regime: Optional[str] = None,
    ) -> list[StrategyConfig]:
        """
        Strategy Selector (ARCHITECTURE): resolve models from strategies.yaml.

        Lookup order:
        1) (asset, freq, market_regime) when regime-aware YAML is present
        2) (asset, freq) legacy flat routing
        3) For high-frequency requests, (asset, low, market_regime) when stocks
           only define regimes under low
        4) (asset, low) if high missing
        5) Python fallback from asset_strategy_config
        """
        snapshot = self.reload_if_changed()
        freq = _prompt_type(time_length)

        if market_regime:
            key3: _RoutingKey = (asset, freq, market_regime)
            if key3 in snapshot.routing:
                return list(snapshot.routing[key3])
            if freq == "high":
                key3_low = (asset, "low", market_regime)
                if key3_low in snapshot.routing:
                    return list(snapshot.routing[key3_low])

        key2: _RoutingKey = (asset, freq)
        if key2 in snapshot.routing:
            return list(snapshot.routing[key2])

        key_low: _RoutingKey = (asset, "low")
        if key_low in snapshot.routing:
            return list(snapshot.routing[key_low])

        return list(get_strategy_list_from_python(asset, time_length))

    def get_fallback_chain(self) -> list[str]:
        snapshot = self.reload_if_changed()
        return list(snapshot.fallback_chain)

    def _load_from_yaml(self) -> StrategyStoreSnapshot:
        if yaml is None:
            return self._build_python_fallback_snapshot()
        with open(self._path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if not isinstance(payload, dict):
            return self._build_python_fallback_snapshot()
        routing = _normalize_routing(payload.get("routing", {}))
        fallback_chain = _normalize_fallback_chain(payload.get("fallback_chain", {}))
        version = str(payload.get("version", "2.0"))
        updated_at = str(payload.get("updated_at", _now_iso()))
        if not routing:
            return self._build_python_fallback_snapshot()
        return StrategyStoreSnapshot(
            version=version,
            updated_at=updated_at,
            routing=routing,
            fallback_chain=fallback_chain,
        )

    def _build_python_fallback_snapshot(self) -> StrategyStoreSnapshot:
        assets = [
            "BTC",
            "ETH",
            "SOL",
            "XAU",
            "NVDAX",
            "TSLAX",
            "AAPLX",
            "GOOGLX",
            "SPYX",
        ]
        routing: dict[_RoutingKey, list[StrategyConfig]] = {}
        for asset in assets:
            routing[(asset, "high")] = list(get_strategy_list_from_python(asset, 3600))
            routing[(asset, "low")] = list(get_strategy_list_from_python(asset, 86400))
        return StrategyStoreSnapshot(
            version="2.0",
            updated_at=_now_iso(),
            routing=routing,
            fallback_chain=list(PY_DEFAULT_FALLBACK_CHAIN),
        )


_store_singleton: Optional[StrategyStore] = None


def get_strategy_store() -> StrategyStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = StrategyStore()
    return _store_singleton

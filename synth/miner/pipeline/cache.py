"""
cache.py — Two-tier caching layer for price data.

Tier 1: In-memory LRU cache (fast, volatile)
Tier 2: Disk file cache (persistent across runs)

Wraps any DataProvider to add transparent caching.
"""

import json
import os
import hashlib
from datetime import datetime
from functools import lru_cache
from typing import Optional

from synth.miner.pipeline.base import DataProvider


class CachedProvider(DataProvider):
    """
    Caching wrapper around any DataProvider.

    Checks in-memory cache first, then disk cache, then delegates to
    the wrapped provider.
    """

    def __init__(
        self,
        provider: DataProvider,
        cache_dir: str = "synth/miner/data/cache",
        memory_maxsize: int = 64,
    ):
        self._provider = provider
        self._cache_dir = cache_dir
        self._memory_maxsize = memory_maxsize
        self._memory: dict[str, dict[str, float]] = {}
        os.makedirs(cache_dir, exist_ok=True)

    @property
    def name(self) -> str:
        return f"Cached({self._provider.name})"

    def _cache_key(
        self, asset: str, start: Optional[datetime],
        end: Optional[datetime], resolution: str
    ) -> str:
        """Generate a deterministic cache key."""
        start_str = start.isoformat() if start else "none"
        end_str = end.isoformat() if end else "none"
        raw = f"{self._provider.name}:{asset}:{start_str}:{end_str}:{resolution}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _disk_path(self, key: str) -> str:
        return os.path.join(self._cache_dir, f"{key}.json")

    def _load_from_disk(self, key: str) -> Optional[dict[str, float]]:
        path = self._disk_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_to_disk(self, key: str, data: dict[str, float]) -> None:
        try:
            path = self._disk_path(key)
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[Cache] Failed to write disk cache: {e}")

    def fetch(
        self,
        asset: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        resolution: str = "5m",
    ) -> dict[str, float]:
        key = self._cache_key(asset, start, end, resolution)

        # Tier 1: memory
        if key in self._memory:
            return self._memory[key]

        # Tier 2: disk
        disk_data = self._load_from_disk(key)
        if disk_data is not None:
            self._memory[key] = disk_data
            return disk_data

        # Miss: delegate to wrapped provider
        data = self._provider.fetch(asset, start, end, resolution)
        if data:
            self._memory[key] = data
            # Only cache to disk if we got meaningful data
            if len(data) > 10:
                self._save_to_disk(key, data)

        return data

    def invalidate(self, asset: str = None) -> None:
        """Clear cache entries. If asset is None, clear all."""
        if asset is None:
            self._memory.clear()
            for f in os.listdir(self._cache_dir):
                if f.endswith(".json"):
                    os.remove(os.path.join(self._cache_dir, f))
        else:
            # Clear memory entries containing this asset
            keys_to_remove = [
                k for k in self._memory
                if asset in str(self._memory.get(k, {}))
            ]
            for k in keys_to_remove:
                del self._memory[k]

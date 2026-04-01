"""
pipeline/ — Data pipeline with provider abstraction and caching.

Usage:
    from synth.miner.pipeline import PriceLoader, MySQLProvider

    loader = PriceLoader(providers=[MySQLProvider()])
    prices = loader.load("BTC", before=datetime(...), resolution="5m")
"""

from synth.miner.pipeline.base import DataProvider
from synth.miner.pipeline.loader import PriceLoader

__all__ = ["DataProvider", "PriceLoader"]

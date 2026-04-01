"""
ensemble/ — Ensemble construction and outlier trimming.

Extracted from simulations_new_v3.py. Decouples ensemble logic from
the entry point so it can be tested and reused independently.
"""

from synth.miner.ensemble.builder import EnsembleBuilder
from synth.miner.ensemble.trimmer import OutlierTrimmer

__all__ = ["EnsembleBuilder", "OutlierTrimmer"]

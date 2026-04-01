"""
compute_score.py

Compatibility shim: older modules import `synth.miner.compute_score.cal_reward`.

The implementation lives in `_legacy/backtest_scripts/compute_score.py`.
"""

from __future__ import annotations

from synth.miner._legacy.backtest_scripts.compute_score import cal_reward

__all__ = ["cal_reward"]


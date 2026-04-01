"""
defaults.py — Module-level defaults for ensemble, backtesting, and tuning.

These are the "knobs" that were previously scattered across
simulations_new_v3.py and backtest/runner.py.
"""

# ── Ensemble Configuration ──────────────────────────────────────────
ENSEMBLE_TOP_N: int = 3
"""Maximum number of strategies to include in an ensemble."""

OVER_REQUEST_FACTOR: float = 1.10
"""Over-request factor for ensemble path generation (e.g., 1.10 = 10% extra)."""

# ── Outlier Trimming ────────────────────────────────────────────────
TRIM_LOWER: float = 1.0
"""Lower percentile for outlier trimming (e.g., 1.0 = trim bottom 1%)."""

TRIM_UPPER: float = 99.0
"""Upper percentile for outlier trimming (e.g., 99.0 = trim top 1%)."""

# ── Backtesting Defaults ────────────────────────────────────────────
DEFAULT_NUM_SIMS: int = 100
"""Default number of simulation paths per backtest run."""

DEFAULT_SEED: int = 42
"""Default random seed for reproducibility."""

DEFAULT_WINDOW_DAYS: int = 30
"""Default historical window for random date sampling in backtests."""

DEFAULT_LOOKBACK_DAYS: int = 45
"""Default number of days of historical price data to load."""

MAX_DATA_POINTS: int = 100_000
"""Maximum number of historical data points to pass to a strategy."""

# ── Frequency Mapping ───────────────────────────────────────────────
def get_prompt_type(time_length: int) -> str:
    """Map time_length in seconds to frequency label."""
    return "high" if time_length == 3600 else "low"

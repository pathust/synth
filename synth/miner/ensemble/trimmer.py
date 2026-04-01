"""
trimmer.py — Outlier trimming for ensemble paths.

Extracted from simulations_new_v3.py's inline trimming logic.
"""

import numpy as np
from typing import Optional

from synth.miner.config.defaults import TRIM_LOWER, TRIM_UPPER


class OutlierTrimmer:
    """
    Trim extreme simulation paths based on total return percentiles.

    Over-requests paths and then removes outliers to reduce variance
    explosion without losing diversity.
    """

    def __init__(
        self,
        lower_pct: float = TRIM_LOWER,
        upper_pct: float = TRIM_UPPER,
    ):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def trim(
        self,
        paths: np.ndarray,
        target_count: int,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Trim outlier paths and select exactly target_count paths.

        Args:
            paths: Array of shape (n_paths, steps+1).
            target_count: Desired number of output paths.
            seed: Random seed for selection.

        Returns:
            np.ndarray of shape (target_count, steps+1).
        """
        if paths.shape[0] == 0:
            return paths

        rng = np.random.RandomState(seed)

        # Calculate total returns per path
        start_prices = paths[:, 0]
        final_prices = paths[:, -1]
        safe_start = np.where(start_prices == 0, 1e-8, start_prices)
        returns = (final_prices - safe_start) / safe_start

        # Determine percentile bounds
        lower_bound = np.percentile(returns, self.lower_pct)
        upper_bound = np.percentile(returns, self.upper_pct)

        # Keep paths within bounds
        valid_mask = (returns >= lower_bound) & (returns <= upper_bound)
        trimmed = paths[valid_mask]

        if trimmed.shape[0] >= target_count:
            # Random sample without replacement
            indices = rng.choice(trimmed.shape[0], size=target_count, replace=False)
            return trimmed[indices]
        else:
            # Fallback: use untrimmed paths
            print(
                f"[Trimmer] Only {trimmed.shape[0]} paths after trim "
                f"(need {target_count}), using fallback"
            )
            if paths.shape[0] >= target_count:
                indices = rng.choice(paths.shape[0], size=target_count, replace=False)
                return paths[indices]
            else:
                indices = rng.choice(paths.shape[0], size=target_count, replace=True)
                return paths[indices]

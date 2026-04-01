"""
path_plotter.py — Fan chart visualization for simulation paths.

Plots simulation paths with percentile bands and optionally overlays
real prices for comparison.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def plot_fan_chart(
    paths: np.ndarray,
    real_prices: Optional[np.ndarray] = None,
    time_labels: Optional[list] = None,
    title: str = "Simulation Fan Chart",
    percentiles: list[int] = None,
    output_path: Optional[str] = None,
    asset: str = "",
    strategy_name: str = "",
):
    """
    Fan chart showing simulation paths with percentile bands.

    Args:
        paths: np.ndarray of shape (n_sims, steps+1).
        real_prices: Optional real price array for overlay.
        time_labels: Optional X-axis labels (timestamps, etc.).
        title: Chart title.
        percentiles: Percentile bands to draw (default: [5, 25, 50, 75, 95]).
        output_path: If provided, save figure to this path.
        asset: Asset name for labeling.
        strategy_name: Strategy name for labeling.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    fig, ax = plt.subplots(figsize=(14, 6))

    steps = paths.shape[1]
    x = np.arange(steps)

    if time_labels is not None:
        x_labels = time_labels[:steps]
    else:
        x_labels = x

    # Compute percentile bands
    pct_values = {}
    for p in percentiles:
        pct_values[p] = np.percentile(paths, p, axis=0)

    # Draw filled bands (outermost to innermost)
    band_colors = [
        ("#e8f4f8", 0.6),   # 5-95 band
        ("#b3d9e8", 0.6),   # 25-75 band
    ]

    if 5 in pct_values and 95 in pct_values:
        ax.fill_between(
            x, pct_values[5], pct_values[95],
            alpha=band_colors[0][1], color=band_colors[0][0],
            label="5th-95th percentile",
        )
    if 25 in pct_values and 75 in pct_values:
        ax.fill_between(
            x, pct_values[25], pct_values[75],
            alpha=band_colors[1][1], color=band_colors[1][0],
            label="25th-75th percentile",
        )

    # Draw median
    if 50 in pct_values:
        ax.plot(x, pct_values[50], color="#2196F3", linewidth=2, label="Median")

    # Draw sample paths (thin, semi-transparent)
    n_sample = min(20, paths.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(paths.shape[0], size=n_sample, replace=False)
    for idx in sample_idx:
        ax.plot(x, paths[idx], alpha=0.15, color="#666", linewidth=0.5)

    # Overlay real prices
    if real_prices is not None:
        n_real = min(len(real_prices), steps)
        ax.plot(
            x[:n_real], real_prices[:n_real],
            color="red", linewidth=2.5, label="Real Price",
            linestyle="--", zorder=10,
        )

    # Styling
    label_parts = [title]
    if asset:
        label_parts.append(f"({asset})")
    if strategy_name:
        label_parts.append(f"[{strategy_name}]")
    ax.set_title(" ".join(label_parts), fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Step", fontsize=10)
    ax.set_ylabel("Price", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved fan chart to {output_path}")

    return fig

"""
strategy_compare.py — Side-by-side strategy comparison charts.

Generates bar charts and box plots for comparing strategy performance
across assets and frequencies.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def plot_strategy_comparison(
    results: list[dict],
    metric: str = "CRPS",
    title: str = "Strategy Comparison",
    output_path: Optional[str] = None,
):
    """
    Bar chart comparing strategies side-by-side per asset.

    Args:
        results: List of benchmark result dicts (from BacktestRunner.scan_all).
            Each must have: strategy, asset, frequency, avg_score.
        metric: Metric name for labeling.
        title: Chart title.
        output_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    # Group by asset
    assets = sorted(set(r["asset"] for r in results))
    strategies = sorted(set(r["strategy"] for r in results))

    # Build matrix: asset × strategy → score
    scores = {}
    for r in results:
        key = (r["asset"], r["strategy"])
        scores[key] = r.get("avg_score", float("inf"))

    fig, axes = plt.subplots(1, len(assets), figsize=(6 * len(assets), 5), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))

    for i, asset in enumerate(assets):
        ax = axes[0, i]
        asset_scores = []
        asset_names = []
        for strat in strategies:
            score = scores.get((asset, strat))
            if score is not None and score < float("inf"):
                asset_scores.append(score)
                asset_names.append(strat)

        if asset_scores:
            bars = ax.barh(
                range(len(asset_names)), asset_scores,
                color=colors[:len(asset_names)],
                edgecolor="white", linewidth=0.5,
            )
            ax.set_yticks(range(len(asset_names)))
            ax.set_yticklabels(asset_names, fontsize=8)
            ax.set_xlabel(f"Avg {metric} (lower = better)", fontsize=9)
            ax.set_title(asset, fontsize=11, fontweight="bold")
            ax.invert_yaxis()

            # Annotate best
            best_idx = np.argmin(asset_scores)
            bars[best_idx].set_edgecolor("red")
            bars[best_idx].set_linewidth(2)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved comparison chart to {output_path}")

    return fig


def plot_score_distribution(
    results: list[dict],
    metric: str = "CRPS",
    title: str = "Score Distribution",
    output_path: Optional[str] = None,
):
    """
    Box plot of score distributions across backtest dates.

    Args:
        results: List of benchmark result dicts.
            Each must have: strategy, asset, all_scores.
        metric: Metric name for labeling.
        title: Chart title.
        output_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Collect all score distributions
    labels = []
    data = []
    for r in results:
        scores = r.get("all_scores", [])
        if not scores:
            continue
        label = f"{r['strategy']}\n({r['asset']})"
        labels.append(label)
        data.append(scores)

    if data:
        bp = ax.boxplot(
            data, labels=labels, patch_artist=True,
            boxprops=dict(facecolor="lightblue", edgecolor="navy"),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(color="navy"),
            capprops=dict(color="navy"),
        )
        ax.set_ylabel(f"{metric} Score", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved distribution chart to {output_path}")

    return fig

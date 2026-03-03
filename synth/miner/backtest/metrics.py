"""
metrics.py — Fitness/scoring functions for backtesting.

Extracted from backtest_framework.py to keep metrics independent.
"""

import numpy as np


def compute_crps_score(predictions: np.ndarray, real_prices: np.ndarray,
                       time_increment: int, scoring_intervals: dict) -> float:
    """
    Compute CRPS score using the validator's calculation.
    Returns the sum of all CRPS scores (lower is better).
    """
    from synth.validator.crps_calculation import calculate_crps_for_miner

    score, _ = calculate_crps_for_miner(
        predictions,
        real_prices,
        time_increment,
        scoring_intervals,
    )
    return float(score) if score != -1 and not np.isnan(score) else float("inf")


def compute_rmse(predictions: np.ndarray, real_prices: np.ndarray) -> float:
    """Root Mean Squared Error of the mean prediction path."""
    mean_pred = np.mean(predictions, axis=0)
    min_len = min(len(mean_pred), len(real_prices))
    if min_len == 0:
        return float("inf")
    return float(np.sqrt(np.mean((mean_pred[:min_len] - real_prices[:min_len]) ** 2)))


def compute_mae(predictions: np.ndarray, real_prices: np.ndarray) -> float:
    """Mean Absolute Error of the mean prediction path."""
    mean_pred = np.mean(predictions, axis=0)
    min_len = min(len(mean_pred), len(real_prices))
    if min_len == 0:
        return float("inf")
    return float(np.mean(np.abs(mean_pred[:min_len] - real_prices[:min_len])))


def compute_directional_accuracy(
    predictions: np.ndarray, real_prices: np.ndarray
) -> float:
    """
    Directional Accuracy error (1 - accuracy) so lower is better.
    """
    mean_pred = np.mean(predictions, axis=0)
    min_len = min(len(mean_pred), len(real_prices))
    if min_len <= 1:
        return 1.0

    pred_diff = np.diff(mean_pred[:min_len])
    real_diff = np.diff(real_prices[:min_len])

    correct = np.sum(np.sign(pred_diff) == np.sign(real_diff))
    acc = correct / len(real_diff)
    return float(1.0 - acc)


# Registry of metric functions (except CRPS which is special)
METRICS = {
    "CRPS": None,  # Handled via compute_crps_score
    "RMSE": compute_rmse,
    "MAE": compute_mae,
    "DIR_ACC": compute_directional_accuracy,
}

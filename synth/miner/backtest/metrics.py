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


def compute_var(predictions: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Value at Risk (VaR) of the prediction paths at a given confidence level.
    Lower (more negative) means higher risk.
    Calculates the percentage return VaR at the end of the simulation.
    """
    if len(predictions) == 0 or len(predictions[0]) < 2:
        return 0.0
    
    # Calculate returns for each path from start to end
    start_prices = predictions[:, 0]
    end_prices = predictions[:, -1]
    
    # Avoid division by zero
    valid_mask = start_prices > 0
    if not np.any(valid_mask):
        return 0.0
        
    returns = (end_prices[valid_mask] - start_prices[valid_mask]) / start_prices[valid_mask]
    var_percentile = (1 - confidence_level) * 100
    var_value = np.percentile(returns, var_percentile)
    
    return float(var_value)


def compute_es(predictions: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Expected Shortfall (ES) / Conditional VaR of the prediction paths.
    Expected return given that the return falls below the VaR threshold.
    """
    if len(predictions) == 0 or len(predictions[0]) < 2:
        return 0.0
        
    start_prices = predictions[:, 0]
    end_prices = predictions[:, -1]
    
    valid_mask = start_prices > 0
    if not np.any(valid_mask):
        return 0.0
        
    returns = (end_prices[valid_mask] - start_prices[valid_mask]) / start_prices[valid_mask]
    var_percentile = (1 - confidence_level) * 100
    var_value = np.percentile(returns, var_percentile)
    
    # Average of returns worse than VaR
    worse_returns = returns[returns <= var_value]
    if len(worse_returns) == 0:
        return float(var_value)
        
    return float(np.mean(worse_returns))


# Registry of metric functions (except CRPS which is special)
METRICS = {
    "CRPS": None,  # Handled via compute_crps_score
    "RMSE": compute_rmse,
    "MAE": compute_mae,
    "DIR_ACC": compute_directional_accuracy,
    "VaR_95": lambda p, r: compute_var(p, 0.95),
    "ES_95": lambda p, r: compute_es(p, 0.95),
}
